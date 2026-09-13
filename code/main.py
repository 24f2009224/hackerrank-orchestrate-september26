import os
import re
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Optional local OCR fallback
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

def log_transcript(entry: str, repo_root: str):
    """Appends execution events to log.txt in repo root as required by AGENTS.md."""
    log_path = os.path.join(repo_root, "log.txt")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {entry}\n")

def extract_amount_from_image(image_path: str) -> float:
    """Extracts missing financial event amounts from receipts/invoices."""
    if not os.path.exists(image_path) or not OCR_AVAILABLE:
        return 0.0
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        # Match common price/amount patterns like $1,250.00, INR 450.00, Total: 120.50
        matches = re.findall(r'(?:total|amount|due|paid|balance)?[^\d\n]*([\d,]+\.\d{2})', text, re.IGNORECASE)
        if matches:
            clean = matches[-1].replace(',', '')
            return float(clean)
    except Exception:
        pass
    return 0.0

def get_exchange_rate(from_curr: str, to_curr: str, date_str: str, rates_df: pd.DataFrame) -> float:
    """Converts foreign currencies using exact dated pairs."""
    if from_curr == to_curr or pd.isna(from_curr) or not from_curr:
        return 1.0
    direct = rates_df[(rates_df['from_currency'] == from_curr) & 
                      (rates_df['to_currency'] == to_curr) & 
                      (rates_df['rate_date'] == date_str)]
    if not direct.empty:
        return float(direct.iloc[0]['rate'])
    inverse = rates_df[(rates_df['from_currency'] == to_curr) & 
                       (rates_df['to_currency'] == from_curr) & 
                       (rates_df['rate_date'] == date_str)]
    if not inverse.empty:
        return 1.0 / float(inverse.iloc[0]['rate'])
    return 1.0

def reconcile_events(events_df, images_df, messages_df, media_dir):
    """Fills missing amounts and applies message-based modifications/cancellations."""
    df = events_df.copy()

    # Fill missing amounts from image evidence
    blank_mask = df['amount'].isna() | (df['amount'] == '') | (df['amount'] == 0)
    for idx in df[blank_mask].index:
        ev_id = df.loc[idx, 'event_id']
        img_match = images_df[images_df['related_event_id'] == ev_id]
        if not img_match.empty:
            img_filename = f"{img_match.iloc[0]['image_id']}.png"
            img_path = os.path.join(media_dir, img_filename)
            recovered_val = extract_amount_from_image(img_path)
            if recovered_val > 0:
                df.loc[idx, 'amount'] = recovered_val

    # Apply message overrides (cancellations, adjustments)
    df['is_cancelled'] = False
    for _, msg in messages_df.iterrows():
        rel_ev = msg.get('related_event_id')
        if pd.notna(rel_ev):
            txt = str(msg.get('message_text', '')).lower()
            if any(w in txt for w in ['cancel', 'cancelled', 'refunded', 'waived', 'terminated']):
                df.loc[df['event_id'] == rel_ev, 'is_cancelled'] = True

    return df

def simulate_90_day_balance(start_date_str: str, starting_balance: float, user_events: pd.DataFrame, 
                            home_currency: str, rates_df: pd.DataFrame, days: int = 90):
    """Runs day-by-day cash flow ledger for the 90-day window."""
    t0 = datetime.strptime(start_date_str, "%Y-%m-%d")
    date_index = [t0 + timedelta(days=i) for i in range(days + 1)]
    balance_series = pd.Series(index=date_index, data=0.0)

    daily_net = {d: 0.0 for d in date_index}

    for _, ev in user_events.iterrows():
        if ev.get('is_cancelled') or ev.get('status') in ['cancelled', 'failed', 'unrealized']:
            continue
        try:
            ev_d = datetime.strptime(ev['event_date'], "%Y-%m-%d")
        except Exception:
            continue

        if t0 <= ev_d <= date_index[-1]:
            raw_amt = float(ev.get('amount', 0.0) or 0.0)
            rate = get_exchange_rate(ev.get('currency', home_currency), home_currency, ev['event_date'], rates_df)
            net_amt = raw_amt * rate
            
            if ev.get('event_type') in ['income', 'salary', 'confirmed_credit']:
                daily_net[ev_d] += net_amt
            else:
                daily_net[ev_d] -= net_amt

    running_bal = float(starting_balance)
    for d in date_index:
        running_bal += daily_net[d]
        balance_series[d] = running_bal

    return balance_series

def process_request(req, profile, user_events, options_df, rates_df):
    req_id = req['request_id']
    req_date = req['request_date']
    req_amt = float(req['requested_amount'])
    deadline = req['desired_completion_date']
    allows_partial = str(req.get('allows_partial_payment', 'false')).lower() == 'true'

    home_curr = profile['home_currency']
    min_bal = float(profile['minimum_balance_to_keep'])
    curr_bal = float(profile['available_balance'])
    methods_allowed = [m.strip() for m in str(profile['payment_methods_user_will_consider']).split(',')]

    # Step 1: 90-day base trajectory
    timeline = simulate_90_day_balance(req_date, curr_bal, user_events, home_curr, rates_df, 90)
    headrooms = timeline - min_bal

    # amount_safe_to_pay today (cannot exceed requested_amount or cause balance < min_bal)
    amount_safe_to_pay = max(0.0, min(req_amt, float(headrooms.min())))

    # Step 2: Earliest full payment date
    t0 = datetime.strptime(req_date, "%Y-%m-%d")
    earliest_full_date = ""
    for i in range(91):
        cand_d = t0 + timedelta(days=i)
        sub_timeline = timeline.copy()
        sub_timeline.loc[cand_d:] -= req_amt
        if (sub_timeline.loc[cand_d:] >= min_bal).all():
            earliest_full_date = cand_d.strftime("%Y-%m-%d")
            break

    candidates = []

    # Candidate 1: Full Payment
    if 'full_payment' in methods_allowed and amount_safe_to_pay >= req_amt:
        candidates.append({
            'status': 'affordable_now',
            'method': 'full_payment',
            'plan': f"{req_date}:{int(req_amt) if req_amt.is_integer() else req_amt:.2f}",
            'spending': 'none',
            'completion': req_date,
            'total_cost': req_amt,
            'num_payments': 1,
            'opt_id': -1,
            'explanation': f"Full payment of {req_amt} {home_curr} is safe today without breaching the required minimum balance."
        })

    # Candidate 2: Partial Payment
    if ('partial_payment' in methods_allowed and allows_partial and 
        0 < amount_safe_to_pay < req_amt and earliest_full_date != "" and 
        earliest_full_date <= deadline):
        
        rem = req_amt - amount_safe_to_pay
        candidates.append({
            'status': 'affordable_with_plan',
            'method': 'partial_payment',
            'plan': f"{req_date}:{amount_safe_to_pay:.2f}|{earliest_full_date}:{rem:.2f}",
            'spending': 'none',
            'completion': earliest_full_date,
            'total_cost': req_amt,
            'num_payments': 2,
            'opt_id': -1,
            'explanation': f"Paying {amount_safe_to_pay:.2f} {home_curr} today and {rem:.2f} on {earliest_full_date} completes safely by {deadline}."
        })

    # Candidate 3: Installments
    if 'installments' in methods_allowed:
        rel_opts = options_df[options_df['request_id'] == req_id]
        for _, opt in rel_opts.iterrows():
            n = int(opt['number_of_installments'])
            interval = int(opt['interval_days'])
            inst_amt = float(opt['installment_amount'])
            tot_amt = float(opt['total_amount_payable'])
            opt_start = datetime.strptime(opt['start_date'], "%Y-%m-%d")

            sim = timeline.copy()
            feasible = True
            tokens = []
            final_d = opt_start
            
            for step in range(n):
                pay_d = opt_start + timedelta(days=step * interval)
                final_d = pay_d
                if pay_d > timeline.index[-1]:
                    feasible = False
                    break
                sim.loc[pay_d:] -= inst_amt
                tokens.append(f"{pay_d.strftime('%Y-%m-%d')}:{inst_amt:.2f}")

            if feasible and (sim >= min_bal).all():
                candidates.append({
                    'status': 'affordable_with_plan',
                    'method': 'installments',
                    'plan': "|".join(tokens),
                    'spending': 'none',
                    'completion': final_d.strftime("%Y-%m-%d"),
                    'total_cost': tot_amt,
                    'num_payments': n,
                    'opt_id': int(opt['payment_option_id']),
                    'explanation': f"Installment plan {opt['payment_option_id']} meets cash reserve constraints throughout all payments."
                })

    # Candidate 4: Wait
    if not candidates and earliest_full_date != "" and 'full_payment' in methods_allowed:
        candidates.append({
            'status': 'affordable_later',
            'method': 'wait',
            'plan': 'none',
            'spending': 'none',
            'completion': earliest_full_date,
            'total_cost': req_amt,
            'num_payments': 1,
            'opt_id': 999999,
            'explanation': f"Expense cannot be safely funded today. Wait until {earliest_full_date} for full payment."
        })

    # Strict Tie-Breaking Order
    if candidates:
        candidates.sort(key=lambda c: (
            c['completion'] > deadline,     # Rule 1: Deadline compliance
            c['spending'] != 'none',        # Rule 2: No spending changes
            c['total_cost'],                 # Rule 3: Minimize total amount
            c['completion'],                # Rule 4: Earliest start/completion
            c['num_payments'],              # Rule 5: Fewest payments
            c['opt_id']                     # Rule 6: Lowest payment_option_id
        ))
        best = candidates[0]
    else:
        best = {
            'status': 'not_affordable',
            'method': 'not_recommended',
            'plan': 'none',
            'spending': 'none',
            'explanation': "The request cannot be safely funded within the forecast period without violating minimum balance requirements."
        }

    return {
        'request_id': req_id,
        'amount_safe_to_pay': round(amount_safe_to_pay, 2),
        'affordability_status': best['status'],
        'recommended_payment_method': best['method'],
        'payment_plan': best['plan'],
        'earliest_date_for_full_payment': earliest_full_date if best['status'] == 'affordable_now' or earliest_full_date != "" else "",
        'spending_changes_needed': best['spending'],
        'decision_explanation': best['explanation']
    }

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(repo_root, "dataset")
    media_dir = os.path.join(dataset_dir, "media", "images")

    log_transcript("Starting agent execution for Buy or Wait challenge.", repo_root)

    requests_df = pd.read_csv(os.path.join(dataset_dir, "requests.csv"))
    profiles_df = pd.read_csv(os.path.join(dataset_dir, "financial_profiles.csv"))
    events_df = pd.read_csv(os.path.join(dataset_dir, "financial_events.csv"))
    rates_df = pd.read_csv(os.path.join(dataset_dir, "exchange_rates.csv"))
    options_df = pd.read_csv(os.path.join(dataset_dir, "request_payment_options.csv"))
    messages_df = pd.read_csv(os.path.join(dataset_dir, "messages.csv"))
    images_df = pd.read_csv(os.path.join(dataset_dir, "images.csv"))

    # Reconcile events
    resolved_events = reconcile_events(events_df, images_df, messages_df, media_dir)
    profiles_map = {row['user_id']: row for _, row in profiles_df.iterrows()}

    results = []
    for _, req in requests_df.iterrows():
        uid = req['user_id']
        u_prof = profiles_map[uid]
        u_events = resolved_events[resolved_events['user_id'] == uid]
        row_eval = process_request(req, u_prof, u_events, options_df, rates_df)
        results.append(row_eval)

    cols = [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation"
    ]
    out_df = pd.DataFrame(results)[cols]
    output_path = os.path.join(repo_root, "output.csv")
    out_df.to_csv(output_path, index=False)
    log_transcript(f"Generated output.csv with {len(out_df)} predictions.", repo_root)

if __name__ == "__main__":
    main()
