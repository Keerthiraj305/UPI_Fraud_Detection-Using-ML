"""
UPI Fraud Detection - Mode C Hybrid Prediction Engine

This module implements a HYBRID ML + RULES + FUSION system:
1. ML Ensemble: XGBoost + Isolation Forest + Meta learner
2. Rule Engine: LEGIT, SUSPICIOUS, FRAUD rules with probabilities
3. Weighted Fusion: Combines ML (60%) + Rules (40%)
4. Hard Overrides: Critical rules that override fusion
5. Soft Thresholding: Final classification with user thresholds

Architecture:
    run_ml() → ml_scores
    run_rules() → rule_output  
    fuse_results() → final_prob + decision
    Hard overrides → final adjustments

Author: UPI Fraud Detection Team
Version: 4.0 - Hybrid Fusion Engine
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

# ================================================================================
# MODULE CONFIGURATION
# ================================================================================

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USUAL_HOURS = list(range(6, 23))

# Fusion weights (ML vs Rules)
ML_WEIGHT = 0.60
RULE_WEIGHT = 0.40

# Soft thresholds for final classification
FRAUD_THRESHOLD = 0.90
SUSPICIOUS_THRESHOLD = 0.55

# ================================================================================
# MODEL LOADING (Safe fallback)
# ================================================================================

def _safe_load(path: Path) -> Optional[Any]:
    """Safely load joblib, return None on failure."""
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[predict_modec] WARNING: Failed to load {path.name}: {e}")
        return None

# Load ML models
xgb = _safe_load(ROOT / 'models' / 'xgb_raw.joblib')
iso = _safe_load(ROOT / 'models' / 'isolation_forest.joblib')
meta = _safe_load(ROOT / 'models' / 'meta_lr.joblib')
cal = _safe_load(ROOT / 'models' / 'final_calibrated_meta.joblib')

# Load features
try:
    features = pd.read_csv(ROOT / 'models' / 'feature_list.csv')['feature'].tolist()
except Exception:
    features = ['amount', 'hour', 'is_night', 'amount_over_user_avg', 'z_amount_user']
    print(f"[predict_modec] WARNING: Using fallback features")

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def build_input(tx: Dict[str, Any], profiles: pd.DataFrame) -> pd.DataFrame:
    """Build feature DataFrame from transaction dict and user profiles."""
    user_id = str(tx.get('user_id', '0'))
    user_profile = profiles[profiles['user_id'].astype(str) == user_id]
    
    feature_dict = {f: 0.0 for f in features}
    feature_dict['amount'] = float(tx.get('amount', 0.0))
    feature_dict['hour'] = int(tx.get('hour', 12))
    feature_dict['is_night'] = 1 if feature_dict['hour'] in [0,1,2,3,4,5] else 0
    
    if not user_profile.empty:
        user_row = user_profile.iloc[0]
        user_avg = float(user_row.get('avg_transaction_amount', 
                                       user_row.get('user_avg_amount_30d', 1.0)))
        user_std = float(user_row.get('user_std_amount_30d', 1.0))
        
        if 'amount_over_user_avg' in feature_dict:
            feature_dict['amount_over_user_avg'] = feature_dict['amount'] / (user_avg + 1e-6)
        if 'z_amount_user' in feature_dict:
            feature_dict['z_amount_user'] = (feature_dict['amount'] - user_avg) / (user_std + 1e-6)
    
    return pd.DataFrame([feature_dict])[features]


def get_user_profile_data(user_id: str, profiles: pd.DataFrame) -> Dict[str, Any]:
    """Extract user profile: user_avg, usual_hours, etc."""
    user_id = str(user_id)
    profile_data = {
        'user_avg': 500.0,
        'usual_hours': DEFAULT_USUAL_HOURS.copy(),
        'user_std': 100.0,
        'profile_found': False
    }
    
    if profiles.empty or user_id not in profiles['user_id'].astype(str).values:
        return profile_data
    
    user_row = profiles[profiles['user_id'].astype(str) == user_id].iloc[0]
    profile_data['profile_found'] = True
    profile_data['user_avg'] = float(user_row.get('avg_transaction_amount',
                                                    user_row.get('user_avg_amount_30d', 500.0)))
    profile_data['user_std'] = float(user_row.get('user_std_amount_30d', 100.0))
    
    usual_hours_str = user_row.get('usual_hours', None)
    if usual_hours_str and isinstance(usual_hours_str, str):
        try:
            profile_data['usual_hours'] = [int(h.strip()) for h in usual_hours_str.split(',')]
        except:
            pass
    
    return profile_data


def is_receiver_known(user_id: str, receiver_id: str, root: Path, threshold: int = 10) -> Tuple[bool, int]:
    """Check if receiver is known (>threshold past transactions)."""
    history_file = root / 'data' / 'transaction_history.csv'
    if not history_file.exists():
        return False, 0
    
    try:
        history = pd.read_csv(history_file)
        count = len(history[
            (history['user_id'].astype(str) == str(user_id)) & 
            (history['receiver_id'].astype(str) == str(receiver_id))
        ])
        return count > threshold, count
    except Exception:
        return False, 0


# ================================================================================
# STEP 1: ML ENSEMBLE
# ================================================================================

def run_ml(X: pd.DataFrame) -> Dict[str, Any]:
    """
    Run ML ensemble: XGBoost + IsolationForest + Meta.
    
    Returns:
        ml_scores dict with xgb_prob, iso_score, meta_prob, final_ml_prob
    """
    ml_scores = {
        'xgb_prob': None,
        'iso_score': None,
        'meta_prob': None,
        'calibrated_prob': None,
        'final_ml_prob': 0.5,  # default fallback
        'models_used': []
    }
    
    # XGBoost
    if xgb is not None:
        try:
            ml_scores['xgb_prob'] = float(xgb.predict_proba(X)[0, 1])
            ml_scores['models_used'].append('XGBoost')
        except Exception as e:
            print(f"[ML] XGBoost failed: {e}")
            ml_scores['xgb_prob'] = 0.5
    
    # Isolation Forest
    if iso is not None:
        try:
            ml_scores['iso_score'] = float(-iso.decision_function(X)[0])
            ml_scores['models_used'].append('IsolationForest')
        except Exception as e:
            print(f"[ML] IsolationForest failed: {e}")
            ml_scores['iso_score'] = 0.0
    
    # Meta ensemble (prefer raw meta over calibrated)
    xgb_p = ml_scores['xgb_prob']
    iso_s = ml_scores['iso_score']
    
    if meta is not None and xgb_p is not None and iso_s is not None:
        try:
            meta_X = np.array([[xgb_p, iso_s]])
            ml_scores['meta_prob'] = float(meta.predict_proba(meta_X)[0, 1])
            ml_scores['final_ml_prob'] = ml_scores['meta_prob']
            ml_scores['models_used'].append('Meta-LR')
        except Exception:
            ml_scores['final_ml_prob'] = xgb_p if xgb_p else 0.5
    elif cal is not None and xgb_p is not None and iso_s is not None:
        try:
            meta_X = np.array([[xgb_p, iso_s]])
            ml_scores['calibrated_prob'] = float(cal.predict_proba(meta_X)[0, 1])
            ml_scores['final_ml_prob'] = ml_scores['calibrated_prob']
            ml_scores['models_used'].append('Calibrated-Meta')
        except Exception:
            ml_scores['final_ml_prob'] = xgb_p if xgb_p else 0.5
    elif xgb_p is not None:
        ml_scores['final_ml_prob'] = xgb_p
    
    return ml_scores


# ================================================================================
# STEP 2: RULE ENGINE (with probability scores)
# ================================================================================

def run_rules(
    tx: Dict[str, Any],
    profile_data: Dict[str, Any],
    root: Path
) -> Dict[str, Any]:
    """
    Run complete rule engine with probability scoring.
    
    Rules produce probabilities:
    - FRAUD rules: 0.90-0.99
    - SUSPICIOUS rules: 0.40-0.70
    - LEGIT rules: 0.00-0.05
    
    Returns:
        {
            "decision": "LEGIT/SUSPICIOUS/FRAUD",
            "rule_prob": float,
            "flags": [list of triggered rules],
            "explanation": str
        }
    """
    amount = float(tx.get('amount', 0.0))
    hour = int(tx.get('hour', 12))
    user_id = str(tx.get('user_id', '0'))
    receiver_id = tx.get('receiver_id', None)
    
    user_avg = profile_data['user_avg']
    usual_hours = profile_data['usual_hours']
    
    triggered_flags = []
    rule_output = {
        'decision': 'SUSPICIOUS',  # default
        'rule_prob': 0.5,
        'flags': [],
        'explanation': 'No specific rules triggered',
        'tier': None
    }
    
    # ============================================================================
    # TIER 1: LEGIT RULES (prob 0.00-0.05)
    # ============================================================================
    
    # LEGIT Rule 1: Amount < 1000
    if amount < 1000:
        return {
            'decision': 'LEGIT',
            'rule_prob': 0.02,
            'flags': ['amount_below_1000'],
            'explanation': f'Amount ₹{amount:.2f} < ₹1000 threshold',
            'tier': 'LEGIT',
            'rule_weight': 1.0  # critical override
        }
    
    # LEGIT Rule 2: Amount < user_avg
    if amount < user_avg:
        return {
            'decision': 'LEGIT',
            'rule_prob': 0.03,
            'flags': ['amount_below_user_avg'],
            'explanation': f'Amount ₹{amount:.2f} < user avg ₹{user_avg:.2f}',
            'tier': 'LEGIT',
            'rule_weight': 1.0
        }
    
    # LEGIT Rule 3: Known receiver
    if receiver_id:
        is_known, count = is_receiver_known(user_id, receiver_id, root)
        if is_known:
            return {
                'decision': 'LEGIT',
                'rule_prob': 0.01,
                'flags': ['known_receiver'],
                'explanation': f'Trusted receiver (sent {count} times)',
                'tier': 'LEGIT',
                'rule_weight': 1.0,
                'receiver_count': count
            }
    
    # LEGIT Rule 4: Safe normal transaction
    is_safe = (
        amount < user_avg * 1.2 and
        hour in usual_hours and
        not tx.get('is_new_receiver', False) and
        not tx.get('device_changed', False) and
        tx.get('receiver_risk_score', 0.0) < 0.6 and
        tx.get('velocity_1h', 0) <= 3
    )
    if is_safe:
        return {
            'decision': 'LEGIT',
            'rule_prob': 0.05,
            'flags': ['safe_normal'],
            'explanation': 'Normal transaction, no risk indicators',
            'tier': 'LEGIT',
            'rule_weight': 0.8
        }
    
    # ============================================================================
    # TIER 2: FRAUD RULES (prob 0.90-0.99)
    # ============================================================================
    
    if amount > user_avg * 2:
        fraud_factors = []
        
        sender_loc = tx.get('sender_location', None)
        receiver_loc = tx.get('receiver_location', None)
        
        if hour not in usual_hours:
            fraud_factors.append('unusual_time')
        if sender_loc and receiver_loc and sender_loc != receiver_loc:
            fraud_factors.append('location_mismatch')
        if tx.get('is_new_receiver', False):
            fraud_factors.append('new_receiver')
        if tx.get('device_changed', False):
            fraud_factors.append('device_changed')
        if tx.get('velocity_1h', 0) > 3:
            fraud_factors.append('high_velocity')
        if tx.get('receiver_risk_score', 0.0) > 0.85:
            fraud_factors.append('receiver_high_risk')
        
        # FRAUD Rule: 2+ risk factors
        if len(fraud_factors) >= 2:
            # More factors → higher probability
            fraud_prob = min(0.99, 0.90 + (len(fraud_factors) * 0.02))
            return {
                'decision': 'FRAUD',
                'rule_prob': fraud_prob,
                'flags': fraud_factors,
                'explanation': f'{len(fraud_factors)} risk factors with large amount',
                'tier': 'FRAUD',
                'rule_weight': 1.0,  # critical override
                'amount_multiplier': round(amount / user_avg, 2)
            }
    
    # ============================================================================
    # TIER 3: SUSPICIOUS RULES (prob 0.40-0.70)
    # ============================================================================
    
    if amount > user_avg:
        suspicious_flags = []
        
        sender_loc = tx.get('sender_location', None)
        receiver_loc = tx.get('receiver_location', None)
        
        if sender_loc and receiver_loc and sender_loc != receiver_loc:
            suspicious_flags.append('location_mismatch')
        if hour not in usual_hours:
            suspicious_flags.append('unusual_time')
        if tx.get('is_new_receiver', False):
            suspicious_flags.append('new_receiver')
        if tx.get('device_changed', False):
            suspicious_flags.append('device_changed')
        if amount > user_avg * 1.2:
            suspicious_flags.append('amount_spike')
        
        receiver_risk = tx.get('receiver_risk_score', 0.0)
        if 0.6 <= receiver_risk <= 0.85:
            suspicious_flags.append('receiver_risk_moderate')
        if tx.get('velocity_1h', 0) > 3:
            suspicious_flags.append('high_velocity')
        
        # SUSPICIOUS Rule: 1+ flags
        if suspicious_flags:
            # More flags → higher suspicion
            susp_prob = min(0.70, 0.40 + (len(suspicious_flags) * 0.05))
            return {
                'decision': 'SUSPICIOUS',
                'rule_prob': susp_prob,
                'flags': suspicious_flags,
                'explanation': f'{len(suspicious_flags)} suspicious indicator(s)',
                'tier': 'SUSPICIOUS',
                'rule_weight': 0.6
            }
    
    # No strong rules triggered - neutral
    return {
        'decision': 'NEUTRAL',
        'rule_prob': 0.5,
        'flags': [],
        'explanation': 'No specific rules triggered',
        'tier': None,
        'rule_weight': 0.3
    }


# ================================================================================
# STEP 3: WEIGHTED FUSION
# ================================================================================

def fuse_results(
    ml_scores: Dict[str, Any],
    rule_output: Dict[str, Any],
    user_threshold: float,
    amount: float
) -> Dict[str, Any]:
    """
    Fuse ML and Rule probabilities with weighted average.
    
    Formula:
        final_prob = ML_WEIGHT * ml_prob + RULE_WEIGHT * rule_prob
    
    Then apply:
    - Hard overrides for critical rules
    - Amount-based sanity caps
    - Soft thresholding
    
    Returns:
        {
            "final_prob": float,
            "final_decision": str,
            "fusion_logic": {...}
        }
    """
    ml_prob = ml_scores['final_ml_prob']
    rule_prob = rule_output['rule_prob']
    
    # Base fusion
    raw_fusion_prob = (ML_WEIGHT * ml_prob) + (RULE_WEIGHT * rule_prob)
    
    fusion_info = {
        'ml_weight': ML_WEIGHT,
        'rule_weight': RULE_WEIGHT,
        'ml_contribution': ML_WEIGHT * ml_prob,
        'rule_contribution': RULE_WEIGHT * rule_prob,
        'raw_fusion_prob': raw_fusion_prob,
        'overridden': False,
        'override_reason': None
    }
    
    final_prob = raw_fusion_prob
    final_decision = 'SUSPICIOUS'  # default
    
    # ============================================================================
    # STEP 4: HARD OVERRIDES (Critical rules bypass fusion)
    # ============================================================================
    
    rule_tier = rule_output.get('tier', None)
    rule_weight = rule_output.get('rule_weight', 0.5)
    
    # Critical LEGIT overrides
    if rule_tier == 'LEGIT' and rule_weight >= 0.8:
        final_prob = 0.01
        final_decision = 'LEGIT'
        fusion_info['overridden'] = True
        fusion_info['override_reason'] = f"Critical LEGIT rule: {rule_output['flags']}"
    
    # Critical FRAUD overrides
    elif rule_tier == 'FRAUD' and rule_weight >= 1.0:
        final_prob = max(0.95, raw_fusion_prob)
        final_decision = 'FRAUD'
        fusion_info['overridden'] = True
        fusion_info['override_reason'] = f"Critical FRAUD rule: {rule_output['flags']}"
    
    # No override - use fusion
    else:
        # Apply amount-based sanity caps
        if amount < 100.0:
            final_prob = min(final_prob, 0.6)
            fusion_info['amount_cap'] = 0.6
        elif amount < 500.0:
            final_prob = min(final_prob, 0.85)
            fusion_info['amount_cap'] = 0.85
        
        # ========================================================================
        # STEP 5: SOFT THRESHOLDING
        # ========================================================================
        
        if final_prob >= FRAUD_THRESHOLD:
            final_decision = 'FRAUD'
        elif final_prob >= SUSPICIOUS_THRESHOLD:
            final_decision = 'SUSPICIOUS'
        else:
            final_decision = 'LEGIT'
        
        # User-specific threshold adjustment
        if final_prob >= (user_threshold + 0.10):
            if final_decision == 'LEGIT':
                final_decision = 'SUSPICIOUS'
                fusion_info['user_threshold_boost'] = True
    
    fusion_info['final_prob'] = final_prob
    fusion_info['final_decision'] = final_decision
    fusion_info['fraud_threshold'] = FRAUD_THRESHOLD
    fusion_info['suspicious_threshold'] = SUSPICIOUS_THRESHOLD
    fusion_info['user_threshold'] = user_threshold
    
    return {
        'final_prob': final_prob,
        'final_decision': final_decision,
        'fusion_logic': fusion_info
    }


# ================================================================================
# MAIN PREDICTION FUNCTION
# ================================================================================

def predict_single(tx: Dict[str, Any], root: Path = ROOT) -> Tuple[float, str, Dict[str, Any]]:
    """
    Hybrid ML + Rules + Fusion prediction engine.
    
    Process:
    1. Load user profile
    2. Run ML ensemble → ml_scores
    3. Run rule engine → rule_output
    4. Fuse results (weighted average)
    5. Apply hard overrides & soft thresholds
    6. Return comprehensive output
    
    Args:
        tx: Transaction dict with keys:
            - user_id, amount, hour (required)
            - receiver_id, sender_location, receiver_location (optional)
            - is_new_receiver, device_changed, velocity_1h, receiver_risk_score (optional)
        root: Root directory
        
    Returns:
        (final_prob, final_decision, details_dict)
    """
    # ============================================================================
    # LOAD DATA
    # ============================================================================
    
    profiles_file = root / 'data' / 'user_profiles_modec.csv'
    profiles = pd.read_csv(profiles_file) if profiles_file.exists() else pd.DataFrame()
    
    thresholds_file = root / 'data' / 'user_thresholds_modec.csv'
    thresholds = pd.read_csv(thresholds_file) if thresholds_file.exists() else pd.DataFrame()
    
    user_id = str(tx.get('user_id', '0'))
    amount = float(tx.get('amount', 0.0))
    hour = int(tx.get('hour', 12))
    
    profile_data = get_user_profile_data(user_id, profiles)
    
    user_threshold = 0.1
    if not thresholds.empty and user_id in thresholds['user_id'].astype(str).values:
        thresh_row = thresholds[thresholds['user_id'].astype(str) == user_id]
        if 'user_threshold' in thresh_row.columns:
            user_threshold = float(thresh_row['user_threshold'].iloc[0])
        elif 'threshold' in thresh_row.columns:
            user_threshold = float(thresh_row['threshold'].iloc[0])
    
    # ============================================================================
    # STEP 1: RUN ML ENSEMBLE
    # ============================================================================
    
    X = build_input(tx, profiles)
    ml_scores = run_ml(X)
    
    # ============================================================================
    # STEP 2: RUN RULE ENGINE
    # ============================================================================
    
    rule_output = run_rules(tx, profile_data, root)
    
    # ============================================================================
    # STEP 3-5: FUSE RESULTS (with overrides & thresholding)
    # ============================================================================
    
    fusion_result = fuse_results(ml_scores, rule_output, user_threshold, amount)
    
    # ============================================================================
    # STEP 6: BUILD FINAL OUTPUT
    # ============================================================================
    
    final_output = {
        'final_prob': fusion_result['final_prob'],
        'final_decision': fusion_result['final_decision'],
        
        'ml_scores': ml_scores,
        
        'rule_output': rule_output,
        
        'fusion_logic': fusion_result['fusion_logic'],
        
        'user_profile': {
            'user_avg': profile_data['user_avg'],
            'usual_hours': profile_data['usual_hours'],
            'threshold': user_threshold,
            'profile_found': profile_data['profile_found']
        },
        
        'transaction': {
            'user_id': user_id,
            'amount': amount,
            'hour': hour
        },
        
        'engine_version': '4.0-Hybrid-Fusion'
    }
    
    return (
        fusion_result['final_prob'],
        fusion_result['final_decision'],
        final_output
    )
