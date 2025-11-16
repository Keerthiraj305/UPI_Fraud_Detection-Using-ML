"""
Demo Predictions - Showcase UPI Fraud Detection System

Run this to demonstrate the hybrid ML + Rules system with realistic scenarios.

Usage:
    python3 UPI_Fraud_ModeC_Professional_UI/demo_predictions.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from UPI_Fraud_ModeC_Professional_UI.src.predict_modec import predict_single


def print_prediction(name: str, tx: dict):
    """Pretty print a prediction result."""
    print("\n" + "=" * 80)
    print(f"SCENARIO: {name}")
    print("=" * 80)
    print(f"Transaction: ₹{tx['amount']:.2f} at {tx['hour']}:00")
    print(f"User: {tx.get('user_id', 'N/A')} → Receiver: {tx.get('receiver_id', 'N/A')}")
    if tx.get('is_new_receiver'):
        print("⚠️  New receiver (first time)")
    if tx.get('device_changed'):
        print("⚠️  Device changed")
    if tx.get('velocity_1h', 0) > 3:
        print(f"⚠️  High velocity: {tx['velocity_1h']} transactions in 1h")
    
    prob, decision, details = predict_single(tx, root=ROOT)
    
    print(f"\n🎯 DECISION: {decision}")
    print(f"   Probability: {prob:.4f}")
    
    # Show ML + Rules breakdown
    ml = details['ml_scores']
    rule = details['rule_output']
    fusion = details['fusion_logic']
    
    print(f"\n📊 ML Analysis:")
    print(f"   XGBoost: {ml['xgb_prob']:.4f}")
    print(f"   Ensemble: {ml['final_ml_prob']:.4f}")
    
    print(f"\n📋 Rule Engine:")
    print(f"   Tier: {rule.get('tier', 'NEUTRAL')}")
    print(f"   Flags: {', '.join(rule['flags']) if rule['flags'] else 'None'}")
    print(f"   Rule Score: {rule['rule_prob']:.4f}")
    
    print(f"\n⚖️  Fusion:")
    print(f"   ML (50%): {fusion['ml_contribution']:.4f}")
    print(f"   Rules (50%): {fusion['rule_contribution']:.4f}")
    if fusion['overridden']:
        print(f"   🔒 Hard Override: {fusion.get('override_reason', 'N/A')}")
    
    print(f"\n💡 Explanation: {rule['explanation']}")
    print("=" * 80)


def main():
    """Run demo predictions."""
    
    print("\n" + "🚀" * 40)
    print("UPI FRAUD DETECTION SYSTEM - DEMO")
    print("Hybrid ML + Rules Engine with Weighted Fusion")
    print("🚀" * 40)
    
    # Scenario 1: Clean legitimate transaction
    print_prediction(
        "✅ Small Amount - Legitimate",
        {
            'user_id': 'u_1',
            'amount': 50.0,
            'hour': 14,
            'receiver_id': 'r_grocery_store',
            'is_new_receiver': False,
            'device_changed': False,
            'velocity_1h': 1,
            'receiver_risk_score': 0.1,
            'sender_location': 'Mumbai',
            'receiver_location': 'Mumbai'
        }
    )
    
    # Scenario 2: Known receiver - high trust
    print_prediction(
        "✅ Known Receiver - Legitimate",
        {
            'user_id': 'u_1',
            'amount': 5000.0,
            'hour': 11,
            'receiver_id': 'R001',  # Known receiver with 12+ past transactions
            'is_new_receiver': False,
            'device_changed': False,
            'velocity_1h': 1,
            'receiver_risk_score': 0.05,
            'sender_location': 'Mumbai',
            'receiver_location': 'Mumbai'
        }
    )
    
    # Scenario 3: Suspicious - new receiver + amount spike
    print_prediction(
        "⚠️  New Receiver + Amount Spike - Suspicious",
        {
            'user_id': 'u_3',
            'amount': 1500.0,
            'hour': 14,
            'receiver_id': 'r_unknown_001',
            'is_new_receiver': True,
            'device_changed': False,
            'velocity_1h': 2,
            'receiver_risk_score': 0.65,
            'sender_location': 'Delhi',
            'receiver_location': 'Delhi'
        }
    )
    
    # Scenario 4: Suspicious - unusual time + location mismatch
    print_prediction(
        "⚠️  Late Night + Location Mismatch - Suspicious",
        {
            'user_id': 'u_5',
            'amount': 1200.0,
            'hour': 2,  # 2 AM
            'receiver_id': 'r_unknown_002',
            'is_new_receiver': True,
            'device_changed': False,
            'velocity_1h': 1,
            'receiver_risk_score': 0.55,
            'sender_location': 'Mumbai',
            'receiver_location': 'Kolkata'
        }
    )
    
    # Scenario 5: High fraud probability - multiple risk factors
    print_prediction(
        "🚨 Multiple Risk Factors - Likely Fraud",
        {
            'user_id': 'u_7',
            'amount': 8000.0,
            'hour': 3,  # 3 AM
            'receiver_id': 'r_suspicious_account',
            'is_new_receiver': True,
            'device_changed': True,
            'velocity_1h': 5,
            'receiver_risk_score': 0.92,
            'sender_location': 'Bangalore',
            'receiver_location': 'Unknown'
        }
    )
    
    # Scenario 6: Edge case - moderate amount with mixed signals
    print_prediction(
        "🤔 Mixed Signals - ML + Rules Balance",
        {
            'user_id': 'u_10',
            'amount': 2500.0,
            'hour': 20,
            'receiver_id': 'r_merchant_xyz',
            'is_new_receiver': False,
            'device_changed': False,
            'velocity_1h': 3,
            'receiver_risk_score': 0.45,
            'sender_location': 'Chennai',
            'receiver_location': 'Chennai'
        }
    )
    
    print("\n" + "🎉" * 40)
    print("DEMO COMPLETE")
    print("All scenarios processed successfully!")
    print("🎉" * 40 + "\n")


if __name__ == '__main__':
    main()
