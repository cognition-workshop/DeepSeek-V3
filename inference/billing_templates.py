"""
Billing template response generator for DeepSeek-V3.
This module provides functionality to detect billing-related queries
and generate appropriate template-based responses.
"""

from typing import Dict, List, Tuple, Optional
import re

BILLING_CATEGORIES = {
    "refund": [
        "refund", "money back", "cancel order", "return", "reimbursement",
        "chargeback", "payment reversal", "cancel subscription"
    ],
    "invoice": [
        "invoice", "receipt", "bill", "statement", "tax document",
        "proof of payment", "payment record", "billing statement"
    ],
    "pricing": [
        "pricing", "cost", "price", "fee", "charge", "subscription cost",
        "monthly fee", "annual plan", "discount", "promotion"
    ],
    "payment_method": [
        "payment method", "credit card", "debit card", "bank account",
        "paypal", "payment option", "billing info", "update card"
    ],
    "subscription": [
        "subscription", "plan", "upgrade", "downgrade", "renewal",
        "cancel subscription", "change plan", "billing cycle"
    ],
    "general_billing": [
        "billing", "payment", "charge", "account", "transaction"
    ]
}

BILLING_TEMPLATES = {
    "refund": """I understand you're inquiring about a refund. Here's our refund policy information:

1. Refund requests must be submitted within 30 days of purchase
2. Refunds typically process within 5-7 business days
3. Original payment method will be refunded
4. To process your refund request, please provide:
   - Order/transaction ID
   - Date of purchase
   - Reason for refund

If you have specific questions about your refund status, our billing team can assist further with your case details.""",

    "invoice": """Regarding your invoice inquiry, here's some helpful information:

1. Invoices are automatically generated and emailed after each payment
2. You can also access all invoices from your account dashboard
3. Invoices include:
   - Itemized charges
   - Payment method used
   - Transaction date
   - Tax information

If you need a specific invoice or have questions about charges, please provide your account ID or transaction details, and our billing team can assist further.""",

    "pricing": """Regarding your pricing inquiry, here's our current pricing structure:

1. Basic Plan: $X/month - Includes standard features
2. Professional Plan: $Y/month - Includes advanced features
3. Enterprise Plan: Custom pricing - Includes all features plus dedicated support

All plans are available with monthly or annual billing (save 20% with annual billing).
Special discounts may be available for educational or non-profit organizations.

For specific pricing questions or to request a custom quote, please provide more details about your needs.""",

    "payment_method": """Regarding your payment method inquiry:

1. We accept various payment methods including:
   - Credit/debit cards (Visa, Mastercard, American Express)
   - PayPal
   - Bank transfers (for annual enterprise plans)

2. To update your payment method:
   - Log into your account
   - Navigate to Billing Settings
   - Select "Update Payment Method"
   - Enter your new payment details

If you're experiencing issues updating your payment information or have specific questions, please provide more details about your account.""",

    "subscription": """Regarding your subscription inquiry:

1. You can manage your subscription from your account dashboard:
   - Upgrade/downgrade plans
   - Change billing frequency (monthly/annual)
   - Cancel subscription

2. Subscription changes:
   - Upgrades: Applied immediately with prorated charges
   - Downgrades: Applied at the end of billing cycle
   - Cancellations: Service continues until the end of billing period

If you have specific questions about your subscription or need assistance making changes, please provide your account details.""",

    "general_billing": """Regarding your billing inquiry:

Our billing system processes payments securely and provides detailed transaction records for all account activities. For specific billing questions, please provide:

1. Your account ID or email
2. Details about your specific billing concern
3. Any relevant transaction IDs or dates

Our billing support team can then provide personalized assistance with your specific situation."""
}

def detect_billing_category(query: str) -> Optional[str]:
    """
    Detect the billing category of a customer query.
    
    Args:
        query (str): The customer's query text
        
    Returns:
        Optional[str]: The detected billing category or None if not billing-related
    """
    query = query.lower()
    
    for category, keywords in BILLING_CATEGORIES.items():
        for keyword in keywords:
            if keyword.lower() in query:
                return category
    
    return None

def get_billing_response(query: str) -> Optional[str]:
    """
    Generate a template-based response for a billing-related query.
    
    Args:
        query (str): The customer's query text
        
    Returns:
        Optional[str]: The template response or None if not billing-related
    """
    category = detect_billing_category(query)
    if category:
        return BILLING_TEMPLATES[category]
    return None
