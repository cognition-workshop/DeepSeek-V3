"""Test script for billing templates"""
from billing_templates import get_billing_response

test_queries = [
    "How do I request a refund for my last purchase?",
    "I need a copy of my invoice for tax purposes.",
    "What are your current pricing plans?",
    "How can I update my credit card information?",
    "I want to upgrade my subscription plan.",
    "I have a question about a charge on my account.",
    "What's the weather like today in San Francisco?"
]

for query in test_queries:
    response = get_billing_response(query)
    print(f"Query: {query}")
    print(f"Has billing template: {'Yes' if response else 'No'}")
    if response:
        print(f"Response: {response[:100]}...\n")
    else:
        print("No billing template detected, would use model response\n")
