import csv
import random
from datetime import datetime, timedelta
import os


def create_mock_data():
    """Generate mock CSV files for testing the system"""

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # 1. Generate app_store_reviews.csv
    app_store_reviews = []
    review_id = 1000

    bug_templates = [
        "App crashes when I try to upload photos. Please fix this!",
        "Can't login since the last update. Getting error code 401",
        "Data sync not working between devices. Lost all my notes!",
        "The app freezes when switching between tabs",
        "Notifications stopped working after iOS update",
        "Export function broken - generates empty files",
        "Search feature returns no results even for existing items",
        "App uses too much battery in background",
        "Offline mode doesn't save changes properly",
        "Calendar sync with Google Calendar is broken"
    ]

    feature_requests = [
        "Please add dark mode! My eyes hurt at night",
        "Would love to see widget support for home screen",
        "Missing functionality for batch operations",
        "Can you add markdown support in notes?",
        "Please add collaboration features for teams",
        "Need better export options (PDF, Word)",
        "Would be great to have custom themes",
        "Please add voice notes feature",
        "Missing integration with Notion",
        "Need ability to create templates"
    ]

    praise_templates = [
        "Amazing app! Best productivity tool ever!",
        "Love the new update! Works perfectly now",
        "This app changed my life. So organized now!",
        "Works perfectly! Great job developers!",
        "Beautiful design and super intuitive",
        "Worth every penny! Premium is amazing",
        "Best note-taking app on the market",
        "Sync works flawlessly across all devices",
        "Customer support is incredible!",
        "Updates keep making it better and better"
    ]

    complaint_templates = [
        "Too expensive for basic features",
        "Poor customer service, no response for weeks",
        "App is slow and laggy on older phones",
        "Too many ads in free version",
        "Subscription model is a ripoff",
        "Interface is confusing and cluttered",
        "Takes forever to load large documents",
        "Premium features should be free",
        "Competitors offer more for less money",
        "Too many unnecessary features"
    ]

    spam_templates = [
        "Visit www.spam-site.com for free prizes!",
        "asdfghjkl random text 12345",
        "MAKE $5000 WORKING FROM HOME",
        "This review is about a different app entirely",
        "♠♣♥♦ symbols only ★☆★☆",
        "Buy bitcoin now! Best investment opportunity",
        "Check out my Instagram @fake_account",
        "This is not related to the app at all",
        "Lorem ipsum dolor sit amet consectetur",
        "CLICK HERE FOR FREE IPHONE!!!"
    ]

    platforms = ["Google Play", "App Store"]
    versions = ["2.1.3", "3.0.1", "3.0.2", "3.1.0", "2.9.8"]

    # Generate reviews
    for category, templates, rating_range in [
        ("bug", bug_templates, (1, 2)),
        ("feature", feature_requests, (3, 4)),
        ("praise", praise_templates, (4, 5)),
        ("complaint", complaint_templates, (1, 3)),
        ("spam", spam_templates, (1, 5))
    ]:
        for template in templates:
            review_id += 1
            date = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
            app_store_reviews.append({
                'review_id': f'R{review_id}',
                'platform': random.choice(platforms),
                'rating': random.randint(*rating_range),
                'review_text': template,
                'user_name': f'User_{random.randint(100, 999)}',
                'date': date,
                'app_version': random.choice(versions)
            })

    # Shuffle reviews
    random.shuffle(app_store_reviews)

    # Write app store reviews
    with open('/Users/sumathibabu/PycharmProjects/CapstonProject/MultiAgent/create_mockup_data/app_store_reviews.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['review_id', 'platform', 'rating',
                                               'review_text', 'user_name', 'date', 'app_version'])
        writer.writeheader()
        writer.writerows(app_store_reviews)

    # 2. Generate support_emails.csv
    support_emails = []
    email_id = 2000

    bug_emails = [
        {
            'subject': 'App Crash Report - Critical Issue',
            'body': 'Hi Support,\n\nThe app crashes immediately after launch on my iPhone 13 Pro running iOS 16.5. Steps to reproduce:\n1. Open the app\n2. Try to access settings\n3. App crashes\n\nPlease fix this urgently as I cannot use the app at all.\n\nBest regards',
            'priority': 'High'
        },
        {
            'subject': 'Login Issue - Cannot Access Account',
            'body': 'Hello,\n\nI have been unable to login for the past 3 days. Error message: "Invalid credentials" even though my password is correct. I have tried:\n- Resetting password\n- Clearing cache\n- Reinstalling app\n\nDevice: Samsung Galaxy S22\nAndroid version: 13\nApp version: 3.0.1\n\nPlease help!',
            'priority': 'High'
        },
        {
            'subject': 'Data Loss Problem',
            'body': 'Dear Team,\n\nI lost all my data after the recent update! I had over 500 notes that are now gone. This is unacceptable. I need my data recovered immediately.\n\nDevice: iPad Pro 2022\niOS: 16.4\nApp version: 3.0.2\n\nVery disappointed.',
            'priority': 'Critical'
        },
        {
            'subject': 'Sync Issues Between Devices',
            'body': 'Hi,\n\nSyncing stopped working between my phone and laptop. Changes made on phone don\'t appear on laptop and vice versa. Started happening after update to version 3.0.2.\n\nDevices affected:\n- iPhone 12 (iOS 16.3)\n- MacBook Pro 2021 (macOS Ventura)\n\nThanks',
            'priority': 'Medium'
        },
        {
            'subject': 'Performance Issues',
            'body': 'The app has become extremely slow lately. Takes 10+ seconds to open, freezes when scrolling. My device has plenty of storage and RAM available.\n\nDevice: Google Pixel 7\nAndroid: 13\n\nPlease investigate.',
            'priority': 'Medium'
        }
    ]

    feature_emails = [
        {
            'subject': 'Feature Request: Dark Mode',
            'body': 'Hello Product Team,\n\nI would really appreciate if you could add a dark mode option. I use the app extensively at night and the bright white interface strains my eyes.\n\nThis would be a great addition for many users.\n\nThank you for considering!',
            'priority': 'Low'
        },
        {
            'subject': 'Suggestion for Improvement - Collaboration',
            'body': 'Hi,\n\nI love your app but it would be perfect if you added real-time collaboration features. My team needs to work on documents together.\n\nFeatures needed:\n- Real-time editing\n- Comments\n- Version history\n- User permissions\n\nThanks!',
            'priority': 'Medium'
        },
        {
            'subject': 'Feature Request: API Access',
            'body': 'Dear Developers,\n\nPlease consider adding API access for developers. This would allow us to integrate your app with our workflows.\n\nUse cases:\n- Automated backups\n- Custom integrations\n- Bulk operations\n\nBest regards',
            'priority': 'Low'
        }
    ]

    general_emails = [
        {
            'subject': 'Great App!',
            'body': 'Just wanted to say your app is amazing! Keep up the great work!',
            'priority': ''
        },
        {
            'subject': 'Subscription Question',
            'body': 'Hi,\n\nCan you explain the difference between monthly and annual subscriptions? Also, do you offer student discounts?\n\nThanks',
            'priority': 'Low'
        }
    ]

    # Generate support emails
    all_emails = bug_emails + feature_emails + general_emails
    for email_data in all_emails:
        email_id += 1
        timestamp = (datetime.now() - timedelta(days=random.randint(0, 15),
                                                hours=random.randint(0, 23))).strftime("%Y-%m-%d %H:%M:%S")
        support_emails.append({
            'email_id': f'E{email_id}',
            'subject': email_data['subject'],
            'body': email_data['body'],
            'sender_email': f'user{random.randint(100, 999)}@email.com',
            'timestamp': timestamp,
            'priority': email_data['priority']
        })

    # Shuffle emails
    random.shuffle(support_emails)

    # Write support emails
    with open('/Users/sumathibabu/PycharmProjects/CapstonProject/MultiAgent/create_mockup_data/support_emails.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['email_id', 'subject', 'body',
                                               'sender_email', 'timestamp', 'priority'])
        writer.writeheader()
        writer.writerows(support_emails)

    # 3. Generate expected_classifications.csv (for validation)
    expected_classifications = []

    # Classify reviews
    for review in app_store_reviews[:20]:  # Sample for expected classifications
        if any(word in review['review_text'].lower() for word in ['crash', 'broken', 'error', 'freeze', 'sync']):
            category = 'Bug'
            priority = 'High' if review['rating'] == 1 else 'Medium'
            title = f"Bug: {review['review_text'][:30]}..."
        elif any(word in review['review_text'].lower() for word in ['please add', 'would love', 'missing', 'need']):
            category = 'Feature Request'
            priority = 'Medium'
            title = f"Feature: {review['review_text'][:30]}..."
        elif any(word in review['review_text'].lower() for word in ['amazing', 'love', 'perfect', 'great']):
            category = 'Praise'
            priority = 'Low'
            title = f"Positive: {review['review_text'][:30]}..."
        elif any(word in review['review_text'].lower() for word in ['expensive', 'slow', 'poor', 'ripoff']):
            category = 'Complaint'
            priority = 'Medium'
            title = f"Complaint: {review['review_text'][:30]}..."
        else:
            category = 'Spam'
            priority = 'Low'
            title = f"Spam: {review['review_text'][:30]}..."

        expected_classifications.append({
            'source_id': review['review_id'],
            'source_type': 'app_review',
            'category': category,
            'priority': priority,
            'technical_details': f"Platform: {review['platform']}, Version: {review['app_version']}",
            'suggested_title': title
        })

    # Write expected classifications
    with open('/Users/sumathibabu/PycharmProjects/CapstonProject/MultiAgent/create_mockup_data/expected_classifications.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source_id', 'source_type', 'category',
                                               'priority', 'technical_details', 'suggested_title'])
        writer.writeheader()
        writer.writerows(expected_classifications)

    print("✅ Mock data files created successfully in 'data' directory:")
    print("   - app_store_reviews.csv")
    print("   - support_emails.csv")
    print("   - expected_classifications.csv")


if __name__ == "__main__":
    create_mock_data()