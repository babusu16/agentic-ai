"""
Multi-Agent Feedback Analysis System
Main implementation using CrewAI for orchestrating agents
"""

import os
import csv
import json
from dotenv import load_dotenv
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import re

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import nltk
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Data classes for structured data
@dataclass
class FeedbackItem:
    """Represents a single feedback item"""
    id: str
    source: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    classification: Optional[str] = None
    priority: Optional[str] = None
    confidence: float = 0.0
    technical_details: Optional[str] = None

@dataclass
class Ticket:
    """Represents a generated ticket"""
    ticket_id: str
    title: str
    description: str
    category: str
    priority: str
    source_id: str
    source_type: str
    technical_details: str
    created_at: str
    status: str = "New"
    assigned_to: str = "Unassigned"

class CSVReaderTool(BaseTool):
    """Tool for reading CSV files"""
    name: str = "csv_reader"
    description: str = "Reads and parses feedback data from CSV files"

    def _run(self, file_path: str) -> List[Dict]:
        """Execute the tool"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return []

class CSVWriterTool(BaseTool):
    """Tool for writing to CSV files"""
    name: str = "csv_writer"
    description: str = "Writes data to CSV files"

    def _run(self, file_path: str, data: List[Dict], fieldnames: List[str]) -> bool:
        """Execute the tool"""
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            return True
        except Exception as e:
            logger.error(f"Error writing CSV file {file_path}: {e}")
            return False

class TextClassifierTool(BaseTool):
    """Tool for classifying text using NLP"""
    name: str = "text_classifier"
    description: str = "Classifies feedback text into categories"

    def _run(self, text: str) -> Dict[str, Any]:
        """Classify text using rule-based NLP"""
        text_lower = text.lower()

        # Keywords for classification
        bug_keywords = ['crash', 'error', 'broken', 'fix', 'issue', 'problem', 'fail',
                       'freeze', 'stuck', 'slow', 'lag', 'sync', 'lost', 'missing data']
        feature_keywords = ['add', 'request', 'would love', 'please add', 'missing',
                           'need', 'want', 'wish', 'could you', 'suggestion', 'improvement']
        praise_keywords = ['love', 'amazing', 'great', 'excellent', 'perfect', 'awesome',
                          'fantastic', 'wonderful', 'best', 'thank']
        complaint_keywords = ['expensive', 'terrible', 'worst', 'hate', 'disappointed',
                             'awful', 'poor', 'bad', 'useless', 'waste']
        spam_keywords = ['click here', 'free', 'winner', 'bitcoin', 'lottery',
                        'make money', 'viagra', 'casino']

        # Count keyword matches
        scores = {
            'Bug': sum(1 for kw in bug_keywords if kw in text_lower),
            'Feature Request': sum(1 for kw in feature_keywords if kw in text_lower),
            'Praise': sum(1 for kw in praise_keywords if kw in text_lower),
            'Complaint': sum(1 for kw in complaint_keywords if kw in text_lower),
            'Spam': sum(1 for kw in spam_keywords if kw in text_lower)
        }

        # Sentiment analysis
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
        except:
            sentiment = 0

        # Adjust scores based on sentiment
        if sentiment > 0.5:
            scores['Praise'] += 2
        elif sentiment < -0.5:
            scores['Complaint'] += 2
            scores['Bug'] += 1

        # Determine category
        max_score = max(scores.values())
        if max_score == 0:
            category = 'General'
            confidence = 0.3
        else:
            category = max(scores, key=scores.get)
            confidence = min(max_score / 5, 1.0)  # Normalize confidence

        # Determine priority
        if category == 'Bug':
            if any(word in text_lower for word in ['crash', 'lost', 'critical', 'urgent']):
                priority = 'Critical'
            elif any(word in text_lower for word in ['error', 'broken', 'fail']):
                priority = 'High'
            else:
                priority = 'Medium'
        elif category == 'Feature Request':
            priority = 'Medium'
        elif category == 'Complaint':
            priority = 'Medium'
        else:
            priority = 'Low'

        return {
            'category': category,
            'confidence': confidence,
            'priority': priority,
            'sentiment': sentiment
        }

class TechnicalDetailExtractorTool(BaseTool):
    """Tool for extracting technical details from feedback"""
    name: str = "technical_extractor"
    description: str = "Extracts technical details from feedback text"

    def _run(self, text: str, metadata: Dict = None) -> str:
        """Extract technical details"""
        details = []

        # Extract version numbers
        version_pattern = r'\d+\.\d+(?:\.\d+)?'
        versions = re.findall(version_pattern, text)
        if versions:
            details.append(f"Version: {versions[0]}")
        elif metadata and 'app_version' in metadata:
            details.append(f"Version: {metadata['app_version']}")

        # Extract device information
        devices = {
            'iphone': 'iPhone',
            'ipad': 'iPad',
            'android': 'Android',
            'samsung': 'Samsung',
            'pixel': 'Google Pixel',
            'galaxy': 'Samsung Galaxy'
        }

        text_lower = text.lower()
        for key, value in devices.items():
            if key in text_lower:
                details.append(f"Device: {value}")
                break

        # Extract OS information
        if 'ios' in text_lower or 'iphone' in text_lower or 'ipad' in text_lower:
            details.append("OS: iOS")
        elif 'android' in text_lower:
            details.append("OS: Android")

        # Extract platform from metadata
        if metadata and 'platform' in metadata:
            details.append(f"Platform: {metadata['platform']}")

        # Look for error codes
        error_pattern = r'error\s*(?:code\s*)?[:\s]*(\w+)'
        errors = re.findall(error_pattern, text_lower)
        if errors:
            details.append(f"Error Code: {errors[0]}")

        # Extract steps to reproduce
        if 'steps' in text_lower or 'reproduce' in text_lower:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if 'step' in line.lower() or '1.' in line or '1)' in line:
                    steps = []
                    for j in range(i, min(i+5, len(lines))):
                        if lines[j].strip():
                            steps.append(lines[j].strip())
                    if steps:
                        details.append(f"Steps to reproduce found: {len(steps)} steps")
                    break

        return "; ".join(details) if details else "No specific technical details found"

class FeedbackAnalysisSystem:
    """Main system orchestrating all agents"""

    def __init__(self, use_local_llm: bool = True):
        """Initialize the system"""
        self.use_local_llm = use_local_llm
        self.setup_llm()
        self.setup_tools()
        self.setup_agents()
        self.processed_feedback = []
        self.generated_tickets = []
        self.metrics = {
            'total_processed': 0,
            'bugs_found': 0,
            'features_requested': 0,
            'critical_issues': 0,
            'processing_time': 0
        }

    def setup_llm(self):
        """Setup the language model"""
        load_dotenv(override=True)
        if self.use_local_llm:
            # Use Ollama for local LLM (requires Ollama to be installed)
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
            )

    def setup_tools(self):
        """Initialize tools"""
        self.csv_reader = CSVReaderTool()
        self.csv_writer = CSVWriterTool()
        self.text_classifier = TextClassifierTool()
        self.tech_extractor = TechnicalDetailExtractorTool()

    def setup_agents(self):
        """Setup CrewAI agents"""

        # CSV Reader Agent
        self.csv_reader_agent = Agent(
            role='Data Ingestion Specialist',
            goal='Read and parse feedback data from CSV files accurately',
            backstory='Expert in data extraction and parsing with attention to detail',
            tools=[self.csv_reader],
            verbose=True
        )

        # Feedback Classifier Agent
        self.classifier_agent = Agent(
            role='Feedback Classification Expert',
            goal='Accurately categorize feedback into appropriate categories',
            backstory='NLP specialist with expertise in text classification and sentiment analysis',
            tools=[self.text_classifier],
            verbose=True
        )

        # Bug Analysis Agent
        self.bug_analyst = Agent(
            role='Bug Analysis Specialist',
            goal='Extract technical details and severity assessment for bugs',
            backstory='Technical expert in identifying and analyzing software bugs',
            tools=[self.tech_extractor],
            verbose=True
        )

        # Feature Extractor Agent
        self.feature_analyst = Agent(
            role='Feature Request Analyst',
            goal='Identify and prioritize feature requests based on user impact',
            backstory='Product specialist understanding user needs and feature prioritization',
            tools=[self.text_classifier],
            verbose=True
        )

        # Ticket Creator Agent
        self.ticket_creator = Agent(
            role='Ticket Generation Specialist',
            goal='Create well-structured tickets with all necessary information',
            backstory='Expert in creating actionable tickets for development teams',
            tools=[self.csv_writer],
            verbose=True
        )

        # Quality Critic Agent
        self.quality_critic = Agent(
            role='Quality Assurance Expert',
            goal='Review and ensure quality of generated tickets',
            backstory='QA specialist ensuring completeness and accuracy of tickets',
            tools=[],
            verbose=True
        )

    def process_feedback(self, reviews_file: str, emails_file: str):
        """Main processing pipeline"""
        start_time = datetime.now()

        # Read feedback data
        logger.info("Reading feedback data...")
        reviews = self.csv_reader._run(reviews_file)
        emails = self.csv_reader._run(emails_file)

        all_feedback = []

        # Process reviews
        for review in reviews:
            feedback = FeedbackItem(
                id=review['review_id'],
                source='app_review',
                content=review['review_text'],
                metadata={
                    'platform': review.get('platform'),
                    'rating': review.get('rating'),
                    'app_version': review.get('app_version'),
                    'date': review.get('date')
                }
            )
            all_feedback.append(feedback)

        # Process emails
        for email in emails:
            feedback = FeedbackItem(
                id=email['email_id'],
                source='support_email',
                content=f"{email['subject']}\n{email['body']}",
                metadata={
                    'sender': email.get('sender_email'),
                    'timestamp': email.get('timestamp'),
                    'existing_priority': email.get('priority')
                }
            )
            all_feedback.append(feedback)

        # Classify and analyze feedback
        logger.info(f"Processing {len(all_feedback)} feedback items...")

        for feedback_item in all_feedback:
            # Classify feedback
            classification = self.text_classifier._run(feedback_item.content)
            feedback_item.classification = classification['category']
            feedback_item.confidence = classification['confidence']
            feedback_item.priority = classification['priority']

            # Extract technical details for bugs
            if feedback_item.classification == 'Bug':
                tech_details = self.tech_extractor._run(
                    feedback_item.content,
                    feedback_item.metadata
                )
                feedback_item.technical_details = tech_details
                self.metrics['bugs_found'] += 1
                if feedback_item.priority == 'Critical':
                    self.metrics['critical_issues'] += 1
            elif feedback_item.classification == 'Feature Request':
                self.metrics['features_requested'] += 1

            self.processed_feedback.append(feedback_item)

        # Generate tickets
        self.generate_tickets()

        # Calculate metrics
        end_time = datetime.now()
        self.metrics['total_processed'] = len(all_feedback)
        self.metrics['processing_time'] = (end_time - start_time).total_seconds()

        logger.info(f"Processing completed. Generated {len(self.generated_tickets)} tickets")

        return self.generated_tickets

    def generate_tickets(self):
        """Generate tickets from processed feedback"""
        ticket_id = 1000

        for feedback in self.processed_feedback:
            # Skip low-priority praise and spam
            if feedback.classification in ['Praise', 'Spam'] and feedback.priority == 'Low':
                continue

            ticket_id += 1

            # Generate ticket title
            if len(feedback.content) > 60:
                title = f"[{feedback.classification}] {feedback.content[:60]}..."
            else:
                title = f"[{feedback.classification}] {feedback.content}"

            # Create ticket
            ticket = Ticket(
                ticket_id=f"T{ticket_id}",
                title=title,
                description=feedback.content,
                category=feedback.classification,
                priority=feedback.priority,
                source_id=feedback.id,
                source_type=feedback.source,
                technical_details=feedback.technical_details or "N/A",
                created_at=datetime.now().isoformat()
            )

            self.generated_tickets.append(ticket)

    def save_results(self):
        """Save all results to CSV files"""
        os.makedirs('data/output', exist_ok=True)

        # Save generated tickets
        tickets_data = [
            {
                'ticket_id': t.ticket_id,
                'title': t.title,
                'description': t.description,
                'category': t.category,
                'priority': t.priority,
                'source_id': t.source_id,
                'source_type': t.source_type,
                'technical_details': t.technical_details,
                'created_at': t.created_at,
                'status': t.status,
                'assigned_to': t.assigned_to
            }
            for t in self.generated_tickets
        ]

        self.csv_writer._run(
            'data/output/generated_tickets.csv',
            tickets_data,
            ['ticket_id', 'title', 'description', 'category', 'priority',
             'source_id', 'source_type', 'technical_details', 'created_at',
             'status', 'assigned_to']
        )

        # Save processing log
        log_data = [
            {
                'feedback_id': f.id,
                'source': f.source,
                'classification': f.classification,
                'confidence': f.confidence,
                'priority': f.priority,
                'processed_at': datetime.now().isoformat()
            }
            for f in self.processed_feedback
        ]

        self.csv_writer._run(
            'data/output/processing_log.csv',
            log_data,
            ['feedback_id', 'source', 'classification', 'confidence', 'priority', 'processed_at']
        )

        # Save metrics
        metrics_data = [{
            'metric': key,
            'value': value
        } for key, value in self.metrics.items()]

        self.csv_writer._run(
            'data/output/metrics.csv',
            metrics_data,
            ['metric', 'value']
        )

        logger.info("Results saved to data/output/ directory")

    def get_summary(self):
        """Get processing summary"""
        return {
            'total_feedback': self.metrics['total_processed'],
            'tickets_generated': len(self.generated_tickets),
            'bugs_found': self.metrics['bugs_found'],
            'features_requested': self.metrics['features_requested'],
            'critical_issues': self.metrics['critical_issues'],
            'processing_time': f"{self.metrics['processing_time']:.2f} seconds",
            'categories': {
                'Bug': len([f for f in self.processed_feedback if f.classification == 'Bug']),
                'Feature Request': len([f for f in self.processed_feedback if f.classification == 'Feature Request']),
                'Complaint': len([f for f in self.processed_feedback if f.classification == 'Complaint']),
                'Praise': len([f for f in self.processed_feedback if f.classification == 'Praise']),
                'Spam': len([f for f in self.processed_feedback if f.classification == 'Spam'])
            }
        }

def main():
    """Main execution function"""
    # Create mock data if it doesn't exist
    if not os.path.exists('data/app_store_reviews.csv'):
        print("Creating mock data files...")
        from create_mock_data import create_mock_data
        create_mock_data()

    # Initialize system
    print("\n🚀 Starting Feedback Analysis System...")
    system = FeedbackAnalysisSystem(use_local_llm=True)  # Set to True to use local LLM

    # Process feedback
    print("\n📊 Processing feedback...")
    system.process_feedback(
        'data/app_store_reviews.csv',
        'data/support_emails.csv'
    )

    # Save results
    print("\n💾 Saving results...")
    system.save_results()

    # Display summary
    print("\n✅ Processing Complete!")
    print("\n📈 Summary:")
    summary = system.get_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  - {k}: {v}")
        else:
            print(f"  - {key}: {value}")

if __name__ == "__main__":
    main()