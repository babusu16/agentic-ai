"""
Enhanced Multi-Agent Feedback Analysis System using CrewAI
Full implementation with proper CrewAI orchestration
"""

import os
import csv
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import re

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/output/crewai_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create output directories
os.makedirs('data', exist_ok=True)
os.makedirs('data/output', exist_ok=True)


# ============= TOOLS DEFINITION =============

class CSVReaderTool(BaseTool):
    """Tool for reading CSV files"""
    name: str = "csv_reader"
    description: str = "Reads and parses CSV files containing user feedback data"

    def _run(self, file_path: str) -> str:
        """Execute the tool"""
        try:
            df = pd.read_csv(file_path)
            # Convert to JSON string for agent processing
            data = df.to_dict('records')
            return json.dumps(data[:20])  # Limit for token management
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return f"Error reading file: {str(e)}"


class FeedbackClassifierTool(BaseTool):
    """Tool for classifying feedback"""
    name: str = "feedback_classifier"
    description: str = "Classifies feedback text into categories (Bug, Feature Request, Praise, Complaint, Spam)"

    def _run(self, feedback_text: str) -> str:
        """Classify feedback using rule-based approach"""
        text_lower = feedback_text.lower()

        # Classification logic
        if any(word in text_lower for word in ['crash', 'error', 'broken', 'bug', 'fix', 'issue']):
            category = 'Bug'
            priority = 'High' if 'crash' in text_lower or 'critical' in text_lower else 'Medium'
        elif any(word in text_lower for word in ['add', 'feature', 'request', 'would like', 'please add']):
            category = 'Feature Request'
            priority = 'Medium'
        elif any(word in text_lower for word in ['love', 'great', 'amazing', 'perfect', 'excellent']):
            category = 'Praise'
            priority = 'Low'
        elif any(word in text_lower for word in ['hate', 'terrible', 'awful', 'worst', 'bad']):
            category = 'Complaint'
            priority = 'Medium'
        elif any(word in text_lower for word in ['click here', 'win', 'prize', 'bitcoin']):
            category = 'Spam'
            priority = 'Low'
        else:
            category = 'General'
            priority = 'Low'

        return json.dumps({
            'category': category,
            'priority': priority,
            'confidence': 0.85
        })


class TechnicalAnalyzerTool(BaseTool):
    """Tool for extracting technical details"""
    name: str = "technical_analyzer"
    description: str = "Extracts technical information like device, OS, version, and steps to reproduce from bug reports"

    def _run(self, feedback_data: str) -> str:
        """Extract technical details"""
        try:
            data = json.loads(feedback_data) if isinstance(feedback_data, str) else feedback_data
            text = str(data.get('text', '')) if isinstance(data, dict) else str(feedback_data)

            details = {}

            # Extract version
            version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', text)
            if version_match:
                details['version'] = version_match.group(1)

            # Extract device info
            devices = ['iphone', 'ipad', 'android', 'samsung', 'pixel']
            for device in devices:
                if device in text.lower():
                    details['device'] = device.capitalize()
                    break

            # Extract OS
            if 'ios' in text.lower():
                details['os'] = 'iOS'
            elif 'android' in text.lower():
                details['os'] = 'Android'

            # Check for steps to reproduce
            if 'step' in text.lower() or 'reproduce' in text.lower():
                details['has_steps'] = True

            # Extract error codes
            error_match = re.search(r'error\s*(?:code)?\s*:?\s*(\w+)', text.lower())
            if error_match:
                details['error_code'] = error_match.group(1).upper()

            return json.dumps(details)
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return json.dumps({'error': str(e)})


class TicketCreatorTool(BaseTool):
    """Tool for creating structured tickets"""
    name: str = "ticket_creator"
    description: str = "Creates structured tickets from analyzed feedback with proper formatting"

    def _run(self, ticket_data: str) -> str:
        """Create a formatted ticket"""
        try:
            data = json.loads(ticket_data) if isinstance(ticket_data, str) else ticket_data

            ticket = {
                'ticket_id': f"T{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'title': data.get('title', 'New Ticket'),
                'description': data.get('description', ''),
                'category': data.get('category', 'General'),
                'priority': data.get('priority', 'Medium'),
                'technical_details': data.get('technical_details', {}),
                'source': data.get('source', 'Unknown'),
                'created_at': datetime.now().isoformat(),
                'status': 'New',
                'assigned_to': 'Unassigned'
            }

            return json.dumps(ticket)
        except Exception as e:
            logger.error(f"Ticket creation error: {e}")
            return json.dumps({'error': str(e)})


class QualityValidatorTool(BaseTool):
    """Tool for validating ticket quality"""
    name: str = "quality_validator"
    description: str = "Validates the completeness and quality of generated tickets"

    def _run(self, ticket_json: str) -> str:
        """Validate ticket quality"""
        try:
            ticket = json.loads(ticket_json) if isinstance(ticket_json, str) else ticket_json

            issues = []
            score = 100

            # Check required fields
            required_fields = ['title', 'description', 'category', 'priority']
            for field in required_fields:
                if not ticket.get(field):
                    issues.append(f"Missing {field}")
                    score -= 20

            # Check title length
            if ticket.get('title') and len(ticket['title']) < 10:
                issues.append("Title too short")
                score -= 10

            # Check description length
            if ticket.get('description') and len(ticket['description']) < 20:
                issues.append("Description too brief")
                score -= 10

            # Validate category
            valid_categories = ['Bug', 'Feature Request', 'Complaint', 'Praise', 'Spam', 'General']
            if ticket.get('category') not in valid_categories:
                issues.append("Invalid category")
                score -= 15

            # Validate priority
            valid_priorities = ['Critical', 'High', 'Medium', 'Low']
            if ticket.get('priority') not in valid_priorities:
                issues.append("Invalid priority")
                score -= 15

            return json.dumps({
                'valid': len(issues) == 0,
                'score': max(score, 0),
                'issues': issues,
                'recommendation': 'Approved' if score >= 70 else 'Needs Review'
            })
        except Exception as e:
            logger.error(f"Quality validation error: {e}")
            return json.dumps({'error': str(e)})


class CSVWriterTool(BaseTool):
    """Tool for writing results to CSV"""
    name: str = "csv_writer"
    description: str = "Writes processed tickets to CSV files"

    def _run(self, data: str) -> str:
        """Write data to CSV"""
        try:
            tickets = json.loads(data) if isinstance(data, str) else data
            if not isinstance(tickets, list):
                tickets = [tickets]

            file_path = 'data/output/generated_tickets.csv'

            # Prepare data for CSV
            csv_data = []
            for ticket in tickets:
                if isinstance(ticket, dict):
                    # Flatten technical details if present
                    if 'technical_details' in ticket and isinstance(ticket['technical_details'], dict):
                        ticket['technical_details'] = json.dumps(ticket['technical_details'])
                    csv_data.append(ticket)

            # Write to CSV
            if csv_data:
                df = pd.DataFrame(csv_data)
                df.to_csv(file_path, index=False, mode='a', header=not os.path.exists(file_path))
                return f"Successfully wrote {len(csv_data)} tickets to {file_path}"
            else:
                return "No valid tickets to write"
        except Exception as e:
            logger.error(f"CSV writing error: {e}")
            return f"Error writing CSV: {str(e)}"


# ============= CREWAI AGENTS SETUP =============

class FeedbackAnalysisCrew:
    """Main CrewAI orchestration class"""

    def __init__(self, use_local_llm: bool = False):
        """Initialize the crew"""
        self.use_local_llm = use_local_llm
        self.setup_llm()
        self.setup_tools()
        self.setup_agents()

    def setup_llm(self):
        """Setup the language model"""
        load_dotenv(override=True)
        if self.use_local_llm:
            # Use OpenAI GPT
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7
            )

    def setup_tools(self):
        """Initialize all tools"""
        self.csv_reader = CSVReaderTool()
        self.classifier = FeedbackClassifierTool()
        self.tech_analyzer = TechnicalAnalyzerTool()
        self.ticket_creator = TicketCreatorTool()
        self.quality_validator = QualityValidatorTool()
        self.csv_writer = CSVWriterTool()

    def setup_agents(self):
        """Create CrewAI agents"""

        # Data Ingestion Agent
        self.data_agent = Agent(
            role='Data Ingestion Specialist',
            goal='Read and parse user feedback from CSV files accurately and efficiently',
            backstory="""You are an expert in data extraction and parsing. Your role is to 
            read feedback from various sources and prepare it for analysis. You ensure no 
            data is lost and maintain data integrity throughout the process.""",
            tools=[self.csv_reader],
            llm=self.llm,
            verbose=True,
            max_iter=3
        )

        # Classification Agent
        self.classifier_agent = Agent(
            role='Feedback Classification Expert',
            goal='Accurately categorize each feedback item into the correct category with appropriate priority',
            backstory="""You are an NLP specialist with years of experience in text classification. 
            You can identify bugs, feature requests, complaints, praise, and spam with high accuracy. 
            You understand user sentiment and can prioritize issues effectively.""",
            tools=[self.classifier],
            llm=self.llm,
            verbose=True,
            max_iter=3
        )

        # Bug Analysis Agent
        self.bug_agent = Agent(
            role='Bug Analysis Specialist',
            goal='Extract detailed technical information from bug reports to help developers reproduce and fix issues',
            backstory="""You are a senior QA engineer who specializes in bug analysis. You can 
            identify critical information like device types, OS versions, error codes, and 
            reproduction steps. Your analysis helps developers fix bugs quickly.""",
            tools=[self.tech_analyzer],
            llm=self.llm,
            verbose=True,
            max_iter=3
        )

        # Feature Extraction Agent
        self.feature_agent = Agent(
            role='Feature Request Analyst',
            goal='Identify and analyze feature requests to understand user needs and estimate impact',
            backstory="""You are a product manager who excels at understanding user needs. 
            You can identify valuable feature requests, assess their potential impact, and 
            prioritize them based on user demand and business value.""",
            tools=[self.classifier],
            llm=self.llm,
            verbose=True,
            max_iter=3
        )

        # Ticket Creation Agent
        self.ticket_agent = Agent(
            role='Ticket Generation Specialist',
            goal='Create well-structured, actionable tickets that development teams can work on immediately',
            backstory="""You are a technical writer and project manager who creates clear, 
            comprehensive tickets. You ensure each ticket has all necessary information, 
            proper formatting, and clear action items for the development team.""",
            tools=[self.ticket_creator],
            llm=self.llm,
            verbose=True,
            max_iter=3
        )

        # Quality Assurance Agent
        self.qa_agent = Agent(
            role='Quality Assurance Expert',
            goal='Review and validate all generated tickets for completeness, accuracy, and actionability',
            backstory="""You are a senior QA lead who ensures quality standards are met. 
            You review every ticket for completeness, accuracy, and clarity. You catch 
            issues before they reach the development team.""",
            tools=[self.quality_validator],
            llm=self.llm,
            verbose=True,
            max_iter=2
        )

        # Output Agent
        self.output_agent = Agent(
            role='Data Output Coordinator',
            goal='Save all processed tickets and results to appropriate CSV files',
            backstory="""You manage the final output of the system, ensuring all results 
            are properly formatted and saved for future reference and analysis.""",
            tools=[self.csv_writer],
            llm=self.llm,
            verbose=True,
            max_iter=2
        )

    def create_tasks(self, reviews_file: str, emails_file: str):
        """Create CrewAI tasks for the workflow"""

        # Task 1: Read App Store Reviews
        task1 = Task(
            description=f"""
            Read the app store reviews from the CSV file: {reviews_file}
            Extract all review data including review_id, platform, rating, review_text, 
            user_name, date, and app_version. 
            Return the data in a structured format for analysis.
            """,
            agent=self.data_agent,
            expected_output="JSON formatted review data"
        )

        # Task 2: Read Support Emails
        task2 = Task(
            description=f"""
            Read the support emails from the CSV file: {emails_file}
            Extract all email data including email_id, subject, body, sender_email, 
            timestamp, and priority.
            Return the data in a structured format for analysis.
            """,
            agent=self.data_agent,
            expected_output="JSON formatted email data"
        )

        # Task 3: Classify Reviews
        task3 = Task(
            description="""
            Analyze each review from Task 1 and classify them into categories:
            - Bug: Technical issues, crashes, errors
            - Feature Request: User suggestions and requests
            - Praise: Positive feedback
            - Complaint: Negative feedback about service/pricing
            - Spam: Irrelevant or promotional content

            Assign appropriate priority levels (Critical, High, Medium, Low) based on severity and impact.
            """,
            agent=self.classifier_agent,
            expected_output="Classified reviews with categories and priorities"
        )

        # Task 4: Classify Emails
        task4 = Task(
            description="""
            Analyze each email from Task 2 and classify them using the same categories as reviews.
            Pay special attention to technical details in email bodies.
            Consider existing priority if provided in the email data.
            """,
            agent=self.classifier_agent,
            expected_output="Classified emails with categories and priorities"
        )

        # Task 5: Analyze Bugs
        task5 = Task(
            description="""
            For all feedback classified as 'Bug' in Tasks 3 and 4:
            1. Extract technical details (device, OS, version, error codes)
            2. Identify steps to reproduce if mentioned
            3. Assess severity based on impact
            4. Compile technical information for developers
            """,
            agent=self.bug_agent,
            expected_output="Detailed bug analysis with technical information"
        )

        # Task 6: Analyze Feature Requests
        task6 = Task(
            description="""
            For all feedback classified as 'Feature Request' in Tasks 3 and 4:
            1. Identify the specific feature being requested
            2. Assess potential user impact
            3. Estimate implementation complexity
            4. Prioritize based on demand and value
            """,
            agent=self.feature_agent,
            expected_output="Analyzed feature requests with impact assessment"
        )

        # Task 7: Create Tickets
        task7 = Task(
            description="""
            Create structured tickets for all high-priority items from Tasks 5 and 6:
            1. Generate clear, actionable titles
            2. Write comprehensive descriptions
            3. Include all technical details
            4. Set appropriate priority and category
            5. Add source information for traceability

            Focus on bugs and feature requests, but also create tickets for 
            critical complaints that need attention.
            """,
            agent=self.ticket_agent,
            expected_output="Well-structured tickets ready for development team"
        )

        # Task 8: Quality Review
        task8 = Task(
            description="""
            Review all tickets created in Task 7:
            1. Verify completeness of information
            2. Check clarity and actionability
            3. Validate priority assignments
            4. Ensure technical details are accurate
            5. Flag any tickets that need revision

            Provide a quality score and recommendations for each ticket.
            """,
            agent=self.qa_agent,
            expected_output="Quality-reviewed tickets with validation scores"
        )

        # Task 9: Save Results
        task9 = Task(
            description="""
            Save all validated tickets from Task 8 to CSV files:
            1. Write tickets to 'generated_tickets.csv'
            2. Include all fields: ticket_id, title, description, category, priority, etc.
            3. Ensure proper formatting for CSV output
            4. Create a summary report of processing results
            """,
            agent=self.output_agent,
            expected_output="Confirmation of saved tickets and summary report"
        )

        return [task1, task2, task3, task4, task5, task6, task7, task8, task9]

    def run(self, reviews_file: str = 'data/app_store_reviews.csv',
            emails_file: str = 'data/support_emails.csv'):
        """Execute the CrewAI workflow"""

        logger.info("Starting CrewAI Feedback Analysis Crew...")

        # Create tasks
        tasks = self.create_tasks(reviews_file, emails_file)

        # Create and run the crew
        crew = Crew(
            agents=[
                self.data_agent,
                self.classifier_agent,
                self.bug_agent,
                self.feature_agent,
                self.ticket_agent,
                self.qa_agent,
                self.output_agent
            ],
            tasks=tasks,
            process=Process.sequential,  # Tasks run in sequence
            verbose=True
        )

        # Execute the crew
        logger.info("Executing crew tasks...")
        result = crew.kickoff()

        logger.info("CrewAI processing complete!")

        # Save final summary
        self.save_summary(result)

        return result

    def save_summary(self, result):
        """Save processing summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'result': str(result),
            'status': 'Complete'
        }

        with open('data/output/processing_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("Summary saved to data/output/processing_summary.json")


def main():
    """Main execution function"""

    # Ensure mock data exists
    if not os.path.exists('data/app_store_reviews.csv'):
        print("Creating mock data files...")
        from create_mock_data import create_mock_data
        create_mock_data()

    print("\n" + "=" * 60)
    print("🚀 CREWAI FEEDBACK ANALYSIS SYSTEM")
    print("=" * 60)

    # Initialize and run the crew
    crew = FeedbackAnalysisCrew(use_local_llm=True)

    print("\n📊 Starting multi-agent processing...")
    print("This will orchestrate 7 specialized agents to analyze your feedback.\n")

    result = crew.run()

    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    print("\nResults saved to: data/output/")
    print("- generated_tickets.csv")
    print("- processing_summary.json")
    print("- crewai_processing.log")

    # Display summary statistics
    try:
        tickets_df = pd.read_csv('data/output/generated_tickets.csv')
        print(f"\n📈 Summary Statistics:")
        print(f"- Total tickets generated: {len(tickets_df)}")
        if 'category' in tickets_df.columns:
            print(f"- Categories: {tickets_df['category'].value_counts().to_dict()}")
        if 'priority' in tickets_df.columns:
            print(f"- Priorities: {tickets_df['priority'].value_counts().to_dict()}")
    except:
        pass


if __name__ == "__main__":
    main()