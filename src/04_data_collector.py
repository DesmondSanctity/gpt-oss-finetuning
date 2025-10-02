"""
Collect training data using Bright Data's web scraping API
"""
from brightdata import bdclient
from config import BRIGHTDATA_API_TOKEN, SCRAPING_URLS
from typing import List, Dict
import re

class DataCollector:
    def __init__(self, api_token: str):
        self.client = bdclient(api_token=api_token)
        self.collected_data = []
        print("✅ Bright Data client initialized")

    def collect_documentation(self, urls: List[str]) -> List[Dict]:
        print(f"Scraping {len(urls)} URLs with Bright Data...")
        try:
            results = self.client.scrape(urls, data_format="markdown")
            if isinstance(results, str):
                training_data = self.process_single_result(results)
            elif isinstance(results, list):
                training_data = []
                for content in results:
                    if content:
                        examples = self.process_single_result(content)
                        training_data.extend(examples)
            else:
                print(f"Unexpected result type: {type(results)}")
                training_data = []
        except Exception as e:
            print(f"Batch scraping failed: {e}")
            training_data = []
            for url in urls:
                try:
                    print(f"  Scraping: {url}")
                    content = self.client.scrape(url, data_format="markdown")
                    if content:
                        examples = self.process_single_result(content)
                        training_data.extend(examples)
                        print(f"    ✓ Got {len(examples)} examples")
                except Exception as url_error:
                    print(f"    ✗ Error: {url_error}")
        self.collected_data = training_data
        print(f"✅ Total examples collected: {len(self.collected_data)}")
        return self.collected_data

    def process_single_result(self, content: str) -> List[Dict]:
        examples = []
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        content = re.sub(r'```[^`]*```', '', content)
        content = re.sub(r'`[^`]+`', '', content)
        content = re.sub(r'[#*_~>`|-]+', ' ', content)
        content = re.sub(r'\\(.)', r'\1', content)
        content = re.sub(r'https?://[^\s]+', '', content)
        content = re.sub(r'\S+\.\w+', '', content)
        content = re.sub(r'\s+', ' ', content)
        sentences = re.split(r'(?<=[.!?])\s+', content)
        clean_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if (len(sent) > 30 and not any(skip in sent.lower() for skip in ['navigation', 'copyright', 'index', 'table of contents', 'previous', 'next'])):
                clean_sentences.append(sent)
        for i in range(0, len(clean_sentences) - 1):
            instruction = clean_sentences[i][:200].strip()
            response = clean_sentences[i + 1][:300].strip()
            if len(instruction) > 20 and len(response) > 30:
                examples.append({"instruction": instruction, "response": response})
        return examples

def final_validation(examples: List[Dict]) -> List[Dict]:
    clean_data = []
    seen = set()
    for ex in examples:
        instruction = ex.get('instruction', '').strip()
        response = ex.get('response', '').strip()
        instruction = re.sub(r'[^a-zA-Z0-9\s\.,?!]', '', instruction)
        response = re.sub(r'[^a-zA-Z0-9\s\.,?!]', '', response)
        if (len(instruction) > 10 and len(response) > 20 and instruction not in seen):
            seen.add(instruction)
            clean_data.append({"instruction": instruction, "response": response})
    return clean_data

def collect_training_data():
    collector = DataCollector(api_token=BRIGHTDATA_API_TOKEN)
    training_data = collector.collect_documentation(SCRAPING_URLS)
    training_data = final_validation(training_data)
    print(f"Final clean dataset: {len(training_data)} examples")
    if len(training_data) == 0:
        raise ValueError("No valid training data after cleaning")
    print("\nClean training examples:")
    for i, example in enumerate(training_data[:3]):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {example['instruction']}")
        print(f"Response: {example['response']}")
    return training_data
