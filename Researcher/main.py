import os
import getpass
from dotenv import load_dotenv
from agents import ResearchAssistant
from utils import get_logger


if __name__ == '__main__':
    logger = get_logger('main')

    print('\n======================* Research Assistant *======================\n')
    print('Guidelines:')
    print('1. Enter a broad research topic (e.g., "Quantum Computing").')
    print('2. The assistant will:')
    print('   - Search academic papers, Wikipedia, and recent news')
    print('   - Extract structured knowledge')
    print('   - Write, critique, and assemble a full research report')
    print('3. Outputs include logs, saved JSON state, and a final PDF report.')
    print('4. Notes:')
    print('   - The process may take several minutes depending on topic complexity.')
    print('   - Requires valid OpenAI + LangSmith credentials.')
    print('   - If keys are missing, you\'ll be prompted securely.')
    print('   - You can cancel anytime with Ctrl+C.\n')
    print('=================================================================\n')

    if load_dotenv():
        logger.info('.env loaded successfully.')
    else:
        logger.warning('.env not found. You will be prompted for missing keys.')

    os.environ['LANGSMITH_TRACING'] = 'true'
    os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'

    required_vars = ['OPENAI_API_KEY', 'LANGSMITH_API_KEY', 'LANGSMITH_PROJECT']
    for var in required_vars:
        if not os.environ.get(var):
            logger.warning(f'"{var}" not found. Requesting input...')
            os.environ[var] = getpass.getpass(f'Enter your {var} (input hidden): ')

    logger.info('All required environment variables available. Proceeding...')

    assistant = ResearchAssistant()
    topic = input('\nEnter the topic of research: ').strip()
    if not topic:
        logger.error('No topic provided. Exiting...')
        exit(1)

    result = assistant.run(topic)
