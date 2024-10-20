# College Enquiry Chatbot

## Overview

The **College Enquiry Chatbot** is an AI-driven conversational agent designed to assist prospective students in obtaining information about college admissions, courses, campus life, and more. Utilizing state-of-the-art technologies like Google Generative AI, FAISS, and Langchain, this chatbot provides context-aware and interactive responses, enhancing the user experience for potential applicants.

## Features

- **Natural Language Understanding**: The chatbot can understand and respond to user queries in natural language, providing accurate and relevant information.
- **Contextual Responses**: Leveraging **FAISS** (Facebook AI Similarity Search) and **Langchain**, the chatbot retrieves answers based on the context of the conversation, ensuring meaningful interactions.
- **24/7 Availability**: The chatbot is accessible at any time, offering assistance outside regular office hours.
- **User-Friendly Interface**: Designed to be intuitive, allowing users to easily navigate and ask questions about college-related topics.

## Technologies Used

- **Programming Language**: Python
- **AI Frameworks**: Google Generative AI, Langchain
- **Data Retrieval**: FAISS for efficient similarity search and context retrieval
- **Deployment**: FastAPI for creating and managing the API

## Getting Started

### Prerequisites

To run the College Enquiry Chatbot locally, you will need:

- Python 3.7 or higher
- Required libraries (install via `requirements.txt`)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/college-enquiry-chatbot.git
   cd college-enquiry-chatbot
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   uvicorn main:app --reload
   ```

4. Access the chatbot in your web browser at `http://localhost:8000`.

## Usage

- Interact with the chatbot by typing your queries in the input box.
- The chatbot will provide responses based on the context and information from the college brochure.
- Examples of queries include:
  - "What are the admission requirements?"
  - "Can you tell me about the courses offered?"
  - "What is campus life like?"

## Contributing

Contributions to improve the College Enquiry Chatbot are welcome! If you have suggestions or enhancements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or feedback, feel free to reach out:

- Email: [raghavkhandelwal39@gmail.com]
- LinkedIn: [[LinkedIn profile](https://www.linkedin.com/in/raghav-khandelwal-a42545228/)]
