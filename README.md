Agentic-RAG Math Tutor System


This project implements an Agentic Retrieval-Augmented Generation system designed to answer mathematical questions with step-by-step solutions using a knowledge base and supplementary web search.



Features


Input and Output Guardrails: Ensures user queries are validated and AI responses stay focused on math education.

Knowledge Base: Uses a Qdrant vector DB with embedded math Q\&A for retrieval.

Web Search Fallback: Calls external search APIs when no relevant KB content exists.

LLM-driven stepwise answer generation using Google's Gemini API.

Frontend React app to interact with the backend API.

