// AI Glossary Terms Data
export const aiTerms = [
  // Machine Learning
  {
    id: "neural-network",
    term: "Neural Network",
    definition: "A computing system inspired by biological neural networks that consists of interconnected nodes (neurons) organized in layers. These networks can learn patterns from data and make predictions or classifications.",
    category: "Machine Learning",
    examples: ["Image recognition", "Natural language processing", "Speech recognition"]
  },
  {
    id: "deep-learning",
    term: "Deep Learning",
    definition: "A subset of machine learning that uses neural networks with multiple hidden layers (deep networks) to model and understand complex patterns in data.",
    category: "Machine Learning",
    examples: ["Computer vision", "Voice assistants", "Autonomous vehicles"]
  },
  {
    id: "supervised-learning",
    term: "Supervised Learning",
    definition: "A machine learning approach where the algorithm learns from labeled training data, with input-output pairs provided to guide the learning process.",
    category: "Machine Learning",
    examples: ["Email spam detection", "Medical diagnosis", "Price prediction"]
  },
  {
    id: "unsupervised-learning",
    term: "Unsupervised Learning",
    definition: "A machine learning approach that finds hidden patterns in data without labeled examples, discovering structure in unlabeled datasets.",
    category: "Machine Learning",
    examples: ["Customer segmentation", "Anomaly detection", "Data compression"]
  },
  {
    id: "reinforcement-learning",
    term: "Reinforcement Learning",
    definition: "A machine learning paradigm where an agent learns to make decisions by taking actions in an environment and receiving rewards or penalties for those actions.",
    category: "Machine Learning",
    examples: ["Game playing AI", "Robot navigation", "Trading algorithms"]
  },
  {
    id: "overfitting",
    term: "Overfitting",
    definition: "A modeling error that occurs when a machine learning model learns the training data too well, including noise and irrelevant patterns, leading to poor performance on new data.",
    category: "Machine Learning",
    examples: ["Memorizing training examples", "Poor generalization", "High training accuracy but low test accuracy"]
  },
  {
    id: "training-data",
    term: "Training Data",
    definition: "The dataset used to teach a machine learning algorithm how to make predictions or decisions. It contains input examples and their corresponding correct outputs.",
    category: "Machine Learning",
    examples: ["Labeled images for classification", "Historical stock prices", "Text with sentiment labels"]
  },

  // Natural Language Processing
  {
    id: "transformer",
    term: "Transformer",
    definition: "A neural network architecture that uses self-attention mechanisms to process sequences of data, revolutionizing natural language processing and forming the basis for models like GPT and BERT.",
    category: "NLP",
    examples: ["Language translation", "Text summarization", "Question answering"]
  },
  {
    id: "bert",
    term: "BERT",
    definition: "Bidirectional Encoder Representations from Transformers - a pre-trained language model that understands context by looking at words both before and after a target word.",
    category: "NLP",
    examples: ["Search query understanding", "Text classification", "Named entity recognition"]
  },
  {
    id: "gpt",
    term: "GPT",
    definition: "Generative Pre-trained Transformer - a family of language models trained to generate human-like text by predicting the next word in a sequence.",
    category: "NLP",
    examples: ["Text generation", "Code completion", "Conversational AI"]
  },
  {
    id: "tokenization",
    term: "Tokenization",
    definition: "The process of breaking down text into smaller units (tokens) such as words, subwords, or characters that can be processed by machine learning models.",
    category: "NLP",
    examples: ["Word splitting", "Subword units", "Character-level tokens"]
  },
  {
    id: "embedding",
    term: "Embedding",
    definition: "A dense vector representation of words, phrases, or other data that captures semantic meaning and relationships in a continuous mathematical space.",
    category: "NLP",
    examples: ["Word2Vec", "Sentence embeddings", "Image embeddings"]
  },
  {
    id: "large-language-model",
    term: "Large Language Model",
    definition: "A neural network with billions of parameters trained on vast amounts of text data to understand and generate human-like language across various tasks.",
    category: "NLP",
    examples: ["ChatGPT", "Claude", "LLaMA"]
  },

  // Computer Vision
  {
    id: "cnn",
    term: "CNN",
    definition: "Convolutional Neural Network - a deep learning architecture particularly effective for processing grid-like data such as images, using convolutional layers to detect features.",
    category: "Computer Vision",
    examples: ["Image classification", "Facial recognition", "Medical image analysis"]
  },
  {
    id: "object-detection",
    term: "Object Detection",
    definition: "A computer vision task that involves identifying and locating objects within images or videos, typically drawing bounding boxes around detected objects.",
    category: "Computer Vision",
    examples: ["Autonomous vehicle vision", "Security surveillance", "Retail inventory"]
  },
  {
    id: "image-classification",
    term: "Image Classification",
    definition: "The task of assigning a label or category to an entire image based on its visual content, determining what the image primarily contains.",
    category: "Computer Vision",
    examples: ["Photo tagging", "Medical diagnosis", "Quality control"]
  },
  {
    id: "segmentation",
    term: "Segmentation",
    definition: "The process of partitioning an image into multiple segments or regions, often to identify object boundaries or classify each pixel in the image.",
    category: "Computer Vision",
    examples: ["Medical imaging", "Autonomous driving", "Photo editing"]
  },

  // Ethics and General
  {
    id: "ai-bias",
    term: "AI Bias",
    definition: "Systematic and unfair discrimination in AI systems that can result from biased training data, algorithms, or human prejudices embedded in the development process.",
    category: "Ethics",
    examples: ["Hiring algorithms", "Facial recognition accuracy", "Loan approval systems"]
  },
  {
    id: "explainable-ai",
    term: "Explainable AI",
    definition: "AI systems designed to provide clear, understandable explanations for their decisions and predictions, making their reasoning process transparent to humans.",
    category: "Ethics",
    examples: ["Medical diagnosis explanations", "Credit score factors", "Legal decision support"]
  },
  {
    id: "agi",
    term: "AGI",
    definition: "Artificial General Intelligence - a theoretical form of AI that matches or exceeds human cognitive abilities across all domains, not just specific tasks.",
    category: "Ethics",
    examples: ["Human-level reasoning", "General problem solving", "Cross-domain learning"]
  },
  {
    id: "model-drift",
    term: "Model Drift",
    definition: "The degradation of a machine learning model's performance over time as the real-world data it encounters differs from its training data.",
    category: "Ethics",
    examples: ["Changing user behavior", "Market conditions shift", "Seasonal variations"]
  },
  {
    id: "hallucination",
    term: "Hallucination",
    definition: "When an AI model generates information that appears plausible but is actually false or not supported by its training data, particularly common in language models.",
    category: "Ethics",
    examples: ["False facts in text generation", "Invented citations", "Confident wrong answers"]
  }
];

export const categories = ["All", "Machine Learning", "NLP", "Computer Vision", "Ethics"];
