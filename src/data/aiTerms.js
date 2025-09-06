// AI Glossary Terms - Comprehensive Collection
export const aiTerms = [
  // Foundational AI
  {
    id: 1,
    term: "Artificial Intelligence",
    definition: "Think of it as making computers smart enough to do things that usually need human brains - like recognizing your face, understanding what you're saying, or deciding what movie you might like.",
    category: "Foundational AI"
  },
  {
    id: 2,
    term: "Artificial General Intelligence",
    definition: "This is the holy grail of AI - a computer that's as smart as a human in every way. It doesn't exist yet, but it would be able to learn any skill, solve any problem, and be creative just like we are.",
    category: "Foundational AI"
  },
  {
    id: 3,
    term: "Algorithm",
    definition: "Basically a recipe for computers. Just like following steps to bake a cake, an algorithm is a step-by-step guide that tells a computer exactly what to do to solve a problem.",
    category: "Foundational AI"
  },
  {
    id: 4,
    term: "Cognitive Computing",
    definition: "Computers that try to think like humans do - making sense of messy information, learning from experience, and even having something like intuition when solving problems.",
    category: "Foundational AI"
  },
  
  // Machine Learning
  {
    id: 5,
    term: "Machine Learning",
    definition: "Instead of programming every possible answer, you show the computer lots of examples and let it figure out the patterns on its own. Like teaching a kid to recognize cats by showing them thousands of cat photos.",
    category: "Machine Learning"
  },
  {
    id: 6,
    term: "Deep Learning",
    definition: "Machine learning with extra layers - imagine a stack of filters where each layer learns something more complex. The first layer might see edges, the next sees shapes, and eventually it recognizes entire objects.",
    category: "Machine Learning"
  },
  {
    id: 7,
    term: "Neural Network",
    definition: "A computer system loosely inspired by how our brain works - lots of simple processing units connected together that pass information around until they figure out the right answer.",
    category: "Machine Learning"
  },
  {
    id: 8,
    term: "Supervised Learning",
    definition: "Learning with a teacher. You give the computer examples with the right answers (like 'this is a dog', 'this is a cat') so it can learn to identify new examples correctly.",
    category: "Machine Learning"
  },
  {
    id: 9,
    term: "Unsupervised Learning",
    definition: "Learning without a teacher. You give the computer a bunch of data and say 'find interesting patterns' - like discovering that customers who buy bread also tend to buy butter.",
    category: "Machine Learning"
  },
  {
    id: 10,
    term: "Reinforcement Learning",
    definition: "Learning through trial and error, just like training a pet. The AI tries different actions and gets rewards for good choices and penalties for bad ones until it learns the best strategy.",
    category: "Machine Learning"
  },
  {
    id: 11,
    term: "Convolutional Neural Network",
    definition: "A special type of AI that's really good at understanding images. It scans pictures in small pieces and builds up understanding from details to the big picture - perfect for recognizing faces or objects.",
    category: "Machine Learning"
  },
  {
    id: 12,
    term: "Recurrent Neural Network",
    definition: "A type of AI with memory that's great for understanding sequences. Like reading a sentence, it remembers what came before to understand what comes next - perfect for things like language and time series.",
    category: "Machine Learning"
  },
  
  // Natural Language Processing
  {
    id: 13,
    term: "Natural Language Processing",
    definition: "Teaching computers to understand human language - not just the words, but what we actually mean. It's what lets Siri understand your questions or Google Translate convert languages.",
    category: "Natural Language Processing"
  },
  {
    id: 14,
    term: "Large Language Model",
    definition: "An AI that's read basically the entire internet and learned how language works. Think ChatGPT - it can write, answer questions, and chat because it's seen millions of examples of human text.",
    category: "Natural Language Processing"
  },
  {
    id: 15,
    term: "Transformer",
    definition: "The breakthrough AI architecture that powers ChatGPT and most modern language models. It's really good at paying attention to the important parts of text, like focusing on key words in a sentence.",
    category: "Natural Language Processing"
  },
  {
    id: 16,
    term: "GPT",
    definition: "The famous AI behind ChatGPT! It stands for 'Generative Pre-trained Transformer' - basically an AI that learned to predict what word comes next by reading tons of text, and got really good at writing like humans.",
    category: "Natural Language Processing"
  },
  {
    id: 17,
    term: "BERT",
    definition: "Google's AI that reads text from both directions at once (like reading a sentence forwards and backwards) to really understand context. It's what makes Google Search so good at understanding what you mean.",
    category: "Natural Language Processing"
  },
  {
    id: 18,
    term: "Tokenization",
    definition: "Breaking text into bite-sized pieces that AI can understand. Like cutting a sentence into individual words, or even smaller chunks. It's like preparing ingredients before cooking.",
    category: "Natural Language Processing"
  },
  {
    id: 19,
    term: "Sentiment Analysis",
    definition: "Teaching AI to read emotions in text - whether a review is positive or negative, if a tweet is angry or happy. It's like giving computers emotional intelligence for reading.",
    category: "Natural Language Processing"
  },
  
  // Computer Vision
  {
    id: 20,
    term: "Computer Vision",
    definition: "Giving computers eyes and teaching them to see like we do. They can recognize objects, read text, identify people, and understand what's happening in photos and videos.",
    category: "Computer Vision"
  },
  {
    id: 21,
    term: "Image Recognition",
    definition: "AI that can look at a photo and tell you what's in it - 'That's a dog!', 'That's a car!', 'That's your grandma!' It's like having a really smart friend who never forgets what anything looks like.",
    category: "Computer Vision"
  },
  {
    id: 22,
    term: "Object Detection",
    definition: "Not just knowing what's in a picture, but pointing out exactly where it is. Like saying 'There's a cat in the bottom left corner and a dog by the tree.' Super useful for self-driving cars!",
    category: "Computer Vision"
  },
  {
    id: 23,
    term: "Facial Recognition",
    definition: "AI that recognizes faces - like how your phone unlocks when it sees you, or how Facebook suggests who to tag in photos. It learns the unique features that make your face yours.",
    category: "Computer Vision"
  },
  {
    id: 24,
    term: "Optical Character Recognition",
    definition: "Teaching computers to read text in images, like scanning a document with your phone and having it convert to editable text. It's like giving AI reading glasses for pictures.",
    category: "Computer Vision"
  },
  
  // Generative AI
  {
    id: 25,
    term: "Generative AI",
    definition: "AI that creates new stuff - write poems, generate images, compose music, or even code. It's learned from millions of examples and now makes original content that feels human-made.",
    category: "Generative AI"
  },
  {
    id: 26,
    term: "Generative Adversarial Network",
    definition: "Two AIs competing against each other - one tries to create fake images, the other tries to spot the fakes. They keep getting better until the fakes are so good you can't tell the difference!",
    category: "Generative AI"
  },
  {
    id: 27,
    term: "Diffusion Model",
    definition: "The AI behind tools like Stable Diffusion and DALL-E. It starts with pure noise and gradually shapes it into an image, like a sculptor slowly revealing a statue from a block of marble.",
    category: "Generative AI"
  },
  {
    id: 28,
    term: "Prompt Engineering",
    definition: "The art of talking to AI in just the right way to get what you want. Like knowing the magic words that make ChatGPT or DALL-E give you exactly the result you're looking for.",
    category: "Generative AI"
  },
  {
    id: 29,
    term: "DALL-E",
    definition: "OpenAI's famous image generator that creates amazing artwork from text descriptions. Just type 'a cat wearing a space suit riding a unicorn' and boom - you've got a picture!",
    category: "Generative AI"
  },
  {
    id: 30,
    term: "Stable Diffusion",
    definition: "The open-source image generator that anyone can use and modify. It's like DALL-E's free cousin that you can run on your own computer and customize however you want.",
    category: "Generative AI"
  },
  
  // Robotics
  {
    id: 31,
    term: "Robotics",
    definition: "Building machines that can move around and do physical tasks in the real world. Think everything from Roomba vacuum cleaners to factory assembly arms to Mars rovers.",
    category: "Robotics"
  },
  {
    id: 32,
    term: "Autonomous Systems",
    definition: "Machines that can make their own decisions and act independently without humans controlling them every step of the way. Like self-driving cars or drones that deliver packages.",
    category: "Robotics"
  },
  {
    id: 33,
    term: "Robot Operating System",
    definition: "Like Windows or macOS, but for robots! It's the software that helps all the different parts of a robot work together - sensors, motors, AI brains, and everything else.",
    category: "Robotics"
  },
  
  // AI Ethics
  {
    id: 34,
    term: "AI Ethics",
    definition: "The important questions about right and wrong when it comes to AI. Like: Should AI be allowed to make decisions about people's lives? How do we make sure AI is fair to everyone?",
    category: "AI Ethics"
  },
  {
    id: 35,
    term: "Algorithmic Bias",
    definition: "When AI accidentally learns to be unfair to certain groups of people. Like if a hiring AI was trained mostly on male resumes, it might unfairly prefer male candidates even when they're equally qualified.",
    category: "AI Ethics"
  },
  {
    id: 36,
    term: "Explainable AI",
    definition: "AI that can explain its reasoning in human terms. Instead of just saying 'Trust me, this person is a credit risk,' it explains 'because of their income, debt ratio, and payment history.'",
    category: "AI Ethics"
  },
  {
    id: 37,
    term: "AI Alignment",
    definition: "Making sure AI wants the same things humans want. The challenge is programming an AI to pursue goals that actually help humanity, not just follow instructions literally like a genie in a lamp.",
    category: "AI Ethics"
  },
  
  // Technical Concepts
  {
    id: 38,
    term: "Training Data",
    definition: "All the examples you show an AI to teach it what you want. Like showing a kid thousands of pictures of cats and dogs labeled correctly so they can learn to tell the difference.",
    category: "Technical Concepts"
  },
  {
    id: 39,
    term: "Overfitting",
    definition: "When an AI memorizes the training examples too perfectly but can't handle new situations. Like a student who memorizes practice tests but fails the real exam because the questions are slightly different.",
    category: "Technical Concepts"
  },
  {
    id: 40,
    term: "Gradient Descent",
    definition: "The way AI learns by making tiny adjustments to get better results. Imagine trying to find the bottom of a hill in the dark - you feel which way is downhill and take small steps until you reach the bottom.",
    category: "Technical Concepts"
  },
  {
    id: 41,
    term: "Backpropagation",
    definition: "How neural networks learn from their mistakes. When the AI gets something wrong, it traces back through all its layers to figure out what went wrong and fixes it.",
    category: "Technical Concepts"
  },
  {
    id: 42,
    term: "Hyperparameter",
    definition: "The settings you choose before training an AI, like how fast it should learn or how complex it should be. Think of them as the recipe settings - oven temperature, cooking time, etc.",
    category: "Technical Concepts"
  },
  {
    id: 43,
    term: "Feature Engineering",
    definition: "The art of preparing and transforming raw data into useful inputs for AI. Like a chef prepping ingredients - you don't just throw raw vegetables into a pot, you chop, season, and combine them thoughtfully.",
    category: "Technical Concepts"
  },
  {
    id: 44,
    term: "Cross-Validation",
    definition: "A testing method that splits your data into multiple pieces to see how well your AI really performs. Like taking several practice tests to see if you're truly ready for the final exam.",
    category: "Technical Concepts"
  },
  {
    id: 45,
    term: "Ensemble Learning",
    definition: "Combining multiple AI models to get better results than any single model could achieve. Like asking several experts and combining their opinions instead of trusting just one person.",
    category: "Technical Concepts"
  },
  {
    id: 46,
    term: "Regularization",
    definition: "Techniques to prevent AI from memorizing training data too perfectly by adding some constraints or penalties. Like adding speed bumps to prevent overeager learning that hurts real-world performance.",
    category: "Technical Concepts"
  },
  {
    id: 47,
    term: "Dropout",
    definition: "A training trick where you randomly ignore some neurons during learning to prevent overfitting. Like practicing with some players benched so the team doesn't become too dependent on any one player.",
    category: "Technical Concepts"
  },
  {
    id: 48,
    term: "Activation Function",
    definition: "The decision-maker in each brain cell of a neural network. It decides whether the neuron should get excited and pass information along, like a bouncer deciding who gets into the club.",
    category: "Technical Concepts"
  },
  {
    id: 49,
    term: "Loss Function",
    definition: "The AI's report card that tells it how wrong it was. The bigger the number, the worse it did. The AI tries to make this number as small as possible during training.",
    category: "Technical Concepts"
  },
  {
    id: 50,
    term: "Epoch",
    definition: "One round of training where the AI looks at every single example in your training data once. Like reading through an entire textbook one time before taking the test.",
    category: "Technical Concepts"
  },
  {
    id: 51,
    term: "Batch Size",
    definition: "How many examples the AI looks at before updating its brain. Like studying 32 flashcards at a time before pausing to think about what you learned.",
    category: "Technical Concepts"
  },
  {
    id: 52,
    term: "Learning Rate",
    definition: "How big steps the AI takes when learning. Too big and it might miss the answer by overshooting, too small and it learns really slowly. It's like adjusting your walking speed.",
    category: "Technical Concepts"
  },
  {
    id: 53,
    term: "Confusion Matrix",
    definition: "A table used to evaluate the performance of classification algorithms.",
    category: "Technical Concepts"
  },
  {
    id: 54,
    term: "Precision",
    definition: "The ratio of correctly predicted positive observations to total predicted positives.",
    category: "Technical Concepts"
  },
  {
    id: 55,
    term: "Recall",
    definition: "The ratio of correctly predicted positive observations to all actual positives.",
    category: "Technical Concepts"
  },
  {
    id: 56,
    term: "F1 Score",
    definition: "The harmonic mean of precision and recall, providing a single metric for model performance.",
    category: "Technical Concepts"
  },

  // Additional Foundational AI Terms
  {
    id: 57,
    term: "Expert System",
    definition: "AI that tries to be like a human expert in one specific field - like a computer doctor that diagnoses diseases or a digital mechanic that troubleshoots car problems. It knows a lot about one thing.",
    category: "Foundational AI"
  },
  {
    id: 58,
    term: "Knowledge Base",
    definition: "The AI's encyclopedia - all the facts, rules, and information it knows stored in an organized way. Like having a perfectly organized library in the computer's brain.",
    category: "Foundational AI"
  },
  {
    id: 59,
    term: "Inference Engine",
    definition: "The thinking part of an AI system that connects the dots. It takes what it knows from the knowledge base and figures out new conclusions, like a digital detective solving mysteries.",
    category: "Foundational AI"
  },
  {
    id: 60,
    term: "Turing Test",
    definition: "The famous test where you chat with something and try to figure out if it's human or AI. If you can't tell the difference, the AI passes! It's like the ultimate AI disguise contest.",
    category: "Foundational AI"
  },
  {
    id: 61,
    term: "Weak AI",
    definition: "AI that's really good at one specific thing but can't do anything else. Like a chess master who can only play chess, or Siri who can set timers but can't actually understand your life.",
    category: "Foundational AI"
  },
  {
    id: 62,
    term: "Strong AI",
    definition: "The dream AI that would be as smart as humans in every way - creative, emotional, flexible, learning anything. It doesn't exist yet, but it's what sci-fi movies imagine.",
    category: "Foundational AI"
  },
  {
    id: 63,
    term: "Symbolic AI",
    definition: "The old-school way of building AI using symbols and logic rules, like programming with 'IF this THEN that' statements. Think of it as teaching AI through a rulebook rather than examples.",
    category: "Foundational AI"
  },
  {
    id: 64,
    term: "Connectionism",
    definition: "The newer approach to AI that copies how brains work - lots of simple units connected together in networks. Instead of rules, it learns patterns from examples, like how we actually think.",
    category: "Foundational AI"
  },

  // Additional Machine Learning Terms
  {
    id: 65,
    term: "Decision Tree",
    definition: "AI that makes decisions like playing 20 questions. It asks yes/no questions about your data ('Is age > 30?', 'Is income > 50k?') and follows the answers down branches until it reaches a decision.",
    category: "Machine Learning"
  },
  {
    id: 66,
    term: "Random Forest",
    definition: "Instead of trusting one decision tree, this creates a whole forest of them and takes a vote. Like asking 100 doctors for their opinion and going with the majority - usually more accurate than just one!",
    category: "Machine Learning"
  },
  {
    id: 67,
    term: "Support Vector Machine",
    definition: "Imagine trying to separate cats and dogs with a line on a graph. SVM finds the best possible line that gives the most space between the two groups - like drawing the perfect fence between neighbors.",
    category: "Machine Learning"
  },
  {
    id: 68,
    term: "K-Means Clustering",
    definition: "Like organizing your music into playlists without knowing the genres beforehand. It groups similar songs together and you end up with rock, pop, jazz clusters automatically.",
    category: "Machine Learning"
  },
  {
    id: 69,
    term: "Linear Regression",
    definition: "Drawing the best straight line through scattered dots on a graph to predict trends. Like predicting house prices based on size - bigger houses generally cost more, and the line shows that relationship.",
    category: "Machine Learning"
  },
  {
    id: 70,
    term: "Logistic Regression",
    definition: "For yes/no questions like 'Will it rain?' or 'Is this email spam?' It gives you a probability curve instead of a straight line - more realistic for true/false predictions.",
    category: "Machine Learning"
  },
  {
    id: 71,
    term: "Naive Bayes",
    definition: "A simple but effective AI classifier that assumes all features are independent (which is often wrong, hence 'naive'). Despite this simplistic assumption, it works surprisingly well for things like spam detection.",
    category: "Machine Learning"
  },
  {
    id: 72,
    term: "Gradient Boosting",
    definition: "A teamwork approach where multiple weak AI models work together, each one learning to fix the mistakes of the previous ones. Like having a chain of editors, each improving the work.",
    category: "Machine Learning"
  },
  {
    id: 73,
    term: "XGBoost",
    definition: "The supercharged version of gradient boosting that's often the go-to choice for winning machine learning competitions. It's fast, accurate, and handles messy data really well.",
    category: "Machine Learning"
  },
  {
    id: 74,
    term: "Principal Component Analysis",
    definition: "A data compression technique that finds the most important directions in your data and throws away the less important ones. Like summarizing a 3D object by looking at its most informative 2D shadow.",
    category: "Machine Learning"
  },
  {
    id: 75,
    term: "K-Nearest Neighbors",
    definition: "One of the simplest AI algorithms - it classifies things based on what their neighbors are like. If you're surrounded by dogs, you're probably a dog too. The 'k' is how many neighbors to ask.",
    category: "Machine Learning"
  },
  {
    id: 76,
    term: "LSTM",
    definition: "A smart type of AI memory that remembers important things for a long time but forgets the unimportant stuff. Perfect for understanding long sentences or predicting stock prices over time.",
    category: "Machine Learning"
  },
  {
    id: 77,
    term: "GRU",
    definition: "LSTM's simpler cousin that does almost the same job with fewer moving parts. Like choosing a basic smartphone over a flagship - sometimes simpler is better and faster.",
    category: "Machine Learning"
  },
  {
    id: 78,
    term: "Autoencoder",
    definition: "An AI that learns to compress and uncompress data, like a smart zip file. It squashes information down to the essential bits, then rebuilds it. Great for finding what's really important in data.",
    category: "Machine Learning"
  },
  {
    id: 79,
    term: "Variational Autoencoder",
    definition: "An autoencoder that can also generate new stuff by tweaking the compressed version. Like having a magic photo editor that can create new faces by mixing features from the photos it's seen.",
    category: "Machine Learning"
  },

  // Additional Natural Language Processing Terms
  {
    id: 80,
    term: "Word Embedding",
    definition: "A way to turn words into numbers that capture their meaning. Words with similar meanings get similar numbers, so 'king' and 'queen' would be close together in this number space.",
    category: "Natural Language Processing"
  },
  {
    id: 81,
    term: "Word2Vec",
    definition: "Google's famous method for teaching AI what words mean by looking at their neighbors. It learns that 'doctor' and 'nurse' are related because they appear in similar sentences.",
    category: "Natural Language Processing"
  },
  {
    id: 82,
    term: "GloVe",
    definition: "Stanford's approach to word embeddings that looks at how often words appear together across huge amounts of text. Like building a friendship map of words based on how much they hang out together.",
    category: "Natural Language Processing"
  },
  {
    id: 83,
    term: "Named Entity Recognition",
    definition: "Teaching AI to spot important things in text like people's names, companies, dates, and places. Like having a highlighter that automatically marks all the key information in documents.",
    category: "Natural Language Processing"
  },
  {
    id: 84,
    term: "Part-of-Speech Tagging",
    definition: "Teaching AI grammar by labeling each word as a noun, verb, adjective, etc. Like having an English teacher that instantly marks up every sentence with grammatical roles.",
    category: "Natural Language Processing"
  },
  {
    id: 85,
    term: "Machine Translation",
    definition: "What Google Translate does - automatically converting text from one language to another. Modern versions understand context, not just word-for-word replacement.",
    category: "Natural Language Processing"
  },
  {
    id: 86,
    term: "Text Summarization",
    definition: "AI that reads long documents and creates short summaries with the key points. Like having a smart assistant that can turn a 20-page report into a one-page executive summary.",
    category: "Natural Language Processing"
  },
  {
    id: 87,
    term: "Question Answering",
    definition: "AI that can read text and answer questions about it, just like a really smart student. Ask it 'What's the capital of France?' after giving it a geography textbook, and it'll find the answer.",
    category: "Natural Language Processing"
  },
  {
    id: 88,
    term: "Information Extraction",
    definition: "AI that turns messy, unstructured text into organized data. Like reading through thousands of resumes and automatically creating a neat spreadsheet with names, skills, and experience.",
    category: "Natural Language Processing"
  },
  {
    id: 89,
    term: "Coreference Resolution",
    definition: "Teaching AI to understand when different words refer to the same thing. Like knowing that 'John', 'he', 'the doctor', and 'Mr. Smith' all refer to the same person in a story.",
    category: "Natural Language Processing"
  },
  {
    id: 90,
    term: "Language Model",
    definition: "An AI that has learned how language works by reading tons of text. It can predict what word comes next, like when your phone suggests the next word as you type.",
    category: "Natural Language Processing"
  },
  {
    id: 91,
    term: "Attention Mechanism",
    definition: "The AI's way of focusing on the important parts while ignoring the rest. Like how you focus on keywords when skimming an article - the AI learns to do the same thing.",
    category: "Natural Language Processing"
  },
  {
    id: 92,
    term: "BLEU Score",
    definition: "A way to grade how good machine translation is by comparing it to human translations. Like having multiple teachers grade the same test and averaging their scores.",
    category: "Natural Language Processing"
  },

  // Additional Computer Vision Terms
  {
    id: 93,
    term: "Feature Detection",
    definition: "Teaching AI to spot important visual clues in images like edges, corners, and textures. Like training someone to quickly identify the key details that make a face recognizable.",
    category: "Computer Vision"
  },
  {
    id: 94,
    term: "Image Segmentation",
    definition: "Cutting up an image into meaningful pieces, like separating the sky, trees, and road in a landscape photo. Each piece gets labeled so the AI knows what's what.",
    category: "Computer Vision"
  },
  {
    id: 95,
    term: "Edge Detection",
    definition: "Finding the outlines and boundaries in images - like tracing around objects with a digital pencil. It's how AI figures out where one thing ends and another begins.",
    category: "Computer Vision"
  },
  {
    id: 96,
    term: "Template Matching",
    definition: "Like playing 'Where's Waldo?' - the AI has a small reference image and searches through a larger image to find matching parts. Simple but effective for finding specific objects.",
    category: "Computer Vision"
  },
  {
    id: 97,
    term: "Histogram of Oriented Gradients",
    definition: "A fancy way to describe the shape and texture patterns in images that helps AI recognize objects. Think of it as creating a fingerprint for different types of visual patterns.",
    category: "Computer Vision"
  },
  {
    id: 98,
    term: "SIFT",
    definition: "A classic computer vision technique that finds distinctive features in images that stay recognizable even if the image is rotated, scaled, or slightly changed. Like finding landmarks that are always recognizable.",
    category: "Computer Vision"
  },
  {
    id: 99,
    term: "SURF",
    definition: "A faster version of SIFT that does basically the same job - finding distinctive features in images - but with better performance. Like SIFT's speedier younger sibling.",
    category: "Computer Vision"
  },
  {
    id: 100,
    term: "Stereo Vision",
    definition: "Using two cameras (like our two eyes) to see depth and create 3D understanding from flat images. It's how robots and self-driving cars can tell how far away things are.",
    category: "Computer Vision"
  },
  {
    id: 101,
    term: "Optical Flow",
    definition: "Tracking how things move in videos by following the motion of pixels from frame to frame. Like watching a ball fly through the air and predicting where it'll go next.",
    category: "Computer Vision"
  },
  {
    id: 102,
    term: "Image Registration",
    definition: "Lining up multiple images of the same thing so they match perfectly, like overlaying before/after photos to see what changed. Essential for medical scans and satellite imagery.",
    category: "Computer Vision"
  },
  {
    id: 103,
    term: "YOLO",
    definition: "A super-fast object detection AI that looks at an image once and immediately spots all the objects in it. Perfect for real-time applications like security cameras or self-driving cars.",
    category: "Computer Vision"
  },
  {
    id: 104,
    term: "R-CNN",
    definition: "An older but important object detection method that first proposes regions that might contain objects, then classifies them. Like having a smart guess about where to look before examining closely.",
    category: "Computer Vision"
  },

  // Additional Generative AI Terms
  {
    id: 105,
    term: "Autoregressive Model",
    definition: "A model that generates sequences by predicting each element based on previous elements.",
    category: "Generative AI"
  },
  {
    id: 106,
    term: "Variational Inference",
    definition: "A mathematical trick for dealing with complex probability problems in generative AI. Think of it as finding a simpler approximation when the exact answer would take forever to calculate.",
    category: "Generative AI"
  },
  {
    id: 107,
    term: "Flow-based Model",
    definition: "A type of generative AI that learns to transform simple random noise into complex data through a series of reversible steps. Like having a magic recipe that can turn flour into cake and cake back into flour.",
    category: "Generative AI"
  },
  {
    id: 108,
    term: "StyleGAN",
    definition: "NVIDIA's famous AI for creating incredibly realistic fake faces and images. It can even mix and match styles, like combining one person's hair with another's facial features.",
    category: "Generative AI"
  },
  {
    id: 109,
    term: "CLIP",
    definition: "OpenAI's groundbreaking model that understands both images and text together. It's what lets DALL-E know that 'a red bicycle' should actually show a red bicycle, not a blue car.",
    category: "Generative AI"
  },
  {
    id: 110,
    term: "ControlNet",
    definition: "An add-on for image generators that gives you precise control over the output. Want the person in your generated image to be in a specific pose? ControlNet makes it happen.",
    category: "Generative AI"
  },
  {
    id: 111,
    term: "LoRA",
    definition: "A clever way to customize large AI models without retraining the whole thing. Like adding a small modification kit to your car instead of rebuilding the entire engine.",
    category: "Generative AI"
  },
  {
    id: 112,
    term: "Inpainting",
    definition: "AI that can fill in missing parts of images, like magically removing people from photos or fixing damaged artwork. Point to what you want gone, and the AI fills it in seamlessly.",
    category: "Generative AI"
  },
  {
    id: 113,
    term: "Text-to-Image",
    definition: "The magic behind tools like DALL-E and Midjourney - type a description and get a picture. 'A purple elephant riding a motorcycle' becomes an actual image in seconds.",
    category: "Generative AI"
  },
  {
    id: 114,
    term: "Few-Shot Learning",
    definition: "AI that can learn new tasks from just a handful of examples, like humans do. Show it 3 pictures of a new animal species and it can recognize more of them.",
    category: "Generative AI"
  },

  // Additional Robotics Terms
  {
    id: 115,
    term: "Simultaneous Localization and Mapping",
    definition: "How robots figure out where they are AND build a map at the same time - like exploring a dark house with a flashlight while drawing a floor plan. Essential for autonomous navigation.",
    category: "Robotics"
  },
  {
    id: 116,
    term: "Path Planning",
    definition: "How robots figure out the best route from point A to point B while avoiding obstacles. Like GPS for robots, but in 3D space and considering the robot's physical limitations.",
    category: "Robotics"
  },
  {
    id: 117,
    term: "Inverse Kinematics",
    definition: "Working backwards from where you want the robot's hand to be to figure out how all the joints should move. Like solving a puzzle to get the robot arm to reach the right spot.",
    category: "Robotics"
  },
  {
    id: 118,
    term: "Forward Kinematics",
    definition: "The easier direction - if you know all the joint angles, calculating where the robot's hand ends up. Like following a recipe step by step to see what you get.",
    category: "Robotics"
  },
  {
    id: 119,
    term: "Motion Planning",
    definition: "Breaking down complex robot movements into smaller, safe steps. Like planning how to move a heavy couch through doorways without hitting walls or people.",
    category: "Robotics"
  },
  {
    id: 120,
    term: "Sensor Fusion",
    definition: "Combining information from multiple robot sensors to get a clearer picture. Like using your eyes, ears, and touch together to understand your surroundings better than any one sense alone.",
    category: "Robotics"
  },
  {
    id: 121,
    term: "Actuator",
    definition: "The robot's muscles - the motors and mechanisms that actually make things move. Without actuators, a robot would just be a very expensive computer that can't do anything physical.",
    category: "Robotics"
  },
  {
    id: 122,
    term: "End Effector",
    definition: "The robot's hand or tool at the end of its arm - could be a gripper, welding torch, paint sprayer, or any tool designed for the specific job the robot needs to do.",
    category: "Robotics"
  },
  {
    id: 123,
    term: "Swarm Robotics",
    definition: "Coordinating lots of simple robots to work together like a bee hive or ant colony. Each robot is basic, but together they can accomplish complex tasks through teamwork.",
    category: "Robotics"
  },
  {
    id: 124,
    term: "Human-Robot Interaction",
    definition: "Making robots that can work naturally and safely with humans. It's about communication, safety, and making robots that people actually want to be around.",
    category: "Robotics"
  },

  // Additional AI Ethics Terms
  {
    id: 125,
    term: "Fairness in AI",
    definition: "Making sure AI systems don't discriminate or treat people unfairly based on race, gender, age, or other characteristics. It's harder than it sounds because bias can hide in data.",
    category: "AI Ethics"
  },
  {
    id: 126,
    term: "Transparency",
    definition: "Making AI systems open and understandable, not black boxes. Users should know how decisions are made, especially for important things like loan approvals or medical diagnoses.",
    category: "AI Ethics"
  },
  {
    id: 127,
    term: "Accountability",
    definition: "Making sure someone is responsible when AI makes mistakes. If an AI car crashes or an AI hiring system discriminates, there should be clear accountability, not just 'the algorithm did it.'",
    category: "AI Ethics"
  },
  {
    id: 128,
    term: "Privacy by Design",
    definition: "Building privacy protection into AI systems from the very beginning, not adding it as an afterthought. Like building a house with locks on the doors, not trying to add security later.",
    category: "AI Ethics"
  },
  {
    id: 129,
    term: "Differential Privacy",
    definition: "A mathematical way to protect individual privacy in datasets while still allowing useful analysis. Like adding just enough noise to hide individuals but keep the overall patterns clear.",
    category: "AI Ethics"
  },
  {
    id: 130,
    term: "Federated Learning",
    definition: "Training AI models without centralizing everyone's data. Like having a study group where everyone learns from shared knowledge but keeps their personal notes private.",
    category: "AI Ethics"
  },
  {
    id: 131,
    term: "AI Governance",
    definition: "The rules, policies, and oversight needed to make sure AI development and use benefits everyone. Like having traffic laws for the AI highway to prevent accidents and ensure fair access.",
    category: "AI Ethics"
  },
  {
    id: 132,
    term: "Adversarial Examples",
    definition: "Specially crafted inputs designed to trick AI systems. Like putting a tiny sticker on a stop sign that makes an AI car think it's a speed limit sign - small changes, big problems.",
    category: "AI Ethics"
  },
  {
    id: 133,
    term: "AI Safety",
    definition: "Research into making sure AI systems are safe and beneficial, especially as they get more powerful. It's about preventing AI from causing harm, even when trying to help.",
    category: "AI Ethics"
  },
  {
    id: 134,
    term: "Robustness",
    definition: "How well an AI system performs when things don't go exactly as planned. Like a GPS that still works when there's construction, bad weather, or an unexpected road closure.",
    category: "AI Ethics"
  },
  {
    id: 135,
    term: "AI Interpretability",
    definition: "How well humans can understand and explain what an AI is thinking. Instead of just saying 'the computer said no,' being able to explain 'here's exactly why the computer made that decision.'",
    category: "AI Ethics"
  }
];

// Categories for filtering
export const categories = [
  "All",
  "Foundational AI",
  "Machine Learning", 
  "Natural Language Processing",
  "Computer Vision",
  "Generative AI",
  "Robotics",
  "AI Ethics",
  "Technical Concepts"
];
