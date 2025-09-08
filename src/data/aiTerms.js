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
    term: "Artificial General Intelligence (AGI)",
    definition: "A not‑yet‑real kind of AI that could learn and do almost any mental task a person can. Example: one system that can pass a medical exam, write music, and fix a bike by reading the manual—without special retraining.",
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
    definition: "An older label for \"smart\" business software that tried to mimic human reasoning. Today we just say AI for language, vision, etc. Example: an insurance chatbot answering policy questions.",
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
    term: "Large Language Model (LLM)",
    definition: "A program trained on lots of text to learn how words fit together. It answers by predicting the next words. Example: you ask \"Explain photosynthesis,\" it writes a short paragraph.",
    category: "Natural Language Processing"
  },
  {
    id: 15,
    term: "Transformer",
    definition: "A neural design that \"pays attention\" to the most important parts of what it reads (or sees). Example: in \"The dog chased the ball,\" it focuses on \"dog\" and \"ball\" to understand \"chased.\"",
    category: "Natural Language Processing"
  },
  {
    id: 16,
    term: "GPT",
    definition: "\"Generative Pre‑trained Transformer\": first it learns from lots of text; then it's tuned to be helpful and safe. Example: Chat‑style answers, code help, or summaries.",
    category: "Natural Language Processing"
  },
  {
    id: 17,
    term: "BERT",
    definition: "A model that reads left and right for context. Example: it knows \"bank\" in \"river bank\" means shore, not money.",
    category: "Natural Language Processing"
  },
  {
    id: 18,
    term: "Tokenization",
    definition: "Cutting text into pieces the model can handle (words or sub‑words). Example: \"sunshine\" → \"sun\" + \"shine.\"",
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
    definition: "Labels what's in a picture. Example: \"dog,\" \"car,\" \"tree.\"",
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
    definition: "Matches a face to a person. Example: unlocking your phone with your face. (Note: rules and limits apply in many places.)",
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
    term: "Generative Adversarial Network (GAN)",
    definition: "Two models train together: one makes fakes, the other catches them. Example: the generator learns to create realistic faces.",
    category: "Generative AI"
  },
  {
    id: 27,
    term: "Diffusion Model",
    definition: "An image maker that starts with \"snowy\" noise and cleans it until a clear picture appears. Example: type \"a red bicycle on the beach\"; it paints one from noise.",
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
  },

  // Additional Foundational AI Terms
  {
    id: 136,
    term: "Abductive Inference",
    definition: "Smart guessing based on incomplete information - like how Sherlock Holmes figures out who did it from just a few clues. It's reasoning backwards from effects to likely causes.",
    category: "Foundational AI"
  },
  {
    id: 137,
    term: "Abductive Reasoning",
    definition: "The art of making educated guesses when you don't have all the facts. Like seeing wet streets and concluding it probably rained, even though you didn't see it happen.",
    category: "Foundational AI"
  },
  {
    id: 138,
    term: "Abductive Logic Programming",
    definition: "A programming approach that lets computers make smart guesses about missing information, like filling in the blanks in a story with the most logical explanations.",
    category: "Foundational AI"
  },
  {
    id: 139,
    term: "Agent Architecture",
    definition: "The blueprint for how an AI agent is built - like the floor plan of a house, it shows how all the thinking, sensing, and acting parts connect together.",
    category: "Foundational AI"
  },
  {
    id: 140,
    term: "Belief-Desire-Intention Model",
    definition: "A way to build AI that thinks like humans do - it has beliefs about the world, desires for what it wants, and intentions about what to do next. Like having goals and making plans.",
    category: "Foundational AI"
  },
  {
    id: 141,
    term: "Belief Revision",
    definition: "How AI updates its understanding when new information contradicts what it thought was true. Like changing your mind about someone when you learn new facts about them.",
    category: "Foundational AI"
  },
  {
    id: 142,
    term: "Cognitive Architecture",
    definition: "The overall design of how artificial minds work - the cognitive equivalent of computer hardware, but for thinking, learning, and understanding instead of just computing.",
    category: "Foundational AI"
  },
  {
    id: 143,
    term: "Cognitive Science",
    definition: "The study of how minds work - both human and artificial. It's like psychology meets computer science to understand thinking, learning, and consciousness itself.",
    category: "Foundational AI"
  },
  {
    id: 144,
    term: "Common Sense Reasoning",
    definition: "Teaching AI the obvious stuff that humans just know - like water is wet, objects fall down, and people get sad when bad things happen. Surprisingly hard to program!",
    category: "Foundational AI"
  },
  {
    id: 145,
    term: "Deliberative Agent",
    definition: "An AI that thinks before it acts, like a chess player considering all possible moves. It plans, weighs options, and makes thoughtful decisions rather than just reacting instantly.",
    category: "Foundational AI"
  },
  {
    id: 146,
    term: "Embodied Cognition",
    definition: "The idea that thinking isn't just in the brain - our bodies and physical experiences shape how we understand the world. Like how touching things helps us learn about them.",
    category: "Foundational AI"
  },
  {
    id: 147,
    term: "Emergent Behavior",
    definition: "When simple parts working together create something surprisingly complex and smart. Like how ant colonies build amazing structures even though individual ants aren't architects.",
    category: "Foundational AI"
  },
  {
    id: 148,
    term: "Grounding",
    definition: "Connecting abstract concepts to real-world experiences. Like how the word 'red' means something because you've actually seen red things - not just read a definition of it.",
    category: "Foundational AI"
  },
  {
    id: 149,
    term: "Goal-Based Agent",
    definition: "An AI that works towards specific objectives, like a GPS that wants to get you to your destination. It makes decisions based on whether they help achieve its goals.",
    category: "Foundational AI"
  },
  {
    id: 150,
    term: "Intelligent Agent",
    definition: "Any AI system that can perceive its environment, think about it, and take actions to achieve goals. Like a smart robot butler that can see, think, and act independently.",
    category: "Foundational AI"
  },
  {
    id: 151,
    term: "Knowledge Acquisition",
    definition: "How AI systems learn and gather new information and skills. Like a student collecting knowledge from books, teachers, and experiences to become smarter over time.",
    category: "Foundational AI"
  },
  {
    id: 152,
    term: "Knowledge Engineering",
    definition: "The art of organizing and structuring information so AI can use it effectively. Like being a librarian for robot brains, making sure knowledge is findable and usable.",
    category: "Foundational AI"
  },
  {
    id: 153,
    term: "Knowledge Representation",
    definition: "How to store facts and ideas in a computer's brain so it can reason with them. Like choosing between filing cabinets, databases, or mind maps for organizing information.",
    category: "Foundational AI"
  },
  {
    id: 154,
    term: "Modularity",
    definition: "Building AI in separate, interchangeable pieces that can work together. Like LEGO blocks - you can swap parts in and out to create different capabilities without rebuilding everything.",
    category: "Foundational AI"
  },
  {
    id: 155,
    term: "Rational Agent",
    definition: "An AI that always tries to make the best possible decision given what it knows. Like an ideally logical person who never lets emotions or biases cloud their judgment.",
    category: "Foundational AI"
  },
  {
    id: 156,
    term: "Situation Calculus",
    definition: "A mathematical way to describe how actions change the world over time. Like writing a story where each chapter shows how the world is different after someone does something.",
    category: "Foundational AI"
  },
  {
    id: 157,
    term: "Synapse",
    definition: "The connection points between artificial neurons in AI networks, just like in real brains. These connections learn and strengthen to form memories and skills, like building pathways in the mind.",
    category: "Foundational AI"
  },
  {
    id: 158,
    term: "Theory of Mind",
    definition: "Understanding that others have their own thoughts, beliefs, and feelings different from yours. It's what lets you predict how someone will react or realize they might not know what you know.",
    category: "Foundational AI"
  },

  // Additional Machine Learning Terms
  {
    id: 159,
    term: "Ablation",
    definition: "Testing what happens when you remove parts of an AI system to see which pieces are actually important. Like taking ingredients out of a recipe one by one to see what each one does.",
    category: "Machine Learning"
  },
  {
    id: 160,
    term: "Adaptive Algorithm",
    definition: "AI that changes its approach based on what it learns, like a student who adjusts their study method when they figure out what works best for them.",
    category: "Machine Learning"
  },
  {
    id: 161,
    term: "Adaptive Neuro Fuzzy Inference System",
    definition: "A smart system that combines neural networks with fuzzy logic - it can learn from examples and handle uncertainty at the same time. Like having both book smarts and street smarts.",
    category: "Machine Learning"
  },
  {
    id: 162,
    term: "AlphaGo",
    definition: "Google's famous AI that became the first computer to beat world champions at Go, one of the most complex board games ever invented. It taught itself by playing millions of games.",
    category: "Machine Learning"
  },
  {
    id: 163,
    term: "Bayesian Network",
    definition: "A way to map out how different things influence each other with probabilities. Like a family tree of causes and effects, showing how rain affects traffic, which affects mood, and so on.",
    category: "Machine Learning"
  },
  {
    id: 164,
    term: "Bayesian Programming",
    definition: "Programming based on probability and uncertainty rather than absolute rules. It's like making decisions when you're not 100% sure about anything, which is most of real life.",
    category: "Machine Learning"
  },
  {
    id: 165,
    term: "Bayesian Statistics",
    definition: "A way of thinking about probability that updates beliefs as new evidence comes in. Like being a detective who changes theories as new clues appear.",
    category: "Machine Learning"
  },
  {
    id: 166,
    term: "Catastrophic Interference",
    definition: "When an AI forgets old skills while learning new ones, like a student who studies French so hard they start forgetting Spanish. A major challenge in lifelong learning.",
    category: "Machine Learning"
  },
  {
    id: 167,
    term: "Concept Drift",
    definition: "When the patterns in data slowly change over time, making old AI models less accurate. Like how slang evolves - what was cool yesterday might be cringe today.",
    category: "Machine Learning"
  },
  {
    id: 168,
    term: "Concept Learning",
    definition: "How AI figures out general ideas from specific examples. Like learning what 'furniture' means by seeing chairs, tables, and beds until you understand the concept.",
    category: "Machine Learning"
  },
  {
    id: 169,
    term: "Continuous Learning",
    definition: "AI that keeps learning new things without forgetting what it already knows. Like a student who can keep taking new classes without losing knowledge from previous ones.",
    category: "Machine Learning"
  },
  {
    id: 170,
    term: "Deep Neural Network",
    definition: "A neural network with many layers that can learn very complex patterns. Think of it as a stack of filters, each one understanding something more sophisticated than the last.",
    category: "Machine Learning"
  },
  {
    id: 171,
    term: "Early Stopping",
    definition: "Knowing when to stop training an AI before it gets too obsessed with the training data. Like knowing when to stop studying before you overthink and confuse yourself.",
    category: "Machine Learning"
  },
  {
    id: 172,
    term: "Elastic Net",
    definition: "A machine learning technique that finds the sweet spot between being too simple and too complex. Like tuning a guitar - not too loose, not too tight, just right.",
    category: "Machine Learning"
  },
  {
    id: 173,
    term: "Empirical Risk Minimization",
    definition: "A fancy way of saying 'learn from your mistakes on practice problems.' The AI tries to minimize errors on training data to perform better on new, unseen data.",
    category: "Machine Learning"
  },
  {
    id: 174,
    term: "Feedforward Neural Network",
    definition: "The simplest type of neural network where information flows in one direction from input to output. Like an assembly line where each worker passes the product to the next station.",
    category: "Machine Learning"
  },
  {
    id: 175,
    term: "Feature Extraction",
    definition: "Finding the most important characteristics in raw data that help make predictions. Like a detective picking out the key clues from a crime scene while ignoring irrelevant details.",
    category: "Machine Learning"
  },
  {
    id: 176,
    term: "Finite-State Machine",
    definition: "A simple model that can be in one state at a time and changes states based on inputs. Like a traffic light that goes from red to green to yellow based on timing.",
    category: "Machine Learning"
  },
  {
    id: 177,
    term: "Hidden Markov Model",
    definition: "A system that models sequences where you can see the results but not the hidden causes. Like inferring someone's mood from their text messages without seeing their face.",
    category: "Machine Learning"
  },
  {
    id: 178,
    term: "Hierarchical Clustering",
    definition: "Organizing data into groups within groups, like a family tree. It creates a hierarchy from individual items up to one big cluster, showing relationships at different levels.",
    category: "Machine Learning"
  },
  {
    id: 179,
    term: "Incremental Learning",
    definition: "Learning new information bit by bit without forgetting previous knowledge. Like adding new contacts to your phone without losing the old ones.",
    category: "Machine Learning"
  },
  {
    id: 180,
    term: "Imitation Learning",
    definition: "AI that learns by watching and copying experts, just like how kids learn by mimicking adults. Show it enough examples of good behavior and it learns to do the same.",
    category: "Machine Learning"
  },
  {
    id: 181,
    term: "Inductive Logic Programming",
    definition: "Teaching AI to learn logical rules from examples. Like showing a child many situations and letting them figure out the underlying principles, such as 'fire is hot' or 'heavy things fall.'",
    category: "Machine Learning"
  },
  {
    id: 182,
    term: "Inductive Reasoning",
    definition: "Making general rules from specific examples. Like noticing that every swan you've seen is white and concluding that all swans are white (which isn't actually true, but that's the idea).",
    category: "Machine Learning"
  },
  {
    id: 183,
    term: "Instance-Based Learning",
    definition: "Learning by remembering specific examples and comparing new situations to those memories. Like recognizing a new dog breed because it looks similar to dogs you've seen before.",
    category: "Machine Learning"
  },
  {
    id: 184,
    term: "Joint Probability Distribution",
    definition: "A mathematical way to describe how likely different combinations of events are. Like knowing the odds of it being both rainy AND cold versus sunny AND warm.",
    category: "Machine Learning"
  },
  {
    id: 185,
    term: "Latent Variable",
    definition: "Hidden factors that influence what you can observe but can't measure directly. Like 'happiness' - you can't measure it directly, but you see its effects in behavior and choices.",
    category: "Machine Learning"
  },
  {
    id: 186,
    term: "Learning to Rank",
    definition: "Teaching AI to put things in order of importance or relevance, like how search engines decide which results to show first when you Google something.",
    category: "Machine Learning"
  },
  {
    id: 187,
    term: "Multilayer Perceptron",
    definition: "A neural network with multiple layers of simple processing units. Think of it as a committee where each layer votes on the decision, building up from simple to complex understanding.",
    category: "Machine Learning"
  },
  {
    id: 188,
    term: "Meta-Learning",
    definition: "Learning how to learn faster. It's like developing study skills that help you master new subjects quickly, rather than starting from scratch every time.",
    category: "Machine Learning"
  },
  {
    id: 189,
    term: "Model Selection",
    definition: "Choosing the best AI approach for your specific problem. Like picking the right tool from a toolbox - sometimes you need a hammer, sometimes a screwdriver.",
    category: "Machine Learning"
  },
  {
    id: 190,
    term: "Multimodal Learning",
    definition: "AI that learns from different types of data at once - text, images, sound, etc. Like how humans understand a movie by seeing, hearing, and reading subtitles all together.",
    category: "Machine Learning"
  },
  {
    id: 191,
    term: "Neuro-Fuzzy System",
    definition: "Combining neural networks with fuzzy logic to handle both learning and uncertainty. Like having a brain that can both learn from experience and deal with unclear situations.",
    category: "Machine Learning"
  },
  {
    id: 192,
    term: "Neuroevolution",
    definition: "Using evolutionary principles to design neural networks. Instead of traditional training, it breeds networks like you'd breed dogs, keeping the best performers for the next generation.",
    category: "Machine Learning"
  },
  {
    id: 193,
    term: "Noise",
    definition: "Random errors or irrelevant information in data that can confuse AI learning. Like static on a radio - it makes it harder to hear the actual music clearly.",
    category: "Machine Learning"
  },
  {
    id: 194,
    term: "Probabilistic Reasoning",
    definition: "Making decisions based on likelihood rather than certainty. Like carrying an umbrella when there's a 70% chance of rain - you're planning for probable outcomes.",
    category: "Machine Learning"
  },
  {
    id: 195,
    term: "Q-Learning",
    definition: "A way for AI to learn the best actions through trial and error, like a video game character learning which moves lead to high scores and which ones get you killed.",
    category: "Machine Learning"
  },
  {
    id: 196,
    term: "Quantization",
    definition: "Making AI models smaller and faster by using simpler numbers. Like converting a high-definition movie to lower quality so it fits on your phone without losing the story.",
    category: "Machine Learning"
  },
  {
    id: 197,
    term: "Representation Learning",
    definition: "Teaching AI to automatically discover useful ways to represent data. Like learning that images can be described by edges and textures rather than just individual pixels.",
    category: "Machine Learning"
  },
  {
    id: 198,
    term: "Reservoir Computing",
    definition: "A neural network approach that uses a fixed 'reservoir' of connections and only trains the output layer. Like having a complex echo chamber that you learn to interpret.",
    category: "Machine Learning"
  },
  {
    id: 199,
    term: "Self-Supervised Learning",
    definition: "AI that creates its own training labels from the data itself. Like learning language by predicting missing words in sentences - no human labeling required.",
    category: "Machine Learning"
  },
  {
    id: 200,
    term: "Semi-Supervised Learning",
    definition: "Learning from a mix of labeled and unlabeled data. Like having some homework problems with answer keys and others where you have to figure it out yourself.",
    category: "Machine Learning"
  },
  {
    id: 201,
    term: "Sequence Learning",
    definition: "Teaching AI to understand patterns in ordered data like sentences, music, or time series. Like learning that 'once upon a time' usually starts a fairy tale.",
    category: "Machine Learning"
  },
  {
    id: 202,
    term: "Statistical Learning",
    definition: "Using statistical methods to find patterns in data and make predictions. Like using past weather data to forecast tomorrow's temperature - it's all about finding statistical relationships.",
    category: "Machine Learning"
  },
  {
    id: 203,
    term: "Statistical Model",
    definition: "A mathematical representation of real-world processes based on statistical relationships. Like a formula that predicts house prices based on size, location, and age.",
    category: "Machine Learning"
  },
  {
    id: 204,
    term: "Stochastic Process",
    definition: "A system that evolves randomly over time according to probabilistic rules. Like the weather - it follows patterns but has random elements that make exact prediction impossible.",
    category: "Machine Learning"
  },
  {
    id: 205,
    term: "Structured Prediction",
    definition: "Predicting complex, interconnected outputs rather than simple labels. Like predicting entire sentences with grammar and meaning, not just individual words.",
    category: "Machine Learning"
  },
  {
    id: 206,
    term: "Value Function",
    definition: "A way to score how good a particular state or action is for achieving goals. Like having a mental scoreboard that tells you how well you're doing toward your objectives.",
    category: "Machine Learning"
  },
  {
    id: 207,
    term: "Weight",
    definition: "The strength of connections between neurons in a neural network. Like the volume knobs that determine how much influence each input has on the final decision.",
    category: "Machine Learning"
  },
  {
    id: 208,
    term: "Zero-Shot Learning",
    definition: "AI that can recognize things it has never seen before by understanding descriptions or relationships. Like identifying a zebra as 'a horse with stripes' without ever seeing one.",
    category: "Machine Learning"
  },

  // Additional Natural Language Processing Terms
  {
    id: 209,
    term: "Corpus",
    definition: "A large collection of texts used to train language AI. Think of it as the AI's reading library - the more books it reads, the better it understands how language works.",
    category: "Natural Language Processing"
  },
  {
    id: 210,
    term: "Information Retrieval",
    definition: "Finding relevant information from large collections of text. It's what search engines do - sifting through billions of web pages to find exactly what you're looking for.",
    category: "Natural Language Processing"
  },
  {
    id: 211,
    term: "Lexical Semantics",
    definition: "Understanding what individual words mean and how they relate to each other. Like knowing that 'cat' and 'feline' refer to the same type of animal, just with different words.",
    category: "Natural Language Processing"
  },
  {
    id: 212,
    term: "Natural Language Generation",
    definition: "AI that can write human-like text, from simple sentences to entire articles. It's like having a robot writer that can express ideas in natural, flowing language.",
    category: "Natural Language Processing"
  },
  {
    id: 213,
    term: "Natural Language Understanding",
    definition: "AI that truly comprehends what text means, not just recognizes words. Like the difference between parroting back a phrase and actually understanding what someone is telling you.",
    category: "Natural Language Processing"
  },
  {
    id: 214,
    term: "Query Expansion",
    definition: "Making search queries better by adding related terms. Like turning 'car problems' into 'car problems automotive issues vehicle trouble' to find more relevant results.",
    category: "Natural Language Processing"
  },
  {
    id: 215,
    term: "Semantic Analysis",
    definition: "Understanding the actual meaning behind words and sentences, including context and nuance. Like knowing that 'This pizza is sick!' probably means it's really good, not diseased.",
    category: "Natural Language Processing"
  },
  {
    id: 216,
    term: "Semantic Network",
    definition: "A map of how concepts and meanings connect to each other. Like a mind map showing that 'dog' connects to 'pet,' 'animal,' 'loyal,' and 'furry' in a web of relationships.",
    category: "Natural Language Processing"
  },
  {
    id: 217,
    term: "Speech Recognition",
    definition: "Converting spoken words into text that computers can understand. It's what lets you talk to Siri or dictate messages instead of typing them out.",
    category: "Natural Language Processing"
  },
  {
    id: 218,
    term: "Text Mining",
    definition: "Digging through large amounts of text to find patterns, trends, and insights. Like being a detective who reads thousands of documents to solve a case.",
    category: "Natural Language Processing"
  },
  {
    id: 219,
    term: "Text-to-Speech",
    definition: "Converting written text into spoken words. It's how audiobooks are made by computers, or how your GPS gives you turn-by-turn directions out loud.",
    category: "Natural Language Processing"
  },
  {
    id: 220,
    term: "Virtual Assistant",
    definition: "AI helpers like Siri, Alexa, or Google Assistant that can understand speech, answer questions, and help with tasks through natural conversation.",
    category: "Natural Language Processing"
  },
  {
    id: 221,
    term: "Voice Recognition",
    definition: "Identifying who is speaking based on the unique characteristics of their voice. Like how you can recognize your friend calling even before they say their name.",
    category: "Natural Language Processing"
  },
  {
    id: 222,
    term: "Word Sense Disambiguation",
    definition: "Figuring out which meaning of a word is intended when it has multiple meanings. Like knowing whether 'bank' means money or river depending on the context.",
    category: "Natural Language Processing"
  },
  {
    id: 223,
    term: "Chatbot",
    definition: "AI programs designed to have conversations with humans through text or voice. From simple customer service bots to sophisticated companions like ChatGPT.",
    category: "Natural Language Processing"
  },
  {
    id: 224,
    term: "Personalized Recommendation",
    definition: "AI that suggests things you might like based on your past behavior and preferences. Like how Netflix knows what movies to suggest or Spotify creates playlists just for you.",
    category: "Natural Language Processing"
  },
  {
    id: 225,
    term: "Retrieval",
    definition: "Finding and extracting relevant information from large databases or document collections. Like having a super-smart librarian who can instantly find any fact you need.",
    category: "Natural Language Processing"
  },

  // Additional Computer Vision Terms
  {
    id: 226,
    term: "Machine Perception",
    definition: "Giving machines the ability to understand and interpret sensory information like sight, sound, and touch. It's like building artificial senses for computers.",
    category: "Computer Vision"
  },
  {
    id: 227,
    term: "Pattern Recognition",
    definition: "The ability to identify recurring themes, shapes, or structures in data. Like recognizing that a circle is still a circle whether it's big, small, red, or blue.",
    category: "Computer Vision"
  },

  // Additional Generative AI Terms
  {
    id: 232,
    term: "Program Synthesis",
    definition: "AI that can write computer code automatically based on descriptions of what the program should do. Like having a programmer assistant that turns your ideas into working software.",
    category: "Generative AI"
  },
  {
    id: 233,
    term: "Creativity",
    definition: "The ability of AI to generate novel and valuable ideas, art, or solutions. It's what lets AI compose music, write poetry, or design new products that feel genuinely creative.",
    category: "Generative AI"
  },

  // Additional Robotics Terms  
  {
    id: 234,
    term: "Behavior-Based Robotics",
    definition: "Building robots that act based on simple behaviors that combine to create complex actions. Like how flocking birds follow simple rules but create beautiful, coordinated movement patterns.",
    category: "Robotics"
  },
  {
    id: 235,
    term: "Digital Twin",
    definition: "A virtual copy of a real-world object or system that mirrors its behavior in real-time. Like having a detailed computer simulation of a factory that updates as the real factory operates.",
    category: "Robotics"
  },

  // Additional AI Ethics Terms
  {
    id: 236,
    term: "Ethics of AI",
    definition: "The moral principles and guidelines that should govern how AI is developed and used. It's about ensuring AI benefits humanity while avoiding harm and respecting human values.",
    category: "AI Ethics"
  },
  {
    id: 237,
    term: "Affective Computing",
    definition: "Building AI that can recognize, understand, and respond to human emotions. Like giving computers emotional intelligence so they can tell when you're frustrated or happy.",
    category: "AI Ethics"
  },
  {
    id: 238,
    term: "Uncertainty",
    definition: "Dealing with situations where AI doesn't have complete information or can't be 100% confident in its decisions. Like making the best choice possible when you don't have all the facts.",
    category: "AI Ethics"
  },
  {
    id: 239,
    term: "Utility Function",
    definition: "A mathematical way to define what outcomes an AI should value and optimize for. Like programming a robot's sense of right and wrong, or what it should consider 'good' results.",
    category: "AI Ethics"
  },

  // Additional Technical Concepts Terms
  {
    id: 240,
    term: "A* Search",
    definition: "A smart pathfinding algorithm that finds the shortest route by making educated guesses about which direction looks most promising. Like GPS navigation that considers both distance and traffic.",
    category: "Technical Concepts"
  },
  {
    id: 241,
    term: "Abstraction",
    definition: "Simplifying complex systems by focusing on the essential features and hiding unnecessary details. Like using a map instead of satellite photos - you get the important info without clutter.",
    category: "Technical Concepts"
  },
  {
    id: 242,
    term: "Abstract Data Type",
    definition: "A way to organize data that focuses on what you can do with it rather than how it's stored. Like knowing you can withdraw money from an ATM without understanding the bank's computer systems.",
    category: "Technical Concepts"
  },
  {
    id: 243,
    term: "Accelerating Change",
    definition: "The idea that technological progress is speeding up exponentially. Like how smartphones went from basic to incredibly sophisticated in just a few years - change keeps getting faster.",
    category: "Technical Concepts"
  },
  {
    id: 244,
    term: "Action Language",
    definition: "A formal way to describe actions and their effects in AI planning. Like writing a recipe that precisely defines what each step does and what ingredients it changes.",
    category: "Technical Concepts"
  },
  {
    id: 245,
    term: "Action Model Learning",
    definition: "Teaching AI to understand what happens when certain actions are taken. Like learning that turning a key starts a car or that watering plants helps them grow.",
    category: "Technical Concepts"
  },
  {
    id: 246,
    term: "Action Selection",
    definition: "How AI decides what to do next when faced with multiple options. Like choosing whether to grab an umbrella based on weather, time, and destination.",
    category: "Technical Concepts"
  },
  {
    id: 247,
    term: "Admissible Heuristic",
    definition: "A rule of thumb that never overestimates the cost to reach a goal. Like estimating travel time - it's better to guess low and arrive early than high and be stressed.",
    category: "Technical Concepts"
  },
  {
    id: 248,
    term: "Algorithmic Efficiency",
    definition: "How fast and resource-friendly an algorithm is. Like comparing two recipes - one might taste the same but use fewer ingredients and take less time to cook.",
    category: "Technical Concepts"
  },
  {
    id: 249,
    term: "Algorithmic Probability",
    definition: "Using the simplicity of algorithms to estimate how likely different explanations are. Simpler explanations (shorter programs) are considered more probable than complex ones.",
    category: "Technical Concepts"
  },
  {
    id: 250,
    term: "Analysis of Algorithms",
    definition: "Studying how fast algorithms run and how much memory they use. Like analyzing cars to see which ones are most fuel-efficient and can carry the most passengers.",
    category: "Technical Concepts"
  },
  {
    id: 251,
    term: "Analytics",
    definition: "The systematic analysis of data to discover meaningful patterns and insights. Like being a detective who solves mysteries by carefully examining clues and evidence.",
    category: "Technical Concepts"
  },
  {
    id: 252,
    term: "Answer Set Programming",
    definition: "A logic programming approach where you describe what you want rather than how to compute it. Like telling a computer 'I want a schedule with no conflicts' and letting it figure out the details.",
    category: "Technical Concepts"
  },
  {
    id: 253,
    term: "Anytime Algorithm",
    definition: "An algorithm that can give you a decent answer quickly, then improve it if given more time. Like sketch artists who can give you a rough drawing fast, then add details if needed.",
    category: "Technical Concepts"
  },
  {
    id: 254,
    term: "Application Programming Interface",
    definition: "A set of rules that lets different software programs talk to each other. Like a universal translator that helps apps share information and work together smoothly.",
    category: "Technical Concepts"
  },
  {
    id: 255,
    term: "Approximate String Matching",
    definition: "Finding text that's similar but not exactly the same. Like autocorrect figuring out you meant 'hello' when you typed 'helo' - close enough counts.",
    category: "Technical Concepts"
  },
  {
    id: 256,
    term: "Approximation Error",
    definition: "The difference between the real answer and your best guess. Like the gap between your estimated arrival time and when you actually show up.",
    category: "Technical Concepts"
  },
  {
    id: 257,
    term: "Argumentation Framework",
    definition: "A formal way to model debates and reasoning with conflicting information. Like having rules for a debate club that help determine which arguments are strongest.",
    category: "Technical Concepts"
  },
  {
    id: 258,
    term: "Ant Colony Optimization",
    definition: "Solving problems by mimicking how ants find the shortest path to food. Virtual ants leave digital 'pheromone trails' that guide others to good solutions.",
    category: "Technical Concepts"
  },
  {
    id: 259,
    term: "Boolean Logic",
    definition: "Simple true/false reasoning that computers use for making decisions. Everything boils down to yes/no, on/off, 1/0 - the basic language of digital thinking.",
    category: "Technical Concepts"
  },
  {
    id: 260,
    term: "Epistemic Logic",
    definition: "Logic that deals with knowledge and belief - what someone knows, believes, or is uncertain about. Like reasoning about what others know or don't know.",
    category: "Technical Concepts"
  },
  {
    id: 261,
    term: "Non-Monotonic Logic",
    definition: "Reasoning that can change conclusions when new information arrives. Unlike regular logic where facts stay true, this handles situations where you might need to revise beliefs.",
    category: "Technical Concepts"
  },
  {
    id: 262,
    term: "Big Data",
    definition: "Datasets so large and complex that traditional data tools can't handle them effectively. Think of it as information so massive you need special equipment just to store and analyze it.",
    category: "Technical Concepts"
  },
  {
    id: 263,
    term: "Blackboard System",
    definition: "A problem-solving approach where different AI experts contribute knowledge to a shared workspace. Like having multiple specialists work on different parts of a puzzle on the same whiteboard.",
    category: "Technical Concepts"
  },
  {
    id: 264,
    term: "Bounded Rationality",
    definition: "Making the best decisions possible given limited time, information, and thinking power. Like choosing a restaurant when you're hungry - you pick something good enough rather than researching every option.",
    category: "Technical Concepts"
  },
  {
    id: 265,
    term: "Brute-Force Search",
    definition: "Solving problems by trying every possible solution until you find the right one. Like finding your keys by checking every pocket, drawer, and surface in your house.",
    category: "Technical Concepts"
  },
  {
    id: 266,
    term: "Business Intelligence",
    definition: "Using data analysis to help companies make smarter business decisions. Like having a crystal ball that uses past sales data to predict what customers will want next month.",
    category: "Technical Concepts"
  },
  {
    id: 267,
    term: "Case-Based Reasoning",
    definition: "Solving new problems by remembering how you solved similar problems before. Like a doctor diagnosing a patient by recalling similar cases from medical school or experience.",
    category: "Technical Concepts"
  },
  {
    id: 268,
    term: "Causal Reasoning",
    definition: "Understanding cause-and-effect relationships - what makes things happen and why. Like knowing that rain causes wet streets, not the other way around.",
    category: "Technical Concepts"
  },
  {
    id: 269,
    term: "Classifier",
    definition: "AI that puts things into categories. Like a sorting machine that can look at emails and decide which folder they belong in - spam, work, personal, etc.",
    category: "Technical Concepts"
  },
  {
    id: 270,
    term: "Clustering",
    definition: "Grouping similar things together without knowing the categories beforehand. Like organizing your music collection by letting the computer figure out genres based on sound similarities.",
    category: "Technical Concepts"
  },
  {
    id: 271,
    term: "Complexity Class",
    definition: "Categories that group problems by how hard they are to solve computationally. Like organizing puzzles by difficulty level - some are quick to solve, others might take centuries.",
    category: "Technical Concepts"
  },
  {
    id: 272,
    term: "Constraint Satisfaction Problem",
    definition: "Finding solutions that meet all the given requirements and limitations. Like scheduling a meeting where everyone is available, the room is free, and equipment is working.",
    category: "Technical Concepts"
  },
  {
    id: 273,
    term: "Context-Aware Computing",
    definition: "Technology that adapts based on the current situation - time, location, user preferences, and circumstances. Like a smart thermostat that adjusts based on weather and occupancy.",
    category: "Technical Concepts"
  },
  {
    id: 274,
    term: "Crowdsourcing",
    definition: "Getting help from lots of people to solve problems or gather information. Like Wikipedia, where thousands of volunteers contribute knowledge to create a massive encyclopedia.",
    category: "Technical Concepts"
  },
  {
    id: 275,
    term: "Cybernetics",
    definition: "The study of how systems regulate and control themselves through feedback loops. Like how a thermostat maintains temperature by monitoring and adjusting automatically.",
    category: "Technical Concepts"
  },
  {
    id: 276,
    term: "Data Mining",
    definition: "Digging through large amounts of data to find hidden patterns and valuable insights. Like panning for gold, but instead of gold nuggets, you're finding useful information.",
    category: "Technical Concepts"
  },
  {
    id: 277,
    term: "Data Science",
    definition: "The art and science of extracting insights from data using statistics, programming, and domain expertise. Like being a detective who solves mysteries using numbers instead of fingerprints.",
    category: "Technical Concepts"
  },
  {
    id: 278,
    term: "Data Set",
    definition: "A collection of related data organized for analysis. Like a photo album, but instead of pictures, it contains information that can be studied and analyzed.",
    category: "Technical Concepts"
  },
  {
    id: 279,
    term: "Decision Support System",
    definition: "Computer systems that help people make better decisions by organizing information and analyzing options. Like having a smart advisor that presents facts and recommendations.",
    category: "Technical Concepts"
  },
  {
    id: 280,
    term: "Deterministic Algorithm",
    definition: "An algorithm that always produces the same output for the same input. Like a recipe that always makes the same cake if you follow it exactly with the same ingredients.",
    category: "Technical Concepts"
  },
  {
    id: 281,
    term: "Distributed AI",
    definition: "Spreading AI computation across multiple computers or locations. Like having a team of experts in different cities all working together on the same project.",
    category: "Technical Concepts"
  },
  {
    id: 282,
    term: "Domain Adaptation",
    definition: "Helping AI trained in one area work well in a related but different area. Like a doctor trained in human medicine learning to treat animals - similar skills, different application.",
    category: "Technical Concepts"
  },
  {
    id: 283,
    term: "Dynamic Programming",
    definition: "Solving complex problems by breaking them into simpler subproblems and reusing solutions. Like building with LEGO - you solve small parts once and reuse them in bigger constructions.",
    category: "Technical Concepts"
  },
  {
    id: 284,
    term: "Edge Computing",
    definition: "Processing data close to where it's created instead of sending it to distant servers. Like having a local grocery store instead of driving to the city for every purchase.",
    category: "Technical Concepts"
  },
  {
    id: 285,
    term: "Entropy",
    definition: "A measure of uncertainty or randomness in information. High entropy means lots of surprises and unpredictability, low entropy means things are orderly and predictable.",
    category: "Technical Concepts"
  },
  {
    id: 286,
    term: "Evolutionary Algorithm",
    definition: "Problem-solving inspired by natural evolution - create random solutions, keep the best ones, mix them together, and repeat until you get something great.",
    category: "Technical Concepts"
  },
  {
    id: 287,
    term: "Fuzzy Logic",
    definition: "Logic that handles partial truths and degrees of uncertainty. Instead of just true/false, it allows for 'somewhat true' or 'mostly false' - more like human thinking.",
    category: "Technical Concepts"
  },
  {
    id: 288,
    term: "Fuzzy Set",
    definition: "A set where membership can be partial rather than all-or-nothing. Like the set of 'tall people' where someone can be somewhat tall, very tall, or just a little tall.",
    category: "Technical Concepts"
  },
  {
    id: 289,
    term: "Game Theory",
    definition: "Mathematical study of strategic decision-making when multiple players with different goals interact. Like analyzing poker strategies or business competition.",
    category: "Technical Concepts"
  },
  {
    id: 290,
    term: "Genetic Algorithm",
    definition: "Optimization inspired by biological evolution - start with random solutions, breed the best ones together, add mutations, and evolve toward better answers over generations.",
    category: "Technical Concepts"
  },
  {
    id: 291,
    term: "Graph Theory",
    definition: "The mathematics of networks and connections. Like studying subway maps to understand the best routes, or social networks to see how people are connected.",
    category: "Technical Concepts"
  },
  {
    id: 292,
    term: "Heuristic",
    definition: "A rule of thumb or educated guess that usually works well but isn't guaranteed to be perfect. Like 'always take the highway during rush hour' - often good advice but not always.",
    category: "Technical Concepts"
  },
  {
    id: 293,
    term: "High-Dimensional Space",
    definition: "Data that has many different attributes or features. Like describing a person with hundreds of characteristics instead of just height and weight - harder to visualize but more detailed.",
    category: "Technical Concepts"
  },
  {
    id: 294,
    term: "Hybrid Intelligent System",
    definition: "AI that combines different approaches to get better results. Like a Swiss Army knife that has multiple tools working together instead of just one.",
    category: "Technical Concepts"
  },
  {
    id: 295,
    term: "Hypothesis",
    definition: "An educated guess or proposed explanation that can be tested. Like theorizing that students learn better with music, then designing experiments to see if it's true.",
    category: "Technical Concepts"
  },
  {
    id: 296,
    term: "Inference",
    definition: "Drawing logical conclusions from available information. Like seeing footprints in snow and inferring that someone walked there recently.",
    category: "Technical Concepts"
  },
  {
    id: 297,
    term: "Knowledge Graph",
    definition: "A network that connects facts and concepts, showing how different pieces of information relate to each other. Like a family tree, but for all knowledge.",
    category: "Technical Concepts"
  },
  {
    id: 298,
    term: "Markov Model",
    definition: "A system where the next state depends only on the current state, not the entire history. Like predicting tomorrow's weather based only on today's conditions.",
    category: "Technical Concepts"
  },
  {
    id: 299,
    term: "Markov Decision Process",
    definition: "A framework for making sequential decisions where outcomes are partly random. Like planning a road trip where traffic and weather affect your choices along the way.",
    category: "Technical Concepts"
  },
  {
    id: 300,
    term: "Monte Carlo Method",
    definition: "Using random sampling to solve problems that are hard to calculate exactly. Like estimating the area of an irregular shape by randomly throwing darts and seeing where they land.",
    category: "Technical Concepts"
  },
  {
    id: 301,
    term: "Model",
    definition: "A simplified representation of something complex that helps us understand or predict behavior. Like a toy airplane that helps explain how real planes fly.",
    category: "Technical Concepts"
  },
  {
    id: 302,
    term: "Multi-Agent System",
    definition: "Multiple AI agents working together or competing to achieve goals. Like a team of robots where each has different skills but they coordinate to complete tasks.",
    category: "Technical Concepts"
  },
  {
    id: 303,
    term: "Ontology",
    definition: "A formal way to organize and categorize knowledge about a particular domain. Like a detailed taxonomy that defines what things exist and how they relate to each other.",
    category: "Technical Concepts"
  },
  {
    id: 304,
    term: "Optimization",
    definition: "Finding the best solution among all possible options. Like finding the fastest route to work considering traffic, distance, and road conditions.",
    category: "Technical Concepts"
  },
  {
    id: 305,
    term: "Parallel Computing",
    definition: "Using multiple processors to work on different parts of a problem simultaneously. Like having several people work on different pieces of a jigsaw puzzle at the same time.",
    category: "Technical Concepts"
  },
  {
    id: 306,
    term: "Parameter",
    definition: "A variable that defines the characteristics or behavior of a system. Like the settings on your camera - aperture, shutter speed, and ISO are parameters that affect the photo.",
    category: "Technical Concepts"
  },
  {
    id: 307,
    term: "Qualitative Reasoning",
    definition: "Understanding systems using qualities and relationships rather than precise numbers. Like knowing that 'more pressure makes water boil faster' without needing exact measurements.",
    category: "Technical Concepts"
  },
  {
    id: 308,
    term: "Rule-Based System",
    definition: "AI that follows explicit 'if-then' rules to make decisions. Like a flowchart that tells you exactly what to do in each situation - if this happens, then do that.",
    category: "Technical Concepts"
  },
  {
    id: 309,
    term: "Satisfiability",
    definition: "Determining whether there's a way to make a logical formula true. Like solving a puzzle to see if there's any combination of moves that leads to victory.",
    category: "Technical Concepts"
  },
  {
    id: 310,
    term: "Scalability",
    definition: "How well a system handles increasing amounts of work or users. Like a restaurant that can serve 10 customers well but falls apart when 100 people show up.",
    category: "Technical Concepts"
  },
  {
    id: 311,
    term: "Search Algorithm",
    definition: "Systematic methods for finding solutions in a space of possibilities. Like different strategies for finding your keys - random search, systematic room-by-room, or retracing your steps.",
    category: "Technical Concepts"
  },
  {
    id: 312,
    term: "Simulated Annealing",
    definition: "An optimization technique inspired by metallurgy that occasionally accepts worse solutions to avoid getting stuck. Like sometimes taking a longer route to avoid traffic jams.",
    category: "Technical Concepts"
  },
  {
    id: 313,
    term: "Soft Computing",
    definition: "Computing approaches that tolerate imprecision and uncertainty to achieve tractability and low cost. Like being okay with 'close enough' answers when perfect precision isn't necessary.",
    category: "Technical Concepts"
  },
  {
    id: 314,
    term: "Software Agent",
    definition: "A computer program that acts autonomously on behalf of users or other programs. Like having a digital assistant that can book flights, answer emails, or manage your calendar.",
    category: "Technical Concepts"
  },
  {
    id: 315,
    term: "Task",
    definition: "A specific job or activity that needs to be completed. In AI, it's what we want the system to accomplish - like recognizing images, translating text, or playing games.",
    category: "Technical Concepts"
  },
  {
    id: 316,
    term: "Temporal Logic",
    definition: "Logic that includes concepts of time - what was true before, what's true now, and what will be true later. Like reasoning about sequences of events and their timing.",
    category: "Technical Concepts"
  },
  {
    id: 317,
    term: "Threshold Function",
    definition: "A function that activates when input exceeds a certain level. Like a fire alarm that only goes off when smoke reaches a critical concentration.",
    category: "Technical Concepts"
  },
  {
    id: 318,
    term: "Validation",
    definition: "Testing whether a model or system actually works correctly on new, unseen data. Like checking if your study methods actually help you perform better on real exams.",
    category: "Technical Concepts"
  },
  {
    id: 319,
    term: "Variable",
    definition: "A symbol that represents a value that can change. Like using 'x' in math to represent any number, or 'temperature' to represent whatever the current temperature is.",
    category: "Technical Concepts"
  },
  {
    id: 320,
    term: "Vector Space Model",
    definition: "A way to represent text or data as points in multi-dimensional space. Like plotting documents on a graph where similar documents end up close to each other.",
    category: "Technical Concepts"
  },
  {
    id: 321,
    term: "Narrow AI",
    definition: "AI built for a specific job. Example: a model that only translates languages or only spots spam emails.",
    category: "Foundational AI"
  },
  {
    id: 322,
    term: "Language Model",
    definition: "Any model that learns patterns in text and guesses the next word. Example: your phone suggesting the next word as you type.",
    category: "Natural Language Processing"
  },
  {
    id: 323,
    term: "Attention Mechanism",
    definition: "The part that lets the model look harder at the key bits. Example: when translating a sentence, it focuses on the word being translated right now.",
    category: "Natural Language Processing"
  },
  {
    id: 324,
    term: "CLIP",
    definition: "Connects pictures and text in the same \"meaning space.\" Example: find the photo that best matches the caption \"a red bicycle.\"",
    category: "Computer Vision"
  },
  {
    id: 325,
    term: "ControlNet",
    definition: "A helper that lets you guide image generation with edges, poses, or depth. Example: upload a stick‑figure pose; get a person in that pose.",
    category: "Generative AI"
  },
  {
    id: 326,
    term: "LoRA",
    definition: "A way to fine‑tune big models by adding tiny adapters instead of changing the whole model. Example: teaching a model your brand's tone in minutes.",
    category: "Machine Learning"
  },
  {
    id: 327,
    term: "Robot Operating System (ROS)",
    definition: "Not an operating system—more like glue. Tools that help robot parts talk to each other. Example: a robot vacuum's sensors and motors coordinate using ROS tools.",
    category: "Robotics"
  },
  {
    id: 328,
    term: "Retrieval‑Augmented Generation (RAG)",
    definition: "An AI that looks things up while answering. Example: it pulls relevant pages from your knowledge base and cites them.",
    category: "Natural Language Processing"
  },
  {
    id: 329,
    term: "Grounded Generation",
    definition: "Replies are based on specific sources you provide—and can show their citations. Example: \"According to the 2024 policy PDF…\".",
    category: "Natural Language Processing"
  },
  {
    id: 330,
    term: "Hallucination (LLM)",
    definition: "When the AI makes something up that sounds true but isn't. Example: inventing a book title that doesn't exist.",
    category: "Natural Language Processing"
  },
  {
    id: 331,
    term: "Embedding",
    definition: "Turning text (or images) into a list of numbers that capture meaning, so we can compare similarity. Example: \"car\" is closer to \"truck\" than to \"banana.\"",
    category: "Technical Concepts"
  },
  {
    id: 332,
    term: "Vector Database",
    definition: "Stores embeddings and finds the most similar items fast. Example: \"Find 5 docs most similar to this question.\"",
    category: "Technical Concepts"
  },
  {
    id: 333,
    term: "Chunking",
    definition: "Splitting long docs into smaller pieces the AI can handle. Example: a 50‑page manual becomes many short passages for search and citation.",
    category: "Technical Concepts"
  },
  {
    id: 334,
    term: "Context Window",
    definition: "How much text the model can keep \"in mind\" at once. Example: it may remember ~20 pages of text for one answer, but not a whole book.",
    category: "Technical Concepts"
  },
  {
    id: 335,
    term: "Temperature",
    definition: "A creativity dial. Higher = more surprising wording; lower = safer, more predictable wording. Example: temperature 0.2 sounds steady; 0.9 sounds creative.",
    category: "Technical Concepts"
  },
  {
    id: 336,
    term: "Top‑p (Nucleus) Sampling",
    definition: "Pick the next word from only the most likely few, keeping quality but allowing variety. Example: it chooses among the top 90% of probability mass.",
    category: "Technical Concepts"
  },
  {
    id: 337,
    term: "Function Calling / Tool Use",
    definition: "Let the AI call tools (calculator, calendar, API), then use the result in its answer. Example: \"What's 23.8% of 1,249?\" → it calls a calculator.",
    category: "Technical Concepts"
  },
  {
    id: 338,
    term: "Agent (AI Agent)",
    definition: "An AI \"helper\" that plans steps, uses tools, checks its work, then replies. Example: book travel: search flights → pick → draft email.",
    category: "Technical Concepts"
  },
  {
    id: 339,
    term: "Guardrails (Safety Filters)",
    definition: "Rules and filters that reduce harmful, biased, or off‑limits answers. Example: blocking hate speech or medical advice without disclaimers.",
    category: "AI Ethics"
  },
  {
    id: 340,
    term: "Prompt Injection",
    definition: "A trick to fool the AI into ignoring its rules or revealing secrets by hiding instructions in the input. Example: a pasted web page that says \"ignore all prior rules.\"",
    category: "AI Ethics"
  },
  {
    id: 341,
    term: "Data Poisoning",
    definition: "Sneaking bad examples into training/fine‑tuning so the AI learns the wrong thing. Example: adding fake reviews to teach \"5 stars\" = \"awful.\"",
    category: "AI Ethics"
  },
  {
    id: 342,
    term: "RLHF (Reinforcement Learning from Human Feedback)",
    definition: "People compare answers; the model learns to prefer the better one. Example: volunteers pick the clearer, kinder reply, shaping behavior.",
    category: "Machine Learning"
  },
  {
    id: 343,
    term: "DPO (Direct Preference Optimization)",
    definition: "A simpler way to train on people's choices without a separate reward model. Example: feed in \"A vs B\" choices; it learns to pick A when A is preferred.",
    category: "Machine Learning"
  },
  {
    id: 344,
    term: "Mixture of Experts (MoE)",
    definition: "A big model made of many \"experts,\" but it only wakes up a few per request—fast and efficient. Example: a math expert activates for equations; a writing expert for emails.",
    category: "Machine Learning"
  },
  {
    id: 345,
    term: "Multimodal Model",
    definition: "Handles more than one data type at once (text + images + audio…). Example: look at a photo and answer a written question about it.",
    category: "Machine Learning"
  },
  {
    id: 346,
    term: "Vision‑Language Model (VLM)",
    definition: "A multimodal model that connects images and text. Example: \"What's in this X‑ray?\" → \"A fracture on the left side.\"",
    category: "Computer Vision"
  },
  {
    id: 347,
    term: "Text‑to‑Video",
    definition: "Makes short videos from written prompts by predicting frames over time. Example: \"A paper plane flying through clouds at sunset.\"",
    category: "Generative AI"
  },
  {
    id: 348,
    term: "Watermarking (AI Content)",
    definition: "Hiding a special signal in AI‑made media so tools can detect it later. Example: checking if an image was AI‑generated.",
    category: "AI Ethics"
  },
  {
    id: 349,
    term: "On‑device AI / NPU",
    definition: "Running AI on your phone/laptop with special chips (NPUs) for speed and privacy. Example: live translation without sending audio to the cloud.",
    category: "Technical Concepts"
  },
  {
    id: 350,
    term: "GPU / TPU",
    definition: "Powerful chips that train and run AI fast. Example: GPUs in gaming PCs; TPUs in data centers.",
    category: "Technical Concepts"
  },
  {
    id: 351,
    term: "Model Card",
    definition: "A one‑page \"info sheet\" for a model: what it's for, how it was trained, limits and risks. Example: \"Not for medical diagnosis.\"",
    category: "AI Ethics"
  },
  {
    id: 352,
    term: "Datasheet for Datasets",
    definition: "A short report describing a dataset's source, contents, and possible biases. Example: \"News articles from 2018–2023, mainly English.\"",
    category: "AI Ethics"
  },
  {
    id: 353,
    term: "Federated Learning",
    definition: "Many devices train a shared model without sending raw data to a server. Example: phones improve a typing model locally and share only tiny updates.",
    category: "Machine Learning"
  },
  {
    id: 354,
    term: "Differential Privacy",
    definition: "Adds carefully chosen randomness so results show group trends, not details about any one person. Example: a city report shares average bus times without exposing your exact trips.",
    category: "AI Ethics"
  },
  {
    id: 355,
    term: "Precision",
    definition: "Of all the times the model said \"yes,\" how often was it right? Example: it flags 100 emails as spam, and 95 really are → precision 95%.",
    category: "Technical Concepts"
  },
  {
    id: 356,
    term: "Recall",
    definition: "Of all the real \"yes\" cases, how many did the model find? Example: there are 100 spam emails; it caught 80 → recall 80%.",
    category: "Technical Concepts"
  },
  {
    id: 357,
    term: "F1 Score",
    definition: "One number that balances precision and recall. Example: good when you want both \"right when it says yes\" and \"find most of the yeses.\"",
    category: "Technical Concepts"
  },
  {
    id: 358,
    term: "Accuracy",
    definition: "Out of all predictions, how many were correct. Example: 90 right out of 100 = 90% accuracy.",
    category: "Technical Concepts"
  },
  {
    id: 359,
    term: "Class Imbalance",
    definition: "When one class is rare and another is common. Example: 1 \"fraud\" vs 999 \"not fraud\"—accuracy alone can mislead.",
    category: "Technical Concepts"
  },
  {
    id: 360,
    term: "Confusion Matrix",
    definition: "A small table showing what the model got right and wrong per class. Rows = real answers; columns = model answers. Example: you can see it confuses \"cat\" with \"fox.\"",
    category: "Technical Concepts"
  },
  {
    id: 361,
    term: "ROC Curve & AUC",
    definition: "A chart showing the trade‑off between catching positives and raising false alarms as you move the threshold; AUC is one number summarizing the curve (closer to 1 is better). Example: compare models at many thresholds, not just one.",
    category: "Technical Concepts"
  },
  {
    id: 362,
    term: "Cross‑Validation",
    definition: "A fair test that trains and tests on different slices of the data, then averages results. Example: 5‑fold CV rotates which 20% is the test each time.",
    category: "Technical Concepts"
  },
  {
    id: 363,
    term: "Train / Validation / Test Split",
    definition: "Three sets: learn on train, tune settings on validation, judge final quality on test. Example: keep the test set untouched until the end.",
    category: "Technical Concepts"
  },
  {
    id: 364,
    term: "Regularization",
    definition: "Gentle rules (penalties) that stop a model from memorizing noise. Example: adding a small penalty for very large weights to improve generalization.",
    category: "Technical Concepts"
  },
  {
    id: 365,
    term: "Underfitting",
    definition: "The model is too simple and misses real patterns. Example: a straight line fitted to curvy data performs poorly.",
    category: "Technical Concepts"
  },
  {
    id: 366,
    term: "Connection (Weight)",
    definition: "The strength of the link between two neurons in a network; training tweaks these to learn. Example: turning the \"volume knob\" up or down on certain inputs.",
    category: "Technical Concepts"
  },
  {
    id: 367,
    term: "Perceptron",
    definition: "A simple classifier that draws a line between two groups. Example: separate \"spam\" vs \"not spam\" with a line in a 2‑D plot.",
    category: "Machine Learning"
  },
  {
    id: 368,
    term: "Tensor",
    definition: "A multi‑dimensional array of numbers. Example: a color image is a 3‑D tensor: height × width × RGB.",
    category: "Technical Concepts"
  },
  {
    id: 369,
    term: "Feature",
    definition: "A measurable input the model uses. Example: for house price: size, rooms, zip code.",
    category: "Technical Concepts"
  },
  {
    id: 370,
    term: "Processing Unit (Hardware)",
    definition: "Chips that do AI math quickly (e.g., GPU, TPU, NPU). Example: speeding up image generation from minutes to seconds.",
    category: "Technical Concepts"
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
