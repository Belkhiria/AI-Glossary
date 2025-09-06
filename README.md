# AI Glossary

Ever felt lost in AI conversations because of all the jargon? This glossary translates 135+ AI terms into plain English that actually makes sense.

## 🌐 Access the App

**Use it here:** https://belkhiria.github.io/AI-Glossary/

Start searching AI terms right away - no setup required.

## What's This About?

I got tired of AI explanations that sound like they were written by robots for robots. So I built this glossary where "Neural Network" doesn't sound like rocket science and "Transformer" actually explains what it does.

Think of it as your AI dictionary, but one that talks like a human.

## What You Get

- 135+ AI terms explained like you're talking to a friend
- Search through everything instantly 
- Filter by what you care about (ML, computer vision, etc.)
- Dark mode because your eyes matter
- Works on your phone, tablet, whatever
- Click any term to read the full explanation
- No servers, no signup, no BS

## Who This Is For

- Developers jumping into AI
- Students trying to decode their textbooks  
- Anyone curious about AI but allergic to academic speak
- People who want to sound smart in AI meetings

## The Categories

- **Foundational AI** - The basics everyone should know
- **Machine Learning** - Where computers learn stuff
- **Natural Language Processing** - Teaching computers to read
- **Computer Vision** - Teaching computers to see  
- **Generative AI** - The creative stuff like ChatGPT
- **Robotics** - When AI gets physical
- **AI Ethics** - Keeping AI from going rogue
- **Technical Concepts** - The nitty-gritty details

## How It Works

Built with React and Vite. No fancy frameworks, no over-engineering. Just a clean, fast web app that does what it says.

## Getting Started

You know the drill:

```bash
git clone https://github.com/Belkhiria/AI-Glossary.git
cd AI-Glossary
npm install
npm run dev
```

Open http://localhost:5173 and you're good to go.

Want to deploy it? Run `npm run build` and upload the `dist` folder anywhere.

## The Files

```
src/
├── data/aiTerms.js     # All the terms and definitions
├── App.jsx             # The main app
├── App.css             # Makes it look good
└── main.jsx           # Entry point
```

## My Definition Philosophy

No more "A neural network is a computational model inspired by biological neural networks." 

Instead: "A computer system loosely inspired by how our brain works - lots of simple processing units connected together that pass information around until they figure out the right answer."

See the difference? I use analogies, real examples, and language that doesn't require a PhD to understand.

## Want to Help?

Found a definition that sounds too academic? Fix it. Know an AI term that's missing? Add it. Want to improve the UI? Go for it.

Just keep the definitions conversational and jargon-free.

To add a term, edit `src/data/aiTerms.js`:

```javascript
{
  id: 136,
  term: "Some AI Thing",
  definition: "Explain it like you're telling a friend over coffee.",
  category: "Pick the right category"
}
```

## License

MIT License. Use it, modify it, share it.

---

Built because AI knowledge shouldn't be locked behind academic jargon.
