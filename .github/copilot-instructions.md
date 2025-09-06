# Copilot Instructions — AI Glossary Project

> Purpose: Build a **lightweight AI glossary** that makes complex AI terms instantly accessible. Target: 60-minute functional prototype with search + definitions.

## Project Context
- **Product**: Client-side AI glossary for developers, students, and tech enthusiasts
- **Core User Action**: Search AI terms → get instant, clear definitions
- **Tech Stack**: React + Vite, pure client-side, no backend/APIs
- **Success Metric**: User finds "neural network" definition in <10 seconds

## Target Audience Needs
- **Developers**: Quick AI term lookup without leaving workflow
- **Students**: Simple definitions with examples for faster learning
- **Tech Enthusiasts**: Browse trending AI concepts with confidence
- **Beginners**: Category-based exploration of AI topics

## Technical Guardrails
- **Static Data Only**: 20-30 curated AI terms in JavaScript arrays/objects
- **No External Dependencies**: Avoid heavy libraries; prefer vanilla React
- **Mobile-First**: Responsive design that works on all devices
- **Performance**: Real-time search results (<100ms response)
- **Accessibility**: Semantic HTML, keyboard navigation, screen reader friendly

## Data Structure (Use This Format)
```javascript
const aiTerms = [
  {
    id: "neural-network",
    term: "Neural Network",
    definition: "A computing system inspired by biological neural networks...",
    category: "Machine Learning",
    examples: ["Image recognition", "Natural language processing"]
  }
];
```

## Build Phases (60min total)

### Phase 1: Core MVP (30 min)
1. **Search Interface**: Prominent search box with real-time filtering
2. **Term Display**: Expandable cards showing term + definition
3. **Static Data**: Embed 20+ AI terms covering ML, NLP, Computer Vision, Ethics
4. **Basic Styling**: Clean, readable layout that works on mobile

### Phase 2: Enhanced Features (15 min)
1. **Category Filters**: ML, NLP, Computer Vision, Ethics buttons
2. **Search Highlighting**: Bold matching text in results
3. **Smooth Animations**: Expand/collapse transitions
4. **Better UX**: Empty states, loading indicators

### Phase 3: Polish (15 min)
1. **Dark Mode Toggle**: CSS class switching
2. **URL State**: Shareable links for specific terms
3. **Performance**: Optimize search algorithm
4. **Accessibility**: ARIA labels, focus management

## AI Terms to Include (Minimum Set)
**Machine Learning**: Neural Network, Deep Learning, Supervised Learning, Unsupervised Learning, Reinforcement Learning, Overfitting, Training Data
**NLP**: Transformer, BERT, GPT, Tokenization, Embedding, Large Language Model
**Computer Vision**: CNN, Object Detection, Image Classification, Segmentation
**Ethics/General**: AI Bias, Explainable AI, AGI, Model Drift, Hallucination

## Key Prompts for This Project
- "Create the search component with real-time filtering for AI terms"
- "Generate the AI terms data array with definitions and categories"
- "Build expandable term cards with smooth animations"
- "Add category filter buttons that update the search results"
- "Implement search highlighting that bolds matching text"
- "Create a responsive layout for the AI glossary interface"
- "Add dark mode toggle with CSS custom properties"
- "Optimize the search algorithm for better performance"

## Quality Standards
- **Functional**: Search works instantly, definitions are clear and accurate
- **Technical**: No console errors, builds successfully, minimal bundle size
- **UX**: Intuitive interface requiring no instructions
- **Content**: Definitions are beginner-friendly with practical examples

## Success Checklist
- [ ] User can search and find any included AI term
- [ ] Definitions expand/collapse smoothly
- [ ] Category filters work correctly
- [ ] Mobile responsive design
- [ ] No runtime errors
- [ ] Search highlights matching text
- [ ] Accessible keyboard navigation
