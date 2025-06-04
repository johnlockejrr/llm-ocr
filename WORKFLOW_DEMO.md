# LLM OCR Workflow Demo

This directory contains a Jupyter notebook (`workflow_demo.ipynb`) that demonstrates the complete LLM OCR package workflow with real API keys.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -e .[dev]
pip install jupyter
```

### 2. Set Up API Keys

**Option A: Environment Variables**
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key" 
export GEMINI_API_KEY="your-gemini-key"
export TOGETHER_API_KEY="your-together-key"
```

**Option B: .env File** (Recommended)
```bash
cp .env.template .env
# Edit .env file with your actual API keys
```

### 3. Prepare Test Data

Add your test documents to the `ground_truth/` directory:
```
ground_truth/
├── document1.xml      # ALTO XML file
├── document1.jpeg     # Corresponding image
├── document2.xml
├── document2.png
└── ...
```

Each document needs both an XML file (ALTO format) and an image file.

### 4. Run the Demo

```bash
jupyter notebook workflow_demo.ipynb
```

## 📋 What the Demo Tests

1. **Environment Setup** - Checks API keys and dependencies
2. **Package Import** - Verifies all modules load correctly  
3. **Basic Components** - Tests evaluator and model creation
4. **Text Correction** - Simple API call to test connectivity
5. **Full Workflow** - Complete OCR pipeline (optional, requires data)

## ⚠️ Important Notes

- **API Costs**: The notebook includes safety measures - the full workflow test is commented out by default
- **Rate Limits**: Start with simple tests before running full workflows
- **Test Data**: Use small test documents first to verify everything works
- **Error Handling**: The notebook includes comprehensive error checking

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the package root directory
2. **API Key Errors**: Verify your keys are set correctly and have sufficient credits
3. **Missing Data**: Ensure test documents are in the correct format (ALTO XML + images)
4. **Model Errors**: Different providers have different model names - check the model configs in the notebook

### Getting Help

- Check the notebook output for detailed error messages
- Verify API key permissions and quotas with your providers
- Ensure test data follows the expected ALTO XML format

## 📊 Expected Output

The notebook will show:
- ✅ API key status for each provider
- 🤖 Available models and successful connections  
- 📝 Text correction examples with before/after metrics
- 📊 Evaluation results and accuracy improvements
- 📁 Output files (if full workflow is run)

## 🎯 Next Steps

After running the demo successfully:
1. Try with your own OCR documents
2. Experiment with different models and providers
3. Adjust evaluation metrics for your use case
4. Integrate into your own workflows

---

**Ready to test your LLM OCR setup? Open `workflow_demo.ipynb` and get started!** 🚀