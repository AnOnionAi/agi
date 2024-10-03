Welcome to 'AGI' – our ambitious journey towards creating a scaled-down yet powerful AI model. Inspired by the sophisticated capabilities of GPT-like models, this project focuses on harnessing deep learning, specifically using PyTorch and Transformer architectures, to develop an AI adept in programming language comprehension and generation.

Our mission is to engineer an AI model that can not only understand and write code in multiple programming languages but also explain its reasoning and logic. The immediate scope of our project centers on Python, with plans to expand into other languages as we progress.

Insert Code Into Text File:

`find . -type f \( -name "*.py" -o -name "*.toml" -o -name "*.yaml" -o -name "*.json" -o -name "*.md" \) -exec cat {} + > codebase.txt`

Encode:

`poetry run python src/main.py encode data/raw_data.txt data/training_data.txt`

Train:

`poetry run python src/main.py train`

Predict:

`poetry run python src/main.py predict`

Concat the project files so that the project can be seen by the Ai. 
```cat src/*.py > codebase.py```

View Tensorboard
```tensorboard --logdir=tb_logs --bind_all```

Key Features of the Project:

Transformer-Based Model: Leveraging the power of the Transformer architecture to process and generate programming code.
PyTorch Framework: Utilizing the flexibility and robustness of PyTorch for model development and training.
Focus on Python: Initial emphasis on mastering Python, given its prominence and versatility in programming and AI.
Scalable Design: Building the model with scalability in mind, optimized for training on Google Colab's A100 GPUs.
Future Expansion: Plans to incorporate more programming languages and potentially develop a web-based UI for interacting with the model.
Project Roadmap:

Project Setup: Establishing a solid foundation with a well-structured repository and clear coding standards.
Model Architecture Design: Crafting a Transformer model tailored to understand programming languages.
Data Collection & Processing: Amassing a diverse dataset of code examples and documentation for training.
Model Training & Evaluation: Rigorously training and fine-tuning the model to ensure accuracy and efficiency.
Application Development: Building interfaces and applications to demonstrate and utilize the model's capabilities.
Continuous Improvement: Iterative enhancements and expansion into other languages and functionalities.
Contributions & Collaboration:
We welcome contributions, suggestions, and collaborations from enthusiasts and experts alike. Whether you're a seasoned AI researcher, a programming language guru, or just passionate about AI and coding, your input can help shape the future of this project.

Let’s embark on this exciting journey to push the boundaries of AI in programming language understanding and generation!
