# Recommender Systems for Steam Video Games - ALS and NCF
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-%23E25A1C.svg?style=for-the-badge&logo=apache-spark&logoColor=white)


## Overview
This project aims to build a recommender system for Steam video games using two distinct approaches: **Alternating Least Squares (ALS)** and **Neural Collaborative Filtering (NCF)**. The dataset consists of user interactions with Steam games, providing information such as:

* **user_id**: Unique identifier for users.
* **name**: Title of the game.
* **hours**: Time spent playing.
* **action**: Whether the game was purchased or played.
* **zero**: Unknown variable.

You can access the dataset here: [Steam Video Games Data](https://www.kaggle.com/datasets/tamber/steam-video-games/data)

Although the dataset contains 200,000 records, which could be handled using **pandas**, I opted to use **PySpark** for data preprocessing to gain experience with distributed data processing tools. This choice helps develop proficiency in handling larger datasets efficiently and prepares for future projects where scalability might be critical.

Since the dataset lacks explicit feedback (like ratings), we utilize implicit feedback by leveraging the hours feature. The assumption is that the more time a user spends on a game, the stronger their preference for that game, making hours a proxy for preference.

## Models:

* **Alternating Least Squares (ALS):** A collaborative filtering model implemented with PySpark, which is well-suited for implicit feedback data and scales efficiently with larger datasets. ALS leverages matrix factorization to learn latent factors for users and games.
* **Neural Collaborative Filtering (NCF):** A deep learning-based recommendation approach. It models user-game interactions using neural networks, specifically designed to capture non-linear user-item relationships, which may not be captured by traditional matrix factorization techniques like ALS.

## Tech Stack:

* **PySpark**: For data preprocessing and implementing the ALS model.
* **TensorFlow**: For building and training the NCF model.
* **Jupyter Notebook**: For development and experimentation.
* **NumPy & pandas**: For data manipulation.
* **Scikit-Learn**: Used for evaluation metric and train-test splitting for the NCF model.


## Result

* The NCF model outperformed the ALS model, showing a lower RMSE.

## License

This project is licensed under the MIT License.

