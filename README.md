# Movie Recommendation System
The idea for this app came to me one day when I was trying to decide which movie to watch. Google provided some good results, but I had already seen all of them. I ended up changing genres, which made me think about using the power of machine learning to create an algorithm that could provide better recommendations. This way, I could expand the list of suggestions as much as I wanted. The dataset I worked with contained over 100 million movies, which was challenging to manage, but I was able to handle it.

I used the following libraries for this project: Flask, Surprise, SVD, KNNBasic, pandas, matplotlib, and others.

However, I encountered difficulties deploying the app with Flask because many websites did not support the Surprise library. As a result, I switched to Streamlit.

I didn't use many of the dataset's features, as a lot of them weren't relevant for this website.

# Future Work:
First, I want to deploy the website with Flask, as it offers many options and supports templates, which Streamlit lacks. I plan to use additional dataset columns, such as tags, to give users insights into how people reacted to the movies and whether they liked them or not. I also want to integrate this recommendation system with one of my other websites, specifically a mental health chatbot. This chatbot could suggest movies or shows to help users relax or improve their mood.
