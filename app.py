import pickle
import pandas
import streamlit as st


classifier = pickle.load(open('Model.pkl', 'rb'))
cv = pickle.load(open('countvector.pkl','rb'))


st.title("Lets see how did you like the movie")


Movie_name = st.text_input("Which movie did you watch ?")
review = st.text_input("Please write a review")


if st.button("Submit"):
	data = [review]
	vect = cv.transform(data).toarray()
	prediction = (classifier.predict(vect))
	
	if prediction == [0]:
		st.header("We are sorry to hear that you didn't like {}".format(Movie_name))
	
	else :
		st.header("We are glad you loved {}".format(Movie_name))