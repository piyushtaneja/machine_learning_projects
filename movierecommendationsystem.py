import pandas as pd
import numpy as np  
import warnings
warnings.filterwarnings("ignore")

#get the data set
columns_name=["user_id","item_id","rating","timestamp"]

df=pd.read_csv("ml-100k/u.data",sep='\t',names=columns_name)

print(df.head())
df['user_id'].nunique()
df['item_id'].nunique()
movies_title=pd.read_csv("ml-100k/u.item",sep="|",header=None,encoding='latin-1')#here movie title is another file which we bring from csv file 
movies_title.shape
movies_title=movies_title[[0,1]] #we keep only first two coloumns
movies_title.columns=['item_id','title']
movies_title.head()
df =pd.merge(df,movies_title,on="item_id")
df.tail()
#we merge it differently becuase the movie tile data and rating data where in different file in the csv
#exploratory data analysis
import matplotlib.pyplot as plt  
import seaborn as sns  
sns.set_style('white')

(df.groupby('title').mean()['rating'].sort_values(ascending=False)) #here we are calculating average rating for each movie
df.groupby('title').count()['rating'].sort_values(ascending=False)  #counts the number of rating each movie has received

#now we will create a data frame of ratings
ratings=pd.DataFrame(df.groupby('title').mean()['rating'])
ratings.head()

ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])#adds a new coloumn number of ratings to ratings which stores the count of ratings for each movie
ratings.sort_values(by='rating',ascending=False)

plt.figure(figsize=(10,6))
plt.hist(ratings['num of ratings'],bins=70)
plt.show()
plt.hist(ratings['rating'],bins=70)
plt.show()

sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)
moviemat=df.pivot_table(index='user_id',columns='title',values='rating') 
moviemat.head()
ratings.sort_values('num of ratings',ascending=False).head()
starwar_user_ratings=moviemat['Star Wars (1977)']
starwar_user_ratings.head()

similar_to_starwar=moviemat.corrwith(starwar_user_ratings)
similar_to_starwar 
corr_starwars=pd.DataFrame(similar_to_starwar,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()

#now we will sort the movies which are similiar to star wars by checking the correlation
corr_starwars.sort_values('Correlation',ascending=False).head(10)

#now we are putting threshold like minimum 100 user review
corr_starwars=corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()

corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False)

#predict function

def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name] #here we are checking the movie rating from moviemat
    similar_to_movie=moviemat.corrwith(movie_user_ratings)#now we are seeing the corr movies or we can say this compute the correlation between ratings of the specified movie
    corr_movie=pd.DataFrame(similar_to_movie,columns=['Correlation'])#creating a data frame to store correlations
    corr_movie.dropna(inplace=True)#this is used to remove movies which have no correlation and have nana there
    corr_movie=corr_movie.join(ratings['num of ratings'])#here we are joining corr movies with num of rating
    prediction=corr_movie[corr_movie['num of ratings']>100].sort_values('Correlation',ascending=False)#here we are predicitng and filterning the movies which have less than 100 ratings
    return prediction

prediction=predict_movies("Titanic (1997)")
prediction.head()                                                                  
                                                                                     




