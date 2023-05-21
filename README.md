---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.9
  nbformat: 4
  nbformat_minor: 5
---

<div class="cell markdown">

# Movies Recommendation System

## Dataset and Notebook Description

In this kernel we'll be building a baseline **Movie Recommendation
System** using **[TMDB 5000 Movie
Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata).** For novices
like me this kernel will pretty much serve as a foundation in
recommendation systems and will provide you with something to start
with. **There are basically three types of recommender systems:-**

> -   **Demographic Filtering**- They offer generalized recommendations
>     to every user, based on movie popularity and/or genre. The System
>     recommends the same movies to users with similar demographic
>     features. Since each user is different , this approach is
>     considered to be too simple. The basic idea behind this system is
>     that movies that are more popular and critically acclaimed will
>     have a higher probability of being liked by the average audience.

> -   **Content Based Filtering**- They suggest similar items based on a
>     particular item. This system uses item metadata, such as genre,
>     director, description, actors, etc. for movies, to make these
>     recommendations. The general idea behind these recommender systems
>     is that if a person liked a particular item, he or she will also
>     like an item that is similar to it.

> -   **Collaborative Filtering**- This system matches persons with
>     similar interests and provides recommendations based on this
>     matching. Collaborative filters do not require item metadata like
>     its content-based counterparts.

<img src='https://repository-images.githubusercontent.com/586636943/834e392b-1245-4dbc-96c1-2698587a2dde'/>

</div>

<div class="cell markdown">

# Import Libraries

</div>

<div class="cell code" execution_count="1">

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

</div>

<div class="cell markdown">

# Read Movies Datasets

</div>

<div class="cell code" execution_count="2">

``` python
df1=pd.read_csv('tmdb_5000_credits.csv')
df2=pd.read_csv('tmdb_5000_movies.csv')
```

</div>

<div class="cell code" execution_count="3">

``` python
df1
```

<div class="output execute_result" execution_count="3">

          movie_id                                     title   
    0        19995                                    Avatar  \
    1          285  Pirates of the Caribbean: At World's End   
    2       206647                                   Spectre   
    3        49026                     The Dark Knight Rises   
    4        49529                               John Carter   
    ...        ...                                       ...   
    4798      9367                               El Mariachi   
    4799     72766                                 Newlyweds   
    4800    231617                 Signed, Sealed, Delivered   
    4801    126186                          Shanghai Calling   
    4802     25975                         My Date with Drew   

                                                       cast   
    0     [{"cast_id": 242, "character": "Jake Sully", "...  \
    1     [{"cast_id": 4, "character": "Captain Jack Spa...   
    2     [{"cast_id": 1, "character": "James Bond", "cr...   
    3     [{"cast_id": 2, "character": "Bruce Wayne / Ba...   
    4     [{"cast_id": 5, "character": "John Carter", "c...   
    ...                                                 ...   
    4798  [{"cast_id": 1, "character": "El Mariachi", "c...   
    4799  [{"cast_id": 1, "character": "Buzzy", "credit_...   
    4800  [{"cast_id": 8, "character": "Oliver O\u2019To...   
    4801  [{"cast_id": 3, "character": "Sam", "credit_id...   
    4802  [{"cast_id": 3, "character": "Herself", "credi...   

                                                       crew  
    0     [{"credit_id": "52fe48009251416c750aca23", "de...  
    1     [{"credit_id": "52fe4232c3a36847f800b579", "de...  
    2     [{"credit_id": "54805967c3a36829b5002c41", "de...  
    3     [{"credit_id": "52fe4781c3a36847f81398c3", "de...  
    4     [{"credit_id": "52fe479ac3a36847f813eaa3", "de...  
    ...                                                 ...  
    4798  [{"credit_id": "52fe44eec3a36847f80b280b", "de...  
    4799  [{"credit_id": "52fe487dc3a368484e0fb013", "de...  
    4800  [{"credit_id": "52fe4df3c3a36847f8275ecf", "de...  
    4801  [{"credit_id": "52fe4ad9c3a368484e16a36b", "de...  
    4802  [{"credit_id": "58ce021b9251415a390165d9", "de...  

    [4803 rows x 4 columns]

</div>

</div>

<div class="cell code" execution_count="4">

``` python
df2
```

<div class="output execute_result" execution_count="4">

             budget                                             genres   
    0     237000000  [{"id": 28, "name": "Action"}, {"id": 12, "nam...  \
    1     300000000  [{"id": 12, "name": "Adventure"}, {"id": 14, "...   
    2     245000000  [{"id": 28, "name": "Action"}, {"id": 12, "nam...   
    3     250000000  [{"id": 28, "name": "Action"}, {"id": 80, "nam...   
    4     260000000  [{"id": 28, "name": "Action"}, {"id": 12, "nam...   
    ...         ...                                                ...   
    4798     220000  [{"id": 28, "name": "Action"}, {"id": 80, "nam...   
    4799       9000  [{"id": 35, "name": "Comedy"}, {"id": 10749, "...   
    4800          0  [{"id": 35, "name": "Comedy"}, {"id": 18, "nam...   
    4801          0                                                 []   
    4802          0                [{"id": 99, "name": "Documentary"}]   

                                                   homepage      id   
    0                           http://www.avatarmovie.com/   19995  \
    1          http://disney.go.com/disneypictures/pirates/     285   
    2           http://www.sonypictures.com/movies/spectre/  206647   
    3                    http://www.thedarkknightrises.com/   49026   
    4                  http://movies.disney.com/john-carter   49529   
    ...                                                 ...     ...   
    4798                                                NaN    9367   
    4799                                                NaN   72766   
    4800  http://www.hallmarkchannel.com/signedsealeddel...  231617   
    4801                        http://shanghaicalling.com/  126186   
    4802                                                NaN   25975   

                                                   keywords original_language   
    0     [{"id": 1463, "name": "culture clash"}, {"id":...                en  \
    1     [{"id": 270, "name": "ocean"}, {"id": 726, "na...                en   
    2     [{"id": 470, "name": "spy"}, {"id": 818, "name...                en   
    3     [{"id": 849, "name": "dc comics"}, {"id": 853,...                en   
    4     [{"id": 818, "name": "based on novel"}, {"id":...                en   
    ...                                                 ...               ...   
    4798  [{"id": 5616, "name": "united states\u2013mexi...                es   
    4799                                                 []                en   
    4800  [{"id": 248, "name": "date"}, {"id": 699, "nam...                en   
    4801                                                 []                en   
    4802  [{"id": 1523, "name": "obsession"}, {"id": 224...                en   

                                    original_title   
    0                                       Avatar  \
    1     Pirates of the Caribbean: At World's End   
    2                                      Spectre   
    3                        The Dark Knight Rises   
    4                                  John Carter   
    ...                                        ...   
    4798                               El Mariachi   
    4799                                 Newlyweds   
    4800                 Signed, Sealed, Delivered   
    4801                          Shanghai Calling   
    4802                         My Date with Drew   

                                                   overview  popularity   
    0     In the 22nd century, a paraplegic Marine is di...  150.437577  \
    1     Captain Barbossa, long believed to be dead, ha...  139.082615   
    2     A cryptic message from Bond’s past sends him o...  107.376788   
    3     Following the death of District Attorney Harve...  112.312950   
    4     John Carter is a war-weary, former military ca...   43.926995   
    ...                                                 ...         ...   
    4798  El Mariachi just wants to play his guitar and ...   14.269792   
    4799  A newlywed couple's honeymoon is upended by th...    0.642552   
    4800  "Signed, Sealed, Delivered" introduces a dedic...    1.444476   
    4801  When ambitious New York attorney Sam is sent t...    0.857008   
    4802  Ever since the second grade when he first saw ...    1.929883   

                                       production_companies   
    0     [{"name": "Ingenious Film Partners", "id": 289...  \
    1     [{"name": "Walt Disney Pictures", "id": 2}, {"...   
    2     [{"name": "Columbia Pictures", "id": 5}, {"nam...   
    3     [{"name": "Legendary Pictures", "id": 923}, {"...   
    4           [{"name": "Walt Disney Pictures", "id": 2}]   
    ...                                                 ...   
    4798           [{"name": "Columbia Pictures", "id": 5}]   
    4799                                                 []   
    4800  [{"name": "Front Street Pictures", "id": 3958}...   
    4801                                                 []   
    4802  [{"name": "rusty bear entertainment", "id": 87...   

                                       production_countries release_date   
    0     [{"iso_3166_1": "US", "name": "United States o...   2009-12-10  \
    1     [{"iso_3166_1": "US", "name": "United States o...   2007-05-19   
    2     [{"iso_3166_1": "GB", "name": "United Kingdom"...   2015-10-26   
    3     [{"iso_3166_1": "US", "name": "United States o...   2012-07-16   
    4     [{"iso_3166_1": "US", "name": "United States o...   2012-03-07   
    ...                                                 ...          ...   
    4798  [{"iso_3166_1": "MX", "name": "Mexico"}, {"iso...   1992-09-04   
    4799                                                 []   2011-12-26   
    4800  [{"iso_3166_1": "US", "name": "United States o...   2013-10-13   
    4801  [{"iso_3166_1": "US", "name": "United States o...   2012-05-03   
    4802  [{"iso_3166_1": "US", "name": "United States o...   2005-08-05   

             revenue  runtime                                   spoken_languages   
    0     2787965087    162.0  [{"iso_639_1": "en", "name": "English"}, {"iso...  \
    1      961000000    169.0           [{"iso_639_1": "en", "name": "English"}]   
    2      880674609    148.0  [{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...   
    3     1084939099    165.0           [{"iso_639_1": "en", "name": "English"}]   
    4      284139100    132.0           [{"iso_639_1": "en", "name": "English"}]   
    ...          ...      ...                                                ...   
    4798     2040920     81.0      [{"iso_639_1": "es", "name": "Espa\u00f1ol"}]   
    4799           0     85.0                                                 []   
    4800           0    120.0           [{"iso_639_1": "en", "name": "English"}]   
    4801           0     98.0           [{"iso_639_1": "en", "name": "English"}]   
    4802           0     90.0           [{"iso_639_1": "en", "name": "English"}]   

            status                                            tagline   
    0     Released                        Enter the World of Pandora.  \
    1     Released     At the end of the world, the adventure begins.   
    2     Released                              A Plan No One Escapes   
    3     Released                                    The Legend Ends   
    4     Released               Lost in our world, found in another.   
    ...        ...                                                ...   
    4798  Released  He didn't come looking for trouble, but troubl...   
    4799  Released  A newlywed couple's honeymoon is upended by th...   
    4800  Released                                                NaN   
    4801  Released                           A New Yorker in Shanghai   
    4802  Released                                                NaN   

                                             title  vote_average  vote_count  
    0                                       Avatar           7.2       11800  
    1     Pirates of the Caribbean: At World's End           6.9        4500  
    2                                      Spectre           6.3        4466  
    3                        The Dark Knight Rises           7.6        9106  
    4                                  John Carter           6.1        2124  
    ...                                        ...           ...         ...  
    4798                               El Mariachi           6.6         238  
    4799                                 Newlyweds           5.9           5  
    4800                 Signed, Sealed, Delivered           7.0           6  
    4801                          Shanghai Calling           5.7           7  
    4802                         My Date with Drew           6.3          16  

    [4803 rows x 20 columns]

</div>

</div>

<div class="cell markdown">

# **Demographic Filtering** -

Before getting started with this -

-   we need a metric to score or rate movie
-   Calculate the score for every movie
-   Sort the scores and recommend the best rated movie to the users.

We can use the average ratings of the movie as the score but using this
won't be fair enough since a movie with 8.9 average rating and only 3
votes cannot be considered better than the movie with 7.8 as as average
rating but 40 votes. So, I'll be using IMDB's weighted rating (wr) which
is given as :-

![](https://image.ibb.co/jYWZp9/wr.png) where,

-   v is the number of votes for the movie;
-   m is the minimum votes required to be listed in the chart;
-   R is the average rating of the movie; And
-   C is the mean vote across the whole report

We already have v(**vote_count**) and R (**vote_average**) and C can be
calculated as

</div>

<div class="cell code" execution_count="5">

``` python
C= df2['vote_average'].mean()
C
```

<div class="output execute_result" execution_count="5">

    6.092171559442016

</div>

</div>

<div class="cell markdown">

**So, the mean rating for all the movies is approx 6 on a scale of
10.The next step is to determine an appropriate value for m, the minimum
votes required to be listed in the chart. We will use 90th percentile as
our cutoff. In other words, for a movie to feature in the charts, it
must have more votes than at least 90% of the movies in the list.**

</div>

<div class="cell code" execution_count="6">

``` python
m= df2['vote_count'].quantile(0.9)
m
```

<div class="output execute_result" execution_count="6">

    1838.4000000000015

</div>

</div>

<div class="cell markdown">

**Now, we can filter out the movies that qualify for the chart**

</div>

<div class="cell code" execution_count="7">

``` python
q_movies = df2.copy().loc[df2['vote_count'] >= m]
q_movies.shape
```

<div class="output execute_result" execution_count="7">

    (481, 20)

</div>

</div>

<div class="cell markdown">

We see that there are 481 movies which qualify to be in this list. Now,
we need to calculate our metric for each qualified movie. To do this, we
will define a function, **weighted_rating()** and define a new feature
**score**, of which we'll calculate the value by applying this function
to our DataFrame of qualified movies:

</div>

<div class="cell code" execution_count="8">

``` python
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)
```

</div>

<div class="cell code" execution_count="9">

``` python
# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
```

</div>

<div class="cell markdown">

Finally, let's sort the DataFrame based on the score feature and output
the title, vote count, vote average and weighted rating or score of the
top 15 movies.

</div>

<div class="cell code" execution_count="10">

``` python
#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15)
```

<div class="output execute_result" execution_count="10">

                                                      title  vote_count   
    1881                           The Shawshank Redemption        8205  \
    662                                          Fight Club        9413   
    65                                      The Dark Knight       12002   
    3232                                       Pulp Fiction        8428   
    96                                            Inception       13752   
    3337                                      The Godfather        5893   
    95                                         Interstellar       10867   
    809                                        Forrest Gump        7927   
    329       The Lord of the Rings: The Return of the King        8064   
    1990                            The Empire Strikes Back        5879   
    262   The Lord of the Rings: The Fellowship of the Ring        8705   
    2912                                          Star Wars        6624   
    1818                                   Schindler's List        4329   
    3865                                           Whiplash        4254   
    330               The Lord of the Rings: The Two Towers        7487   

          vote_average     score  
    1881           8.5  8.059258  
    662            8.3  7.939256  
    65             8.2  7.920020  
    3232           8.3  7.904645  
    96             8.1  7.863239  
    3337           8.4  7.851236  
    95             8.1  7.809479  
    809            8.2  7.803188  
    329            8.1  7.727243  
    1990           8.2  7.697884  
    262            8.0  7.667341  
    2912           8.1  7.663813  
    1818           8.3  7.641883  
    3865           8.3  7.633781  
    330            8.0  7.623893  

</div>

</div>

<div class="cell markdown">

Hurray! We have made our first(though very basic) recommender. Under the
**Trending Now** tab of these systems we find movies that are very
popular and they can just be obtained by sorting the dataset by the
popularity column.

</div>

<div class="cell code" execution_count="11">

``` python
pop= df2.sort_values('popularity', ascending=False)
```

</div>

<div class="cell code" execution_count="12">

``` python
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
```

<div class="output execute_result" execution_count="12">

    Text(0.5, 1.0, 'Popular Movies')

</div>

<div class="output display_data">

![](945fd10eb53ffa465891660377f16af3135cf197.png)

</div>

</div>

<div class="cell markdown">

# **Content Based Filtering**

**In this recommender system the content of the movie (overview, cast,
crew, keyword, tagline etc) is used to find its similarity with other
movies. Then the movies that are most likely to be similar are
recommended.**

<img src='https://image.ibb.co/f6mDXU/conten.png' />

</div>

<div class="cell markdown">

## **Plot description based Recommender**

We will compute pairwise similarity scores for all movies based on their
plot descriptions and recommend movies based on that similarity score.
The plot description is given in the **overview** feature of our
dataset. Let's take a look at the data. ..

</div>

<div class="cell code" execution_count="13">

``` python
df2['overview'].head(5)
```

<div class="output execute_result" execution_count="13">

    0    In the 22nd century, a paraplegic Marine is di...
    1    Captain Barbossa, long believed to be dead, ha...
    2    A cryptic message from Bond’s past sends him o...
    3    Following the death of District Attorney Harve...
    4    John Carter is a war-weary, former military ca...
    Name: overview, dtype: object

</div>

</div>

<div class="cell markdown">

For any of you who has done even a bit of text processing before knows
we need to convert the word vector of each overview. Now we'll compute
Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each
overview.

Now if you are wondering what is term frequency , it is the relative
frequency of a word in a document and is given as **(term
instances/total instances)**. Inverse Document Frequency is the relative
count of documents containing the term is given as **log(number of
documents/documents with term)** The overall importance of each word to
the documents in which they appear is equal to **TF \* IDF**

This will give you a matrix where each column represents a word in the
overview vocabulary (all the words that appear in at least one document)
and each row represents a movie, as before.This is done to reduce the
importance of words that occur frequently in plot overviews and
therefore, their significance in computing the final similarity score.

Fortunately, scikit-learn gives you a built-in TfIdfVectorizer class
that produces the TF-IDF matrix in a couple of lines. That's great,
isn't it?

</div>

<div class="cell code" execution_count="14">

``` python
tfidf = TfidfVectorizer(stop_words='english')
```

</div>

<div class="cell code" execution_count="15">

``` python
#Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')
```

</div>

<div class="cell code" execution_count="16">

``` python
#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])
```

</div>

<div class="cell code" execution_count="17">

``` python
#Output the shape of tfidf_matrix
tfidf_matrix.shape
```

<div class="output execute_result" execution_count="17">

    (4803, 20978)

</div>

</div>

<div class="cell markdown">

**Since we have used the TF-IDF vectorizer, calculating the dot product
will directly give us the cosine similarity score. Therefore, we will
use sklearn's **linear_kernel()\*\* instead of cosine_similarities()
since it is faster.\*\*

</div>

<div class="cell code" execution_count="18">

``` python
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```

</div>

<div class="cell markdown">

We are going to define a function that takes in a movie title as an
input and outputs a list of the 10 most similar movies. Firstly, for
this, we need a reverse mapping of movie titles and DataFrame indices.
In other words, we need a mechanism to identify the index of a movie in
our metadata DataFrame, given its title.

</div>

<div class="cell code" execution_count="19">

``` python
#Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
```

</div>

<div class="cell markdown">

We are now in a good position to define our recommendation function.
These are the following steps we'll follow :-

-   Get the index of the movie given its title.
-   Get the list of cosine similarity scores for that particular movie
    with all movies. Convert it into a list of tuples where the first
    element is its position and the second is the similarity score.
-   Sort the aforementioned list of tuples based on the similarity
    scores; that is, the second element.
-   Get the top 10 elements of this list. Ignore the first element as it
    refers to self (the movie most similar to a particular movie is the
    movie itself).
-   Return the titles corresponding to the indices of the top elements.

</div>

<div class="cell code" execution_count="20">

``` python
# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]
```

</div>

<div class="cell code" execution_count="21">

``` python
get_recommendations('The Dark Knight Rises')
```

<div class="output execute_result" execution_count="21">

    65                              The Dark Knight
    299                              Batman Forever
    428                              Batman Returns
    1359                                     Batman
    3854    Batman: The Dark Knight Returns, Part 2
    119                               Batman Begins
    2507                                  Slow Burn
    9            Batman v Superman: Dawn of Justice
    1181                                        JFK
    210                              Batman & Robin
    Name: title, dtype: object

</div>

</div>

<div class="cell code" execution_count="22">

``` python
get_recommendations('The Avengers')
```

<div class="output execute_result" execution_count="22">

    7               Avengers: Age of Ultron
    3144                            Plastic
    1715                            Timecop
    4124                 This Thing of Ours
    3311              Thank You for Smoking
    3033                      The Corruptor
    588     Wall Street: Money Never Sleeps
    2136         Team America: World Police
    1468                       The Fountain
    1286                        Snowpiercer
    Name: title, dtype: object

</div>

</div>
