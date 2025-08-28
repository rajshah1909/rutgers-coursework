from collections import defaultdict
from collections import Counter

# ------ TASK 1: READING DATA  --------

# 1.1
def read_ratings_data(f):
    movie_ratings_dict = {}
    with open(f, 'r') as file:
        for line in file:
            movie, rating, _ = line.strip().split('|')
            rating = float(rating)
            if movie not in movie_ratings_dict:
                movie_ratings_dict[movie] = []
            movie_ratings_dict[movie].append(rating)
    return movie_ratings_dict
    

# 1.2
def read_movie_genre(f):
    movie_to_genre = {}
    with open(f, 'r') as file:
        for line in file:
            genre, _, movie = line.strip().split('|')
            movie_to_genre[movie.strip()] = genre.strip()
    return movie_to_genre

# ------ TASK 2: PROCESSING DATA --------

# 2.1
def create_genre_dict(d):
    genre_dict = {}
    for movie, genre in d.items():
        if genre not in genre_dict:
            genre_dict[genre] = []
        genre_dict[genre].append(movie)
    return genre_dict
    
# 2.2
def calculate_average_rating(d):
    return {
        movie: round(sum(ratings) / len(ratings), 2) 
        for movie, ratings in d.items()
    }
    
# ------ TASK 3: RECOMMENDATION --------

# 3.1
def get_popular_movies(d, n=10):
    sorted_movies = sorted(d.items(), key=lambda x: (-x[1], x[0]))
    return dict(sorted_movies[:n])
    
# 3.2
def filter_movies(d, thres_rating=3):
    return {movie: rating for movie, rating in d.items() if rating >= thres_rating}
    
# 3.3
def get_popular_in_genre(genre, genre_to_movies, movie_to_average_rating, n=5):
    if genre not in genre_to_movies:
        return {}
    
    genre_ratings = {
        movie: movie_to_average_rating[movie]
        for movie in genre_to_movies[genre]
        if movie in movie_to_average_rating
    }
    
    return get_popular_movies(genre_ratings, n)
    
# 3.4
def get_genre_rating(genre, genre_to_movies, movie_to_average_rating):
    if genre not in genre_to_movies:
        return 0.0
    
    ratings = [
        movie_to_average_rating[movie]
        for movie in genre_to_movies[genre]
        if movie in movie_to_average_rating
    ]
    
    return round(sum(ratings) / len(ratings), 2) if ratings else 0.0
    
# 3.5
def genre_popularity(genre_to_movies, movie_to_average_rating, n=5):
    genre_avg = {
        genre: round(sum(movie_to_average_rating[movie] for movie in movies if movie in movie_to_average_rating) / len(movies), 2)
        for genre, movies in genre_to_movies.items()
        if any(movie in movie_to_average_rating for movie in movies)
    }
    
    return get_popular_movies(genre_avg, n)

# ------ TASK 4: USER FOCUSED  --------

# 4.1
def read_user_ratings(f):
    user_ratings = {}
    with open(f, 'r') as file:
        for line in file:
            movie, rating, user = line.strip().split('|')
            user_id = int(user)
            rating = float(rating)
            if user_id not in user_ratings:
                user_ratings[user_id] = []
            user_ratings[user_id].append((movie, rating))
    return user_ratings
    
# 4.2
def get_user_genre(user_id, user_to_movies, movie_to_genre):
    if user_id not in user_to_movies:
        return None
    
    genre_sums = {}
    genre_counts = {}
    
    for movie, rating in user_to_movies[user_id]:
        if movie in movie_to_genre:
            genre = movie_to_genre[movie]
            genre_sums[genre] = genre_sums.get(genre, 0) + rating
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    return max(genre_sums, key=lambda g: genre_sums[g] / genre_counts[g]) if genre_sums else None
    
# 4.3    
def recommend_movies(user_id, user_to_movies, movie_to_genre, movie_to_average_rating):
    if user_id not in user_to_movies:
        return {}
    
    favorite_genre = get_user_genre(user_id, user_to_movies, movie_to_genre)
    if not favorite_genre:
        return {}
    
    user_seen = {movie for movie, _ in user_to_movies[user_id]}
    
    eligible_movies = {
        movie: movie_to_average_rating[movie]
        for movie, genre in movie_to_genre.items()
        if genre == favorite_genre and movie not in user_seen and movie in movie_to_average_rating
    }
    
    return dict(sorted(eligible_movies.items(), key=lambda x: (-x[1], x[0]))[:3])

def main():
    ratings_file = "sampleratingsfile.txt"
    movies_file = "samplemoviesfile.txt"
    
    ratings_dict = read_ratings_data(ratings_file)
    print("=== Ratings Data ===")
    print({k: v[:3] for k, v in list(ratings_dict.items())[:5]})  

    movie_genre_dict = read_movie_genre(movies_file)
    print("\n=== Movie Genre Data ===")
    print({k: v for k, v in list(movie_genre_dict.items())[:5]})  

    genre_dict = create_genre_dict(movie_genre_dict)
    avg_ratings = calculate_average_rating(ratings_dict)

    print("\n=== Average Ratings ===")
    print({k: v for k, v in list(avg_ratings.items())[:5]})  

    popular_movies = get_popular_movies(avg_ratings, 5)
    print("\n=== Top 5 Popular Movies ===")
    print(popular_movies)

    user_dict = read_user_ratings(ratings_file)
    example_user = list(user_dict.keys())[0]  

    user_fav_genre = get_user_genre(example_user, user_dict, movie_genre_dict)
    print(f"\n=== User {example_user}'s Favorite Genre ===")
    print(user_fav_genre)

    recommendations = recommend_movies(example_user, user_dict, movie_genre_dict, avg_ratings)
    print(f"\n=== Recommendations for User {example_user} ===")
    print(recommendations)

main() 