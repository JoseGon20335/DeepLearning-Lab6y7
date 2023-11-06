import pandas as pd
import pickle
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for the file paths
BOOKS_CSV = "./data/Books.csv"
RATINGS_CSV = "./data/Ratings.csv"
USERS_CSV = "./data/Users.csv"
PICKLE_DIR = 'dataProcess'


def save_samples(users_df: pd.DataFrame, ratings_df: pd.DataFrame, books_df: pd.DataFrame) -> None:
    if not os.path.exists(PICKLE_DIR):
        os.makedirs(PICKLE_DIR)

    users_df.to_pickle(os.path.join(PICKLE_DIR, 'usersProcess.pkl'))
    ratings_df.to_pickle(os.path.join(PICKLE_DIR, 'ratingProcess.pkl'))
    books_df.to_pickle(os.path.join(PICKLE_DIR, 'bookProcess.pkl'))
    logging.info("Data samples have been saved as pickle files.")


def get_saved_samples() -> tuple:
    try:
        usersProcess = pd.read_pickle(os.path.join(PICKLE_DIR, 'usersProcess.pkl'))
        ratingProcess = pd.read_pickle(os.path.join(PICKLE_DIR, 'ratingProcess.pkl'))
        bookProcess = pd.read_pickle(os.path.join(PICKLE_DIR, 'bookProcess.pkl'))
        return usersProcess, ratingProcess, bookProcess
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise


def get_data_samples(fraction: float = 0.1, save: bool = True) -> tuple:
    if not 0 < fraction <= 1:
        fraction = 0.1
        logging.warning("Fraction value out of bounds, reset to 0.1.")

    try:
        users_df = pd.read_csv(USERS_CSV)
        ratings_df = pd.read_csv(RATINGS_CSV)
        books_df = pd.read_csv(BOOKS_CSV)
    except FileNotFoundError as e:
        logging.error(f"CSV file not found: {e}")
        raise

    usersProcess = users_df.sample(frac=fraction)

    ratingProcess = ratings_df[ratings_df['User-ID'].isin(usersProcess['User-ID'])]
    bookProcess = books_df[books_df['ISBN'].isin(ratingProcess['ISBN'])]

    # Log sizes after filtering
    logging.info(f"Users: {usersProcess.shape[0]} entries after sampling")
    logging.info(f"Ratings: {ratingProcess.shape[0]} entries after filtering")
    logging.info(f"Books: {bookProcess.shape[0]} entries after filtering")

    if save:
        save_samples(usersProcess, ratingProcess, bookProcess)

    return usersProcess, ratingProcess, bookProcess


if __name__ == "__main__":
    try:
        get_data_samples()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
