from dataclasses import dataclass
from typing import Union
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.preprocessing import add_categorical_columns, add_derived_title, add_is_alone_column, impute_nans, train_model


@dataclass
class PassengersRaw:
    PassengerId: pd.Series
    Survived: pd.Series
    Pclass: pd.Series
    Name: pd.Series
    Sex: pd.Series
    Age: pd.Series
    SibSp: pd.Series
    Parch: pd.Series
    Ticket: pd.Series
    Fare: pd.Series
    Cabin: pd.Series
    Embarked: pd.Series


PassengersRawDataFrame = Union[pd.DataFrame, PassengersRaw]


@dataclass
class PreprocessedFeaturesWithTarget:
    Survived: pd.Series
    Pclass: pd.Series
    Sex: pd.Series
    Age: pd.Series
    Fare: pd.Series
    Embarked: pd.Series
    Title: pd.Series
    IsAlone: pd.Series
    AgeGroup: pd.Series
    FareBand: pd.Series


FeaturesAndTargetDataFrame = Union[pd.DataFrame, PreprocessedFeaturesWithTarget]


@dataclass
class PreprocessedFeatures:
    Survived: pd.Series
    Pclass: pd.Series
    Sex: pd.Series
    Age: pd.Series
    Fare: pd.Series
    Embarked: pd.Series
    Title: pd.Series
    IsAlone: pd.Series
    AgeGroup: pd.Series
    FareBand: pd.Series


FeaturesDataFrame = Union[pd.DataFrame, PreprocessedFeatures]


def prepare_data_and_train_model():
    df = pd.read_csv("./data/train.csv")

    df: PassengersRawDataFrame = impute_nans(df, categorical_columns=[
        'Embarked'], continuous_columns=['Fare', 'Age'])
    df = add_derived_title(df)
    df = add_is_alone_column(df)
    df = add_categorical_columns(df)

    df: FeaturesAndTargetDataFrame = df.drop(['Parch', 'SibSp', 'Name', 'PassengerId',
                                              'Ticket', 'Cabin'], axis=1)

    Y = df["Survived"]
    X: FeaturesDataFrame = df.drop("Survived", axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    rf_model, accuracy_random_forest = train_model(
        RandomForestClassifier, X_train, Y_train, n_estimators=100)

    return rf_model, X_test, Y_test


if __name__ == '__main__':
    model, X_test, Y_test = prepare_data_and_train_model()
