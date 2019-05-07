import numpy as np
from ballet import Feature
from ballet.eng import SimpleFunctionTransformer

input = ["Lot Area", "Lot Frontage"]


def fill_frontage(df):
    mask = df["Lot Frontage"].isnull()
    df["Lot Frontage"][mask] = np.sqrt(df["Lot Area"])[mask]
    return df["Lot Frontage"]


transformer = SimpleFunctionTransformer(fill_frontage)
name = "Lot Frontage Fill"
feature = Feature(input=input, transformer=transformer, name=name)
