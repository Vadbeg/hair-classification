"""Module with transforming json coco dataset annotations files to csv"""


from typing import Dict

import pandas as pd


def transform_coco_to_dataframe(coco_metadata: Dict, is_valid_dataset: bool = False) -> pd.DataFrame:
    dataframe = pd.DataFrame(coco_metadata['images'])

    if not is_valid_dataset:
        annotations_dataframe = pd.DataFrame(coco_metadata['annotations'])
        categories_dataframe = pd.DataFrame(coco_metadata['categories'])
        categories_dataframe = categories_dataframe.rename(columns={
            'id': 'category_id'
        })

        dataframe = dataframe.merge(annotations_dataframe, on='id', how='inner')
        dataframe = dataframe.merge(categories_dataframe, on='category_id', how='inner')

    return dataframe

