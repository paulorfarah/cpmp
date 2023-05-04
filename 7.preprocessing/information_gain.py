import pandas as pd
import re
import numpy as np

import category_encoders as ce

from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif


def weka_tokenizer(doc):
    delimiters_regexp = re.compile("[ |\n|\f|\r|\t|.|,|;|:|'|\"|(|)|?|!]")
    # delimiters_regexp = re.compile("[ -\/:-@\[-\`{-~|0-9|\n|\f|\r|\t|\s]")
    return list(filter(None, delimiters_regexp.split(doc)))

def information_gain(databases, main_columns, feature_names):
    for db in databases:
        all_releases_df = pd.read_csv(
            '../6.join_metrics/results/' + db + '-all-releases.csv')#, usecols=main_columns)
        all_releases_df.columns = main_columns
        all_releases_df = all_releases_df.fillna(0)

        total_data_X = np.array(all_releases_df[feature_names].copy())
        total_data_Y = np.array(pd.DataFrame(all_releases_df.loc[:, 'will_change']))

        encoder = ce.LeaveOneOutEncoder(return_df=True)
        X = encoder.fit_transform(total_data_X, total_data_Y)

        selector = SelectKBest(mutual_info_classif, k=20)
        X_reduced = selector.fit_transform(X, total_data_Y)
        print(X_reduced.shape)

        informationGain = dict(zip(X.columns, mutual_info_classif(X, total_data_Y, discrete_features=True)))
        sortedInformationGain = sorted(informationGain, key=informationGain.get, reverse=True)
        i = 0
        sortedInformationGainPosition = []

        for r in sortedInformationGain:
            infGain = {
                'position': i,
                'metric': all_releases_df.columns[r],
                'information_gain': informationGain[r],
                'total_occurences': len(X[X[r] > 0])
            }

            sortedInformationGainPosition.append(infGain)
            i += 1

            print(i, all_releases_df.columns[r], informationGain[r])

        infGainCSV = pd.DataFrame(sortedInformationGainPosition,
                                  columns=['position', 'metric', 'information_gain', 'total_ocurences'])
        infGainCSV.to_csv('results/information_gain_' + db + '.csv')

        cols = selector.get_support(indices=True)
        selected_columns = X.iloc[:, cols].columns.tolist()


