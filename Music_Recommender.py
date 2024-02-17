import pandas as pd
import numpy as np

datas = pd.read_csv('/home/arthur-pulini/Documentos/Programação/Machine learning Alura/Alura-Spotify/Dados_totais.csv')
datasMusicalGenre = pd.read_csv('/home/arthur-pulini/Documentos/Programação/Machine learning Alura/Alura-Spotify/data_by_genres.csv')
datasYears = pd.read_csv('/home/arthur-pulini/Documentos/Programação/Machine learning Alura/Alura-Spotify/data_by_year.csv')

print(datas.head(2))
print(datas["year"].unique())
print(datas.shape)
datas = datas.drop(["explicit", "key", "mode"], axis=1)
print(datas.shape)
print(datas.isnull().sum(), "\n\n",  datas.isna().sum())

print(datasMusicalGenre.head(2))
datasMusicalGenre = datasMusicalGenre.drop(["key", "mode"], axis=1)
print(datasMusicalGenre.shape)
print(datasMusicalGenre.isnull().sum(), "\n\n",  datasMusicalGenre.isna().sum())

print(datasYears.head(2))
print(datasYears["year"].unique())
datasYears = datasYears[datasYears["year"]>=2000] #igualando aos anos da tabela datas
print(datasYears["year"].unique())
datasYears = datasYears.drop(["key", "mode"], axis=1)
print(datasYears.head(2))
datasYears = datasYears.reset_index()
print(datasYears.head(2))
print(datasYears.shape)
print(datasMusicalGenre.isnull().sum(), "\n\n",  datasMusicalGenre.isna().sum())
