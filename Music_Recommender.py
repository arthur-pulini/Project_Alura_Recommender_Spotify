import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

#Análise gráfica
#A análise do Loudness será feita separada, pois, o range é de -60 a 0 db, sendo assim, será difícil a comparação com os outros dados
fig = px.line(datasYears, x="year", y="loudness", markers=True, title="Loudness variation according to the years")
#fig.show()

#Fazendo a análise do ano com os outros dados
fig = go.Figure()
fig.add_trace(go.Scatter(x=datasYears['year'], y=datasYears['acousticness'],
                    name='Acousticness'))
fig.add_trace(go.Scatter(x=datasYears['year'], y=datasYears['valence'],
                    name='Valence'))
fig.add_trace(go.Scatter(x=datasYears['year'], y=datasYears['danceability'],
                    name='Danceability'))
fig.add_trace(go.Scatter(x=datasYears['year'], y=datasYears['energy'],
                    name='Energy'))
fig.add_trace(go.Scatter(x=datasYears['year'], y=datasYears['instrumentalness'],
                    name='Instrumentalness'))
fig.add_trace(go.Scatter(x=datasYears['year'], y=datasYears['liveness'],
                    name='Liveness'))
fig.add_trace(go.Scatter(x=datasYears['year'], y=datasYears['speechiness'],
                    name='Speechiness'))
#fig.show()

datasV1 = datas.drop(["artists", "name", "artists_song", "id"], axis=1) #Para fazer a matriz de correlação foi necessário dropar as linhas com letras
print(datasV1)
fig = px.imshow(datasV1.corr(), text_auto=True)
fig.show()