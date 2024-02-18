import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

SEED = 1224
np.random.seed(SEED)

datas = pd.read_csv('/home/arthur-pulini/Documentos/Programação/Machine learning Alura/Alura-Spotify/Dados_totais.csv')
datasMusicalGenre = pd.read_csv('/home/arthur-pulini/Documentos/Programação/Machine learning Alura/Alura-Spotify/data_by_genres.csv')
datasYears = pd.read_csv('/home/arthur-pulini/Documentos/Programação/Machine learning Alura/Alura-Spotify/data_by_year.csv')

#Tratando os dados manualmente
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

datasV1 = datas.drop(["artists", "name", "artists_song", "id"], axis=1) #Para fazer a matriz de correlação foi necessário dropar as colunas não numéricas
print(datasV1)
fig = px.imshow(datasV1.corr(), text_auto=True)#Aplicando a matriz de correlação
#fig.show()

#Analisando o datasMusicalGenre, para ver se os gêneros se repetem
print(datasMusicalGenre['genres'].value_counts().sum())
datasMusicalGenreV1 = datasMusicalGenre.drop(['genres'], axis=1)#Para fazer o cluster é necessário dropar as colunas não numéricas
print(datasMusicalGenreV1)

#Configurando o Pipeline com o StandartScaler e PCA
pcaPipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2, random_state=SEED))])

#Aplicando o Pipeline nos dados por gêneros tratado
genreEmbeddingPca = pcaPipeline.fit_transform(datasMusicalGenreV1)
projection = pd.DataFrame(columns=['x', 'y'], data = genreEmbeddingPca)#Criando um DataFrame, a partir dos dados por gêneros transformados
print(projection)

#Configurando o KMeans
kmeansPca = KMeans(n_clusters=5, verbose=True, random_state=SEED)

#Aplicando KMeans
kmeansPca.fit(projection)

datasMusicalGenre['cluster_PCA'] = kmeansPca.predict(projection)#Adicionando o número dos clusters em datasMusicalGenre
projection['cluster_PCA'] = kmeansPca.predict(projection)#Adicionando o número dos clusters em projection
projection['genres'] = datasMusicalGenre['genres']#Aplicando a coluna genres em projection
print(projection)

fig = px.scatter(projection, x='x', y='y', color='cluster_PCA', hover_data=['x', 'y', 'genres'])#hover_data=['x', 'y', 'genres'] informa qual o x, y e gênero o ponto pertence
#fig.show()

#Avaliando o cluster
print(pcaPipeline[1].explained_variance_ratio_.sum())#Mostra a taxa de explicando que o PCA esta dando para os dados
print(pcaPipeline[1].explained_variance_.sum())#Mosta o quantas colunas estão sendo explicadas das 11 que tinhamos no início