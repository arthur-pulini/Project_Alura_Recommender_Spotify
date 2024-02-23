import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import spotipy
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import euclidean_distances
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
from skimage import io

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

print(datas['artists'].value_counts())
print(datas['artists_song'].value_counts())
print('.')

ohe = OneHotEncoder(dtype=int)#Ele faz basicamente a mesma coisa que o getDummies, porém ele é aplicado para pipelines

columnsOhe = ohe.fit_transform(datas[['artists']]).toarray()
print(columnsOhe)

datasV2 = datas.drop('artists', axis=1)
datasMusicDummies = pd.concat([datasV2, pd.DataFrame(columnsOhe, columns = ohe.get_feature_names_out(['artists']))], axis=1)
print(datasMusicDummies)
print(datas.shape)
print(datasMusicDummies.shape)

#Aplicando o PCA
pcaPipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=0.7, random_state=SEED))])#Neste caso foi setado a taxa de aprendizado e não o nº de colunas

musicEmbeddingPca = pcaPipeline.fit_transform(datasMusicDummies.drop(['id', 'name', 'artists_song'], axis=1))
projectionM = pd.DataFrame(data = musicEmbeddingPca)
print(pcaPipeline[1].n_components_)

kmeansPcaPipeline = KMeans(n_clusters=50, verbose=False, random_state=SEED)

kmeansPcaPipeline.fit(projectionM)

datas['cluster_PCA'] = kmeansPcaPipeline.predict(projectionM)
projectionM['cluster_PCA'] = kmeansPcaPipeline.predict(projectionM)
projectionM['artist'] = datas['artists']
projectionM['song'] = datas['artists_song']
print(projectionM)

fig = px.scatter_3d(projectionM, x=0, y=1, z=2, color='cluster_PCA', hover_data=[0, 1, 2, 'song'])
#fig.show()

print(pcaPipeline[1].explained_variance_ratio_.sum())
print(pcaPipeline[1].explained_variance_.sum())

#Recomendação da música
songName = 'Ed Sheeran - Shape of You'
cluster = list(projectionM[projectionM['song'] == songName]['cluster_PCA'])[0]
print(cluster)
recommendedSongs = projectionM[projectionM['cluster_PCA'] == cluster][[0, 1, 'song']]
xSong = list(projectionM[projectionM['song'] == songName][0])[0]
ySong = list(projectionM[projectionM['song'] == songName][1])[0]

#Distâncias euclidianas
distances = euclidean_distances(recommendedSongs[[0, 1]], [[xSong, ySong]])
recommendedSongs['id'] = datas['id']
recommendedSongs['distances'] = distances
recommended = recommendedSongs.sort_values('distances').head(10)
print(recommended)

#Configurando o spotipy
scope = "user-library-read playlist-modify-private"
OAuth = SpotifyOAuth(
        scope=scope,         
        redirect_uri='http://localhost:5000/callback',
        client_id = 'a168df4934644d899b576fec4c16c264',
        client_secret = '16eeb4252af54281bf311193b1cb1b99')

client_credentials_manager = SpotifyClientCredentials(client_id = 'a168df4934644d899b576fec4c16c264',client_secret = '16eeb4252af54281bf311193b1cb1b99')
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

#Importando a imagem do álbum
#Achando o Id
songName = 'Ed Sheeran - Shape of You'
id = datas[datas['artists_song'] == songName]['id'].iloc[0]

#Fazendo a requisição na API
track = sp.track(id)
url = track["album"]["images"][1]["url"]
name = track["name"]

image = io.imread(url)
plt.imshow(image)
plt.xlabel(name, fontsize = 10)
#plt.show()

#Buscando os dados da playlist
def recommend_id(playlist_id): 
    url = []
    name = []
    for i in playlist_id:
        track = sp.track(i)
        url.append(track["album"]["images"][1]["url"])
        name.append(track["name"])
    return name, url

name, url = recommend_id(recommended['id'])
print(name, url)

def visualizeSongs (name, url):
    plt.figure(figsize=(15,10))
    columns = 5
    for i, u in enumerate(url):
        # define o ax como o subplot, com a divisão que retorna inteiro do número urls pelas colunas + 1 (no caso, 6)
        ax = plt.subplot(len(url) // columns + 1, columns, i + 1)

        # Lendo a imagem com o Scikit Image
        image = io.imread(u)

        # Mostra a imagem
        plt.imshow(image)

        # Para deixar o eixo Y invisível 
        ax.get_yaxis().set_visible(False)

        # xticks define o local que vamos trocar os rótulos do eixo x, nesse caso, deixar os pontos de marcação brancos
        plt.xticks(color = 'w', fontsize = 0.1)

        # yticks define o local que vamos trocar os rótulos do eixo y, nesse caso, deixar os pontos de marcação brancos
        plt.yticks(color = 'w', fontsize = 0.1)

        # Colocando o nome da música no eixo x
        plt.xlabel(name[i], fontsize = 8)

        # Faz com que todos os parâmetros se encaixem no tamanho da imagem definido
        plt.tight_layout(h_pad=0.7, w_pad=0)

        # Ajusta os parâmetros de layout da imagem.
        # wspace = A largura do preenchimento entre subparcelas, como uma fração da largura média dos eixos.
        # hspace = A altura do preenchimento entre subparcelas, como uma fração da altura média dos eixos.
        plt.subplots_adjust(wspace=0.05, hspace=0.08)

        # Remove os ticks - marcadores, do eixo x, sem remover o eixo todo, deixando o nome da música.
        plt.tick_params(bottom = False)

        # Tirar a grade da imagem, gerada automaticamente pelo matplotlib
        plt.grid(visible=None, which='major', axis='y')
    #plt.show()

visualizeSongs(name, url)