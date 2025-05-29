# umap



datos = pd.read_csv('../../.././../datos/EjemploEstudiantes.csv', delimiter = ';', decimal = ",", index_col = 0) datos
escalar = StandardScaler() datos_escalados = escalar.fit_transform(datos) datos_escalados = pd.DataFrame(datos_escalados) datos_escalados.columns = datos.columns datos_escalados.index = datos.index datos_escalados


umap = UMAP(n_components = 2, n_neighbors = 2) individuos = umap.fit_transform(datos_escalados) individuos = pd.DataFrame(individuos, index=datos_escalados.index) individuos
x = individuos.iloc[:, 0] y = individuos.iloc[:, 1]
fig, ax = plt.subplots(figsize = (10, 6))
no_print = ax.scatter(x, y, color = 'steelblue')
no_print = ax.axhline(y = 0, color = 'dimgrey', linestyle = '--') no_print = ax.axvline(x = 0, color = 'dimgrey', linestyle = '--') no_print = ax.set_xlabel('Componente 1') no_print = ax.set_ylabel('Componente 2')
for i in range(individuos.shape[0]): no_print = ax.annotate(individuos.index[i], (x[i], y[i]))
plt.show()
tsne = TSNE(n_components=2, perplexity=2, learning_rate='auto', init='random') individuos = tsne.fit_transform(datos_escalados) individuos = pd.DataFrame(individuos, index=datos_escalados.index) individuos
x = individuos.iloc[:, 0] y = individuos.iloc[:, 1]
fig, ax = plt.subplots(figsize = (10, 6))
no_print = ax.scatter(x, y, color = 'steelblue')
no_print = ax.axhline(y = 0, color = 'dimgrey', linestyle = '--') no_print = ax.axvline(x = 0, color = 'dimgrey', linestyle = '--') no_print = ax.set_xlabel('Componente 1') no_print = ax.set_ylabel('Componente 2')
for i in range(individuos.shape[0]): no_print = ax.annotate(individuos.index[i], (x[i], y[i]))
plt.show()

