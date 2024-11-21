import pandas as pd

# Carregar o arquivo CSV em um DataFrame
file_path = 'D:\\AG2\\AG2\\Wholesale customers data.csv'
data = pd.read_csv(file_path)

# Contar quantos valores existem em cada classe na coluna 'Channel'
channel_counts = data['Channel'].value_counts()

# Exibir o resultado
print(channel_counts)
