# Dicionário de Dados: Grade de Centróides (2003-2013)
## Modelagem de Risco de Incêndio Florestal por Lógica Fuzzy no Estado do Acre

Este dicionário de dados descreve as variáveis contidas no arquivo `centroides2003a2013.csv`, que serve como base de dados espacial para o Modelo de Risco de Incêndio Florestal (Longo Prazo) desenvolvido no projeto.

O conjunto de dados discretiza o território do Estado do Acre por meio de uma grade regular de células. Cada linha do arquivo representa o centróide geométrico de uma célula e contém atributos geográficos, ambientais e registros históricos de queimadas. No modelo, essas variáveis alimentam um classificador kNN Fuzzy (k-vizinhos mais próximos fuzzy) para estimar a pertinência de cada célula ao conjunto de alto risco de incêndio.

---

### Tabela de Atributos (Campos do CSV)

| Campo | Tipo no CSV | Descrição Detalhada | Exemplo de Valor | Papel no Modelo |
| :--- | :--- | :--- | :--- | :--- |
| **Y** | Inteiro | Coordenada de linha da célula na grade de discretização espacial que recobre o Estado do Acre. | `280` | **Índice Espacial** (Não preditor) |
| **X** | Inteiro | Coordenada de coluna da célula na grade de discretização espacial do Acre. | `886` | **Índice Espacial** (Não preditor) |
| **x_g** | Numérico (Float) | Coordenada geográfica X projetada no sistema **UTM (Easting)**, expressa em metros, referente ao centróide de cada célula. | `699749,0368` | **Divisor Espacial**: Usada para dividir o Acre em duas regiões para modelagem independente: Esquerda (Oeste, se $x_g < 337100$ m) e Direita (Leste, se $x_g \ge 337100$ m). |
| **y_g2** | Numérico (Float) | Coordenada geográfica Y projetada no sistema **UTM (Northing)**, expressa em metros, referente ao centróide de cada célula. | `8858182,6636` | **Mapeamento**: Usada exclusivamente para localização e plotagem espacial dos resultados (não entra como variável preditora) |
| **VEG_TIP** | Texto (String) | Variável categórica que descreve a **Tipologia Florestal ou de Cobertura Vegetal** associada ao centróide de cada célula. | `Áreas Antropizadas` | **Variável Preditora (Entrada)**: Mapeada para uma escala numérica ordenada com base na densidade histórica de queimadas por unidade de área de cada tipologia florestal. |
| **Distance** | Numérico (Float) | Distância física em metros do centróide da célula até a estrada de interesse mais próxima. | `666,1694` | **Variável Preditora (Entrada)**: Representa a influência da infraestrutura humana ou barreiras fluviais. *Nota: Para respeitar os limites de não assunção de dados, a associação explícita das colunas Distance e Distance_1 a estradas ou rios é mantida aberta.* |
| **Distance_1** | Numérico (Float) | Distância física em metros do centróide da célula até o curso d'agua de interesse mais próximo. | `964,6656` | **Variável Preditora (Entrada)**: Atua de forma simétrica ao campo `Distance` no classificador kNN Fuzzy, mapeando o vetor espacial de pressões de ignição do fogo. |
| **RASTERVALU** | Numérico (Float) | **Altitude/Elevação** em metros associada ao centróide da célula, obtida por meio do cruzamento com dados do satélite altimétrico (projeto Brasil em Relevo - Embrapa). | `129` | **Variável Preditora (Entrada)**: A altitude média histórica em que ocorrem focos de calor no Acre é de ~202,65m (com desvio padrão de 20,94m), sendo uma variável preditora crucial. |
| **Count__1** | Inteiro | Contagem de focos de calor (queimadas) registrados por satélite na célula em um período histórico preliminar. | `0` | **Descartada**: Não entra no treinamento do kNN Fuzzy para o ano alvo (evita vazamento de dados), servindo apenas para controle histórico. |
| **Count_2** | Inteiro | Contagem de focos de calor registrados por satélite na célula em outro recorte temporal do monitoramento INPE. | `1` | **Descartada**: Mantida no banco de dados apenas para inventário estatístico e comparações geoespaciais secundárias. |
| **Count2010** | Inteiro | Contagem de focos de calor (queimadas) registrados na célula pelo satélite de referência do INPE durante todo o ano de 2010. | `0` | **Variável Alvo (Target - y)**: No pré-processamento do script, é binarizada: se $Count2010 > 0$, assume valor **`1`** (presença de queimada); caso contrário, assume **`0`** (ausência). |

---

### Detalhes de Pré-processamento e Codificação Fuzzy

#### 1. Codificação Numérica de Tipologias de Vegetação (`VEG_TIP`)
O classificador kNN Fuzzy exige dados numéricos para calcular a matriz de covariância e a distância de Mahalanobis. Por isso, a tipologia florestal qualitativa em `VEG_TIP` é convertida pelo código em uma escala numérica (`TipoDic`) ordenada de forma crescente com base na densidade histórica de focos de calor por área. O mapeamento completo do dicionário original do código é o seguinte:

* **1** = `Áreas Antropizadas` (Zonas de forte pressão antrópica e agropecuária, onde a densidade de fogo é máxima por unidade de área)
* **2** = `FAP - Aluvial + Vs` (Floresta Aberta com Palmeiras em área aluvial + Vegetação Secundária)
* **3** = `FAP - Aluvial` (Floresta Aberta com Palmeiras em área aluvial)
* **4** = `FAP + FD` ou `FD + FAP` (Associação de Floresta Aberta com Palmeiras e Floresta Densa)
* **5** = `FAB - Aluvial` (Floresta Aberta com Bambu em área aluvial)
* **6** = `FAP` (Floresta Aberta com Palmeiras)
* **7** = `FAB + FD` ou `FD + FAB` (Associação de Floresta Aberta com Bambu e Floresta Densa)
* **8** = `FAB + FAP` ou `FAP + FAB` (Associação de Floresta Aberta com Bambu e Floresta Aberta com Palmeiras)
* **9** = `FAP - Aluvial + Pab` (Floresta Aberta com Palmeiras aluvial + Formações Pioneiras)
* **10** = `Campinaranas` (Zonas de vegetação amazônica adaptadas a solos arenosos e hidromórficos de alta fragilidade)
* **11** = `FAB + FAP + FD` / `FAP + FAB + FD` / `FAP + FD + FAB` (Associações florestais triplas)
* **12** = `FABD` (Floresta Aberta com Bambu Dominante)
* **13** = `FD` (Floresta Densa - áreas tipicamente mais úmidas e resilientes ao fogo)
* **14** = `FD - Submontana` (Floresta Densa em áreas submontanas)
* **15** = `FAP + Pab` (Associação de Floresta Aberta com Palmeiras e Formações Pioneiras)
* **16** / **NaN** = Valores nulos ou não classificados.
* **0** = Entradas em branco (filtradas durante o pré-tratamento de integridade dos dados).

#### 2. Divisão Territorial do Estado (Oeste vs. Leste)
Como o padrão de ocupação humana, a umidade e a dinâmica de focos de calor diferem substancialmente entre as porções leste e oeste do Estado do Acre, o modelo de risco utiliza a coordenada geográfica de projeção **`x_g`** para segmentar os dados:
* **Lado Esquerdo (Oeste do Estado / Regional do Juruá)**: Centróides com $x_g < 337.100$ metros. O número ideal de vizinhos para classificação nesta metade é $k_e = 21$ vizinhos no kNN Fuzzy.
* **Lado Direito (Leste do Estado / Regional do Baixo Acre e Purus)**: Centróides com $x_g \ge 337.100$ metros. O número ideal de vizinhos para classificação nesta metade é $kd = 29$ (ou $51$ a depender do ajuste fino regional) vizinhos no kNN Fuzzy.

#### 3. Balanceamento e Padronização
Antes de executar o classificador kNN Fuzzy, as quatro variáveis de entrada ($X = [VEG\_TIP, Distance, Distance\_1, RASTERVALU]$) sofrem dois procedimentos essenciais:
* **Balanceamento do Conjunto**: O número de células sem ocorrência de queimadas ($y=0$, classe majoritária) é subamostrado aleatoriamente para se igualar de forma exata ao número de células com presença de queimadas ($y=1$), evitando o viés do algoritmo por desbalanceamento estatístico.
* **Padronização (`StandardScaler`)**: Os valores das variáveis são normalizados para que tenham **média igual a zero e variância igual a um**. Isso é feito porque as amplitudes numéricas diferem muito (a altitude está na casa das centenas, enquanto as distâncias estão na casa dos milhares), impedindo que as variáveis com maiores valores numéricos distorçam o peso de proximidade geométrica no classificador.
#   i n c e n d i o  
 #   i n c e n d i o  
 #   i n c e n d i o  
 # incendio
