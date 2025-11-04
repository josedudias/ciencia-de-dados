"""Dashboard Streamlit sobre educação brasileira com dados do INEP e API do X."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import plotly.express as px
import requests
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

st.set_page_config(
    page_title="Grupo 6 - Dashboard Educação",
    layout="wide",
)

st.title("Dashboard Interativo sobre Educação Brasileira")
st.caption("Grupo 6 – Atividade 06 • Integração Censo Escolar INEP + X (Twitter)")

# Constantes principais para configuração do app.
X_API_ENDPOINT = "https://api.x.com/2/tweets/search/recent"
INEP_COLUMNS = [
    "NO_ENTIDADE",
    "SG_UF",
    "NO_MUNICIPIO",
    "TP_DEPENDENCIA",
    "IN_BIBLIOTECA",
    "IN_INTERNET",
    "IN_LABORATORIO_INFORMATICA",
    "IN_QUADRA_ESPORTES",
    "QT_MAT_FUND",
    "QT_MAT_MED",
    "QT_MAT_BAS",
]
DEPENDENCIA_MAP = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}


# Persistência simples para manter o dataframe carregado entre interações.
if "df_censo" not in st.session_state:
    st.session_state["df_censo"] = None

if "censo_encoding" not in st.session_state:
    st.session_state["censo_encoding"] = "latin1"


def _read_credential(env_name: str, secret_key: str) -> str:
    """Return credential value from env vars or Streamlit secrets."""
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return env_value

    try:  # st.secrets may not be configured outside Streamlit runtime
        secret_value = str(st.secrets.get(secret_key, "")).strip()
    except Exception:  # noqa: BLE001 - st.secrets may not be available
        secret_value = ""
    return secret_value


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida e padroniza as colunas relevantes do INEP."""
    missing = [col for col in INEP_COLUMNS if col not in frame.columns]
    if missing:
        raise KeyError(
            "Colunas ausentes no CSV: "
            + ", ".join(missing)
            + "\nConsulte o dicionário de dados do INEP e ajuste o arquivo."
        )

    df = frame[INEP_COLUMNS].copy()

    infra_cols = [
        "IN_BIBLIOTECA",
        "IN_INTERNET",
        "IN_LABORATORIO_INFORMATICA",
        "IN_QUADRA_ESPORTES",
    ]
    for col in infra_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    matr_cols = ["QT_MAT_FUND", "QT_MAT_MED", "QT_MAT_BAS"]
    for col in matr_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["TP_DEPENDENCIA"] = (
        pd.to_numeric(df["TP_DEPENDENCIA"], errors="coerce")
        .map(DEPENDENCIA_MAP)
        .fillna("Não Informado")
    )

    return df


@st.cache_data(show_spinner=False)
def load_csv(file: Optional[object], encoding: str, delimiter: str) -> pd.DataFrame:
    """Carrega o CSV informado e devolve apenas as colunas de interesse."""
    if file is None:
        raise FileNotFoundError("Nenhum arquivo CSV foi fornecido.")

    if isinstance(file, (str, os.PathLike)):
        df = pd.read_csv(file, sep=delimiter, encoding=encoding, low_memory=False)
    else:
        df = pd.read_csv(file, sep=delimiter, encoding=encoding, low_memory=False)
    return _normalise_columns(df)


def fetch_tweets(token: str, query: str, max_results: int) -> pd.DataFrame:
    """Busca tweets recentes utilizando o endpoint search da API do X."""
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "Grupo6Ativ06/1.0"}
    params = {
        "query": query,
        "tweet.fields": "public_metrics,lang,created_at",
        "max_results": max(10, min(max_results, 100)),
    }

    response = requests.get(X_API_ENDPOINT, headers=headers, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Falha na requisição ({response.status_code}): {response.text}"
        )

    tweets = response.json().get("data", [])
    rows = []
    for tweet in tweets:
        metrics = tweet.get("public_metrics", {})
        rows.append(
            {
                "id": tweet.get("id"),
                "created_at": tweet.get("created_at"),
                "text": tweet.get("text"),
                "likes": metrics.get("like_count", 0),
                "retweets": metrics.get("retweet_count", 0),
                "replies": metrics.get("reply_count", 0),
            }
        )

    return pd.DataFrame(rows)


def sentimento_regra(texto: str) -> str:
    """Classifica sentimento de forma simplificada via palavras-chave."""
    texto = texto.lower()
    positivos = [
        "excelente",
        "melhoria",
        "sucesso",
        "qualidade",
        "parabéns",
        "conquista",
        "avanço",
        "orgulho",
        "valorização",
    ]
    negativos = [
        "precariedade",
        "falta",
        "abandono",
        "crise",
        "sucateamento",
        "problema",
        "carência",
        "dificuldade",
        "descaso",
    ]

    score_pos = sum(word in texto for word in positivos)
    score_neg = sum(word in texto for word in negativos)
    if score_pos > score_neg:
        return "positivo"
    if score_neg > score_pos:
        return "negativo"
    return "neutro"


def plot_matriculas(df: pd.DataFrame) -> None:
    """Exibe gráfico de matrículas totais por UF por etapa de ensino."""
    df_uf = (
        df.groupby("SG_UF")[["QT_MAT_FUND", "QT_MAT_MED", "QT_MAT_BAS"]]
        .sum()
        .sort_values("QT_MAT_BAS", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    df_uf.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Total de Matrículas por Estado (UF)")
    ax.set_ylabel("Matrículas")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig)


def plot_infra(df: pd.DataFrame) -> None:
    """Mostra mapa de calor com a infraestrutura média disponível por UF."""
    infra_cols = [
        "IN_BIBLIOTECA",
        "IN_INTERNET",
        "IN_LABORATORIO_INFORMATICA",
        "IN_QUADRA_ESPORTES",
    ]
    infra_uf = df.groupby("SG_UF")[infra_cols].mean().sort_index()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        infra_uf,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Proporção Média de Infraestrutura por Estado")
    ax.set_ylabel("Estado (UF)")
    st.pyplot(fig)


def plot_dependencia(df: pd.DataFrame) -> None:
    """Soma matrículas e apresenta distribuição por dependência administrativa."""
    mat_dep = df.groupby("TP_DEPENDENCIA")["QT_MAT_BAS"].sum().reset_index()
    fig = px.bar(
        mat_dep,
        x="TP_DEPENDENCIA",
        y="QT_MAT_BAS",
        color="TP_DEPENDENCIA",
        title="Total de Matrículas por Dependência Administrativa (INEP)",
        labels={"QT_MAT_BAS": "Total de matrículas"},
    )
    st.plotly_chart(fig, use_container_width=True)


st.sidebar.header("Configurações do Censo Escolar")
with st.sidebar.expander("Seleção do CSV", expanded=True):
    # O usuário pode enviar arquivo ou informar caminho local.
    uploaded = st.file_uploader(
        "Envie o arquivo CSV do INEP",
        type=["csv"],
        help="Utilize os microdados do Censo Escolar (delimitador ';').",
    )
    csv_path = st.text_input(
        "Ou informe o caminho completo do arquivo",
        value="",
        placeholder="Ex.: C:/dados/microdados_ed_basica_2024.csv",
    )
    delimiter = st.text_input("Delimitador", value=";", max_chars=1)
    encoding = st.selectbox("Encoding", ["latin1", "utf-8", "utf-8-sig"], index=0)
    carregar_censo = st.button("Carregar dados do Censo", type="primary")

entrada_csv = uploaded if uploaded is not None else (csv_path or None)

if carregar_censo:
    try:
        # Persistimos dados e encoding para uso após outras interações.
        df_censo = load_csv(entrada_csv, encoding, delimiter)
        st.session_state["df_censo"] = df_censo
        st.session_state["censo_encoding"] = encoding
        st.success(
            f"Dados carregados! Linhas: {len(df_censo):,} | Colunas: {len(df_censo.columns)}"
        )
    except Exception as err:  # noqa: BLE001
        st.session_state["df_censo"] = None
        st.error(str(err))

df_censo = st.session_state.get("df_censo")
encoding_atual = st.session_state.get("censo_encoding", encoding)

if df_censo is not None:
    # Visão geral dos dados carregados.
    st.subheader("Visão Geral do Censo Escolar")
    st.dataframe(df_censo.head(20))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de matrículas", f"{int(df_censo['QT_MAT_BAS'].sum()):,}")
    with col2:
        st.metric("Escolas com Internet", f"{int(df_censo['IN_INTERNET'].sum()):,}")
    with col3:
        st.metric(
            "% escolas com laboratório",
            f"{df_censo['IN_LABORATORIO_INFORMATICA'].mean() * 100:.1f}%",
        )

    st.markdown("### Principais gráficos")
    plot_matriculas(df_censo)
    plot_infra(df_censo)
    plot_dependencia(df_censo)

    with st.expander("Filtrar por município"):
        # Seleciona município e permite baixar o subconjunto correspondente.
        municipio = st.selectbox(
            "Município", df_censo["NO_MUNICIPIO"].sort_values().unique()
        )
        df_city = df_censo[df_censo["NO_MUNICIPIO"] == municipio]
        st.write(f"Escolas encontradas em **{municipio}**: {len(df_city)}")
        st.dataframe(df_city)
        st.download_button(
            "Baixar CSV filtrado",
            df_city.to_csv(index=False).encode(encoding_atual),
            file_name="censo_filtrado_grupo6.csv",
            mime="text/csv",
        )
else:
    st.info("Carregue os dados do Censo para visualizar indicadores e filtros.")


st.sidebar.header("Configurações do X (Twitter)")
with st.sidebar.expander("Consulta de tweets", expanded=False):
    # Token vem de variável de ambiente ou de st.secrets.
    default_token = _read_credential("X_BEARER_TOKEN", "x_bearer_token")
    token = st.text_input(
        "Token Bearer",
        value=default_token,
        type="password",
        help=(
            "Defina o token em uma variável de ambiente X_BEARER_TOKEN ou em"
            " st.secrets['x_bearer_token']."
        ),
    )
    query = st.text_input(
        "Palavras-chave",
        value="educação OR escola OR professor OR ensino",
    )
    quantidade = st.slider("Quantidade de tweets", 10, 100, 30, step=10)
    buscar_tweets = st.button("Buscar tweets", key="buscar_tweets")

if buscar_tweets:
    if not token:
        st.warning("Informe um token válido da API do X.")
    else:
        try:
            # Realiza consulta, aplica sentimento básico e exibe resultados.
            tweets_df = fetch_tweets(token, query, quantidade)
            if tweets_df.empty:
                st.info("Nenhum tweet retornado para a consulta.")
            else:
                tweets_df["sentimento"] = tweets_df["text"].apply(sentimento_regra)
                st.subheader("Tweets coletados")
                st.dataframe(tweets_df)

                contagem = tweets_df["sentimento"].value_counts().reset_index()
                contagem.columns = ["sentimento", "quantidade"]
                grafico = px.bar(
                    contagem,
                    x="sentimento",
                    y="quantidade",
                    color="sentimento",
                    title="Sentimentos sobre Educação no X",
                )
                st.plotly_chart(grafico, use_container_width=True)
        except Exception as err:  # noqa: BLE001
            st.error(f"Erro ao consultar a API do X: {err}")

st.markdown(
    "---\n"
    "Integrantes Grupo 6: Gabriel Storti Segalla, Luis Felipe Mozar Chiqueto, "
    "João Pedro Rosa de Paula, José Eduardo Rufino Dias, Igor Lima Ponce."
)
