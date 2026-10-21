"""
Fundos & Indicadores (dados gratuitos)
======================================

Ferramenta da Plataforma Nextep que usa só fontes públicas e gratuitas:

  • Fundos ............ CVM – Dados Abertos (informe diário + cadastro)
  • CDI, Selic, IPCA,
    IGP-M, Dólar ....... Banco Central – API SGS
  • Ibovespa, ações,
    ETFs, FIIs, EUA .... Yahoo Finance (biblioteca yfinance)

Não precisa de chave nem de secrets.

Como a CVM publica um arquivo por mês com TODOS os fundos, a primeira consulta
de cada mês baixa o arquivo (~10–30 MB) e o converte num arquivo compacto
(parquet) no disco do servidor. As consultas seguintes, de qualquer pessoa,
são instantâneas.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
import time
import unicodedata
import zipfile
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────────────────────────────────
CVM_DIARIO = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{ym}.zip"
CVM_DIARIO_HIST = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/HIST/inf_diario_fi_{ano}.zip"
CVM_CADASTRO = [
    # (url, prioridade: menor = preferido quando o CNPJ aparece em mais de um)
    ("https://dados.cvm.gov.br/dados/FI/CAD/DADOS/registro_fundo_classe.zip", 0),
    ("https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv", 2),
]
BCB_SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"

PASTA = os.path.join(tempfile.gettempdir(), "nextep_dados_publicos")
os.makedirs(PASTA, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Plataforma Nextep)"}

# Séries do Banco Central: código SGS -> (nome, tipo)
#   tipo "taxa_diaria"  = % ao dia (CDI, Selic)  -> vira índice acumulado
#   tipo "taxa_mensal"  = % ao mês (IPCA, IGP-M) -> vira índice acumulado
#   tipo "nivel"        = valor (Dólar PTAX)
BCB = {
    "12": ("CDI", "taxa_diaria"),
    "11": ("Selic", "taxa_diaria"),
    "433": ("IPCA", "taxa_mensal"),
    "189": ("IGP-M", "taxa_mensal"),
    "1": ("Dólar (PTAX venda)", "nivel"),
}
ATALHOS = [
    ("CDI", "bcb:12"), ("Selic", "bcb:11"), ("IPCA", "bcb:433"), ("IGP-M", "bcb:189"),
    ("Dólar", "bcb:1"), ("Ibovespa", "idx:ibov"), ("S&P 500 TR", "idx:sp500tr"), ("MSCI World", "idx:msciworld"),
    ("MSCI ACWI", "idx:msciacwi"), ("CPI + 5%", "inf:cpi:5"), ("IPCA + 5%", "inf:ipca:5"),
]
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


class ErroDados(Exception):
    pass


# ──────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────
def so_digitos(x: str) -> str:
    return re.sub(r"\D", "", str(x or ""))


def fmt_cnpj(c: str) -> str:
    c = so_digitos(c).zfill(14)
    return f"{c[:2]}.{c[2:5]}.{c[5:8]}/{c[8:12]}-{c[12:]}"


def sem_acento(t: str) -> str:
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode()
    return t.lower()


def baixar(url: str, destino: str, timeout: int = 180) -> bool:
    """Baixa para o disco. Retorna False se o arquivo não existe (404)."""
    with requests.get(url, headers=UA, stream=True, timeout=timeout) as r:
        if r.status_code == 404:
            return False
        if r.status_code != 200:
            raise ErroDados(f"Erro {r.status_code} ao baixar {url}")
        tmp = destino + ".part"
        with open(tmp, "wb") as f:
            for bloco in r.iter_content(1 << 20):
                f.write(bloco)
        os.replace(tmp, destino)
    return True


def idade_horas(caminho: str) -> float:
    return (time.time() - os.path.getmtime(caminho)) / 3600 if os.path.exists(caminho) else 1e9


def col(cols: list[str], *tokens: str, excluir: tuple[str, ...] = ()) -> str | None:
    """Primeira coluna cujo nome (maiúsculo) contém todos os tokens."""
    for c in cols:
        u = c.upper()
        if all(t in u for t in tokens) and not any(e in u for e in excluir):
            return c
    return None


def ler_csv_texto(dados: bytes, usecols=None) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(dados), sep=";", dtype="string[pyarrow]", encoding=enc,
                               usecols=usecols, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ErroDados("Não consegui ler o arquivo da CVM (codificação).")


# ──────────────────────────────────────────────────────────────────────────
# CVM – Informe diário (cotas, PL, cotistas)
# ──────────────────────────────────────────────────────────────────────────
def _parquet_mes(ano: int, mes: int) -> str:
    return os.path.join(PASTA, f"cvm_{ano}{mes:02d}.parquet")


def _validade_horas(ano: int, mes: int) -> float:
    hoje = date.today()
    atraso = (hoje.year - ano) * 12 + (hoje.month - mes)
    if atraso <= 1:
        return 12          # mês atual e anterior: CVM atualiza todo dia
    if atraso <= 11:
        return 24 * 7      # até 11 meses: atualização semanal
    return 1e9             # mais antigos: não mudam


def _converter_csv_cvm(fobj, destino: str):
    """Lê o CSV do informe diário (formato antigo ou RCVM 175) e grava parquet compacto."""
    cabecalho = fobj.readline().decode("latin-1").strip().split(";")
    fobj.seek(0)
    c_cnpj = col(cabecalho, "CNPJ")
    c_sub = col(cabecalho, "SUBCLASSE")
    mapa = {
        c_cnpj: "cnpj", "DT_COMPTC": "data", "VL_QUOTA": "cota", "VL_PATRIM_LIQ": "pl",
        "NR_COTST": "cotistas", "CAPTC_DIA": "captacao", "RESG_DIA": "resgate",
    }
    if c_sub:
        mapa[c_sub] = "subclasse"
    usar = [c for c in mapa if c in cabecalho]
    tipos = {c_cnpj: pa.string(), "DT_COMPTC": pa.string()}
    if c_sub:
        tipos[c_sub] = pa.string()
    tab = pacsv.read_csv(
        fobj,
        read_options=pacsv.ReadOptions(encoding="latin1"),
        parse_options=pacsv.ParseOptions(delimiter=";", invalid_row_handler=lambda _: "skip"),
        convert_options=pacsv.ConvertOptions(include_columns=usar, column_types=tipos),
    )
    tab = tab.rename_columns([mapa[c] for c in tab.column_names])
    colunas = {}
    for nome in tab.column_names:
        arr = tab[nome]
        if nome == "cnpj":
            arr = pc.replace_substring_regex(arr, r"\D", "")
        elif nome == "data":
            arr = pc.strptime(arr, format="%Y-%m-%d", unit="s").cast(pa.date32())
        elif nome != "subclasse" and pa.types.is_string(arr.type):  # decimal com vírgula
            arr = pc.cast(pc.replace_substring(arr, ",", "."), pa.float64())
        elif nome != "subclasse":
            arr = pc.cast(arr, pa.float64())
        colunas[nome] = arr
    if "subclasse" not in colunas:
        colunas["subclasse"] = pa.nulls(tab.num_rows, pa.string())
    pq.write_table(pa.table(colunas), destino, compression="zstd")


def garantir_mes(ano: int, mes: int) -> str | None:
    """Garante o parquet do mês em disco; baixa da CVM se preciso. None = mês não publicado."""
    destino = _parquet_mes(ano, mes)
    if os.path.exists(destino) and idade_horas(destino) < _validade_horas(ano, mes):
        return destino

    if ano >= 2021:
        zip_path = os.path.join(PASTA, f"tmp_{ano}{mes:02d}.zip")
        try:
            ok = baixar(CVM_DIARIO.format(ym=f"{ano}{mes:02d}"), zip_path)
        except (requests.RequestException, ErroDados):
            return destino if os.path.exists(destino) else None  # usa a versão antiga
        if not ok:
            return destino if os.path.exists(destino) else None
        with zipfile.ZipFile(zip_path) as z:
            nome = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            with z.open(nome) as f:
                _converter_csv_cvm(io.BytesIO(f.read()), destino)
        os.remove(zip_path)
        return destino

    # Antes de 2021: arquivo anual com um CSV por mês
    zip_path = os.path.join(PASTA, f"tmp_{ano}.zip")
    if not baixar(CVM_DIARIO_HIST.format(ano=ano), zip_path):
        return None
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            m = re.search(r"(\d{4})(\d{2})\.csv$", n)
            if m:
                with z.open(n) as f:
                    _converter_csv_cvm(io.BytesIO(f.read()), _parquet_mes(int(m[1]), int(m[2])))
    os.remove(zip_path)
    return destino if os.path.exists(destino) else None


def meses_entre(inicio: date, fim: date) -> list[tuple[int, int]]:
    out, a, m = [], inicio.year, inicio.month
    while (a, m) <= (fim.year, fim.month):
        out.append((a, m))
        a, m = (a + 1, 1) if m == 12 else (a, m + 1)
    return out


def preparar_meses_cvm(inicio: date, fim: date):
    """Baixa (com barra de progresso) os meses que ainda não estão no disco."""
    faltando = [
        (a, m) for a, m in meses_entre(inicio, fim)
        if not (os.path.exists(_parquet_mes(a, m)) and idade_horas(_parquet_mes(a, m)) < _validade_horas(a, m))
    ]
    if not faltando:
        return
    barra = st.progress(0.0, text="Baixando informes da CVM (só na primeira vez)…")
    for k, (a, m) in enumerate(faltando, 1):
        barra.progress(k / len(faltando), text=f"Baixando informe da CVM de {m:02d}/{a} ({k}/{len(faltando)})…")
        garantir_mes(a, m)
    barra.empty()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def serie_fundo(cnpj: str, inicio: date, fim: date) -> pd.DataFrame:
    partes = []
    for a, m in meses_entre(inicio, fim):
        caminho = garantir_mes(a, m)
        if caminho:
            partes.append(pq.read_table(caminho, filters=[("cnpj", "=", cnpj)]).to_pandas())
    if not partes:
        return pd.DataFrame(columns=["valor", "pl", "cotistas", "captacao", "resgate"])
    df = pd.concat(partes, ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=["valor", "pl", "cotistas", "captacao", "resgate"])

    # Classes com subclasses: usa a linha sem subclasse; se não houver, a subclasse de maior PL.
    df["subclasse"] = df["subclasse"].fillna("").str.strip()
    if (df["subclasse"] == "").any():
        df = df[df["subclasse"] == ""]
    else:
        ultima = df[df["data"] == df["data"].max()]
        df = df[df["subclasse"] == ultima.loc[ultima["pl"].fillna(0).idxmax(), "subclasse"]]

    df["data"] = pd.to_datetime(df["data"])
    df = df.drop_duplicates("data", keep="last").set_index("data").sort_index()
    df = df.loc[pd.Timestamp(inicio): pd.Timestamp(fim)]
    df = df[df["cota"] > 0]
    return df.rename(columns={"cota": "valor"})[["valor", "pl", "cotistas", "captacao", "resgate"]]


# ──────────────────────────────────────────────────────────────────────────
# CVM – Cadastro (nome, situação, classificação…)
# ──────────────────────────────────────────────────────────────────────────
PALAVRAS_FICHA = ("CNPJ", "DENOM", "SIT", "CLASS", "TIPO", "RENTAB", "INDICADOR", "ANBIMA", "GESTOR",
                  "ADMIN", "TRIB", "PUBLICO", "CONDOM", "EXCLUSIVO", "DT_INI", "DATA_INI", "TAXA", "PATRIM")


def _tabela_cadastro(dados: bytes, prioridade: int, fonte: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    df = ler_csv_texto(dados, usecols=lambda c: any(p in c.upper() for p in PALAVRAS_FICHA))
    cols = list(df.columns)
    c_cnpj = (col(cols, "CNPJ", "CLASSE") or col(cols, "CNPJ", "FUNDO") or col(cols, "CNPJ")) if cols else None
    c_nome = col(cols, "DENOM")
    if not c_cnpj or not c_nome:
        return None
    c_sit = col(cols, "SITUA") or col(cols, "SIT")
    out = pd.DataFrame({
        "cnpj": df[c_cnpj].str.replace(r"\D", "", regex=True).astype(str),
        "nome": df[c_nome].fillna("").str.strip(),
        "situacao": df[c_sit].fillna("") if c_sit else "",
        "prioridade": prioridade,
        "fonte": fonte,
        "linha": np.arange(len(df)),
    })
    return out[out["cnpj"].str.len() == 14], df


@st.cache_resource(ttl=24 * 3600, show_spinner="Carregando cadastro de fundos da CVM…")
def _cadastro_completo() -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Tabela CNPJ → nome/situação, juntando o cadastro novo (RCVM 175) e o antigo.
    Também devolve as tabelas originais, usadas na ficha cadastral."""
    tabelas, brutos = [], []

    def guardar(t):
        if t is not None:
            tabelas.append(t[0])
            brutos.append(t[1])

    for url, prio in CVM_CADASTRO:
        destino = os.path.join(PASTA, os.path.basename(url))
        try:
            if idade_horas(destino) > 24:
                baixar(url, destino)
        except (requests.RequestException, ErroDados):
            pass
        if not os.path.exists(destino):
            continue
        if destino.endswith(".zip"):
            with zipfile.ZipFile(destino) as z:
                for n in z.namelist():
                    if n.lower().endswith(".csv") and "subclasse" not in n.lower():
                        # classe antes de fundo: o informe diário usa o CNPJ da classe
                        p = prio if "classe" in n.lower() else prio + 1
                        guardar(_tabela_cadastro(z.read(n), p, len(brutos)))
        else:
            with open(destino, "rb") as f:
                guardar(_tabela_cadastro(f.read(), prio, len(brutos)))
    if not tabelas:
        return pd.DataFrame(columns=["nome", "situacao", "fonte", "linha", "busca", "ativo"]), []
    cad = pd.concat(tabelas, ignore_index=True)
    cad = cad.sort_values("prioridade").drop_duplicates("cnpj").set_index("cnpj")
    cad["busca"] = cad["nome"].map(sem_acento)
    cad["ativo"] = cad["situacao"].map(sem_acento).str.contains("funcionamento|ativ", regex=True)
    return cad, brutos


def cadastro() -> pd.DataFrame:
    return _cadastro_completo()[0]


def ficha(cnpj: str) -> dict | None:
    cad, brutos = _cadastro_completo()
    if cnpj not in cad.index:
        return None
    r = cad.loc[cnpj]
    return brutos[int(r["fonte"])].iloc[int(r["linha"])].fillna("").to_dict()


def buscar_fundos(termo: str, limite: int = 30) -> pd.DataFrame:
    cad = cadastro()
    if cad.empty:
        return cad
    dig = so_digitos(termo)
    if len(dig) >= 6 and not re.sub(r"[\d./\-\s]", "", termo):  # digitou um CNPJ (ou começo dele)
        res = cad[cad.index.str.startswith(dig)]
    else:
        mask = pd.Series(True, index=cad.index)
        for palavra in sem_acento(termo).split():
            mask &= cad["busca"].str.contains(palavra, regex=False)
        res = cad[mask]
    return res.sort_values(["ativo", "nome"], ascending=[False, True]).head(limite)


# ──────────────────────────────────────────────────────────────────────────
# Banco Central – SGS
# ──────────────────────────────────────────────────────────────────────────
def _sgs_bloco(codigo: str, nome: str, a: date, b: date) -> list:
    """Uma consulta à API SGS, com até 3 tentativas (a API às vezes devolve página vazia ou HTML)."""
    ultimo = ""
    for tentativa in range(3):
        if tentativa:
            time.sleep(1.5 * tentativa)
        try:
            r = requests.get(
                BCB_SGS.format(codigo=codigo),
                params={"formato": "json", "dataInicial": f"{a:%d/%m/%Y}", "dataFinal": f"{b:%d/%m/%Y}"},
                headers={**UA, "Accept": "application/json"}, timeout=45,
            )
        except requests.RequestException as e:
            ultimo = type(e).__name__
            continue
        if r.status_code == 404:  # sem dados no intervalo
            return []
        if r.status_code != 200:
            ultimo = f"erro {r.status_code}"
            continue
        try:
            dados = r.json()
        except ValueError:
            ultimo = "resposta inválida"
            continue
        if isinstance(dados, list):
            return dados
        ultimo = str(dados)[:80]
    if (b - a).days < 20:  # janela curtinha (ex.: só feriados): trata como sem dados
        return []
    raise ErroDados(f"Banco Central não respondeu para {nome} ({ultimo}).")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def serie_bcb(codigo: str, inicio: date, fim: date) -> pd.DataFrame:
    nome, tipo = BCB[codigo]
    # mensal: pega o mês anterior também, para ter o ponto-base
    ini = (inicio.replace(day=1) - timedelta(days=1)).replace(day=1) if tipo == "taxa_mensal" else inicio
    partes, a = [], ini
    while a <= fim:  # a API limita o tamanho da janela: busca em blocos de 2 anos
        b = min(fim, (pd.Timestamp(a) + pd.DateOffset(years=2) - pd.Timedelta(days=1)).date())
        partes += _sgs_bloco(codigo, nome, a, b)
        a = b + timedelta(days=1)
    if not partes:
        return pd.DataFrame(columns=["valor"])
    s = pd.DataFrame(partes)
    s["data"] = pd.to_datetime(s["data"], format="%d/%m/%Y")
    s["v"] = pd.to_numeric(s["valor"].astype(str).str.replace(",", "."), errors="coerce")
    s = s.dropna().drop_duplicates("data").set_index("data")["v"].sort_index()

    if tipo == "taxa_diaria":
        # base 1 no próprio dia de início; acumula as taxas dos dias seguintes
        # (convenção de mercado: CDI "no ano" começa a render em 02/01, com base em 31/12)
        s = s[s.index > pd.Timestamp(inicio)]
        if s.empty:
            return pd.DataFrame(columns=["valor"])
        indice = (1 + s / 100).cumprod()
        indice = pd.concat([pd.Series([1.0], index=[pd.Timestamp(inicio)]), indice])
    elif tipo == "taxa_mensal":
        # taxa do mês M fica registrada no fim do mês M; base 1 no fim do mês anterior
        s.index = s.index + pd.offsets.MonthEnd(0)
        indice = (1 + s / 100).cumprod()
        indice = pd.concat([pd.Series([1.0], index=[indice.index[0] - pd.offsets.MonthEnd(1)]), indice])
        base = pd.offsets.MonthEnd().rollback(pd.Timestamp(inicio))  # último fim de mês ≤ início
        indice = indice[indice.index >= base]
    else:
        indice = s
    return indice.to_frame("valor")


# ──────────────────────────────────────────────────────────────────────────
# Yahoo Finance
# ──────────────────────────────────────────────────────────────────────────
def ticker_yahoo(t: str) -> str:
    t = t.strip().upper()
    if re.fullmatch(r"[A-Z]{4}\d{1,2}[A-Z]?", t):  # padrão B3: PETR4, BOVA11, TAEE11...
        return t + ".SA"
    return t


def _yahoo_direto(ticker: str, inicio: date, fim: date) -> pd.Series:
    """Fechamento ajustado (com proventos) direto da API de gráficos do Yahoo, sem a biblioteca yfinance."""
    p1 = int(pd.Timestamp(inicio).timestamp())
    p2 = int((pd.Timestamp(fim) + pd.Timedelta(days=1)).timestamp())
    ultimo = ""
    for host in ("query1", "query2"):
        try:
            r = requests.get(
                f"https://{host}.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(ticker)}",
                params={"period1": p1, "period2": p2, "interval": "1d", "events": "div,splits",
                        "includeAdjustedClose": "true"},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"},
                timeout=30,
            )
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            ts = pd.to_datetime(np.asarray(res["timestamp"]) + res["meta"].get("gmtoffset", 0), unit="s").normalize()
            ind = res["indicators"]
            valores = (ind.get("adjclose") or [{}])[0].get("adjclose") or ind["quote"][0]["close"]
            s = pd.Series(pd.to_numeric(pd.Series(valores), errors="coerce").values, index=ts)
            s = s[~s.index.duplicated(keep="last")].dropna()
            if len(s):
                return s
            ultimo = "sem dados"
        except Exception as e:  # noqa: BLE001
            ultimo = type(e).__name__
    raise ErroDados(f"Yahoo direto: {ultimo}")


def _serie_yf(ticker: str, inicio: date, fim: date) -> pd.Series:
    """Yahoo: tenta a API direta; se falhar e a biblioteca yfinance estiver instalada, usa ela."""
    try:
        return _yahoo_direto(ticker, inicio, fim).loc[pd.Timestamp(inicio): pd.Timestamp(fim)]
    except ErroDados as e:
        erro_direto = str(e)
    try:
        import yfinance as yf
    except ImportError:
        raise ErroDados(erro_direto) from None
    h = yf.Ticker(ticker).history(start=inicio, end=fim + timedelta(days=1), auto_adjust=True)
    if h is None or h.empty:
        raise ErroDados(f"{erro_direto}; yfinance sem dados para {ticker}")
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
    return h["Close"].dropna()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def serie_yahoo(ticker: str, inicio: date, fim: date) -> tuple[str, pd.DataFrame]:
    try:
        s = _serie_yf(ticker, inicio, fim)
    except ErroDados as e:
        raise ErroDados(f"Yahoo Finance não retornou dados para {ticker} ({e}). "
                        "Confira o ticker ou tente de novo em alguns minutos.") from None
    return ticker, s.to_frame("valor")


# ──────────────────────────────────────────────────────────────────────────
# Índices de mercado (com fontes reserva) e inflação + spread
# ──────────────────────────────────────────────────────────────────────────
B3_INDICE = "https://sistemaswebb3-listados.b3.com.br/indexStatisticsProxy/IndexCall/GetPortfolioDay/{payload}"
MSCI_URL = "https://app2.msci.com/products/service/index/indexmaster/getLevelDataForGraph"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# código -> (nome, moeda, [fontes em ordem de preferência])
INDICES = {
    "ibov": ("Ibovespa", "BRL", [("b3", "IBOV"), ("yf", "^BVSP"), ("yf", "BOVA11.SA")]),
    "sp500tr": ("S&P 500 TR (USD)", "USD", [("yf", "^SP500TR"), ("yf", "SPY"), ("yf", "IVV")]),
    "msciworld": ("MSCI World NR (USD)", "USD", [("msci", "990100:NETR"), ("yf", "URTH")]),
    "msciacwi": ("MSCI ACWI NR (USD)", "USD", [("msci", "892400:NETR"), ("yf", "ACWI")]),
}
NOME_FONTE = {"b3": "B3", "yf": "Yahoo Finance", "msci": "MSCI", "bcb": "Banco Central", "fred": "FRED (Fed St. Louis)"}
FONTES_USADAS: dict[str, str] = {}


def _num_br(v) -> float:
    """'134.185,24' -> 134185.24 ; '134185.24' -> 134185.24"""
    s = str(v or "").strip()
    if not s:
        return np.nan
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _serie_b3(indice: str, inicio: date, fim: date) -> pd.Series:
    import base64
    import json

    pontos = {}
    for ano in range(inicio.year, fim.year + 1):
        payload = base64.b64encode(
            json.dumps({"index": indice, "language": "pt-br", "year": str(ano)}).encode()
        ).decode()
        r = requests.get(B3_INDICE.format(payload=payload), headers=UA, timeout=60)
        r.raise_for_status()
        for linha in (r.json() or {}).get("results") or []:
            dia = int(linha.get("day") or 0)
            for mes in range(1, 13):
                v = _num_br(linha.get(f"rateValue{mes}"))
                if dia and not np.isnan(v):
                    try:
                        pontos[pd.Timestamp(ano, mes, dia)] = v
                    except ValueError:
                        pass
    if not pontos:
        raise ErroDados("B3 não retornou dados")
    return pd.Series(pontos).sort_index()


def _serie_msci(codigo: str, inicio: date, fim: date) -> pd.Series:
    cod, variante = codigo.split(":")
    fim = min(fim, date.today() - timedelta(days=1))
    r = requests.get(
        MSCI_URL,
        params={"currency_symbol": "USD", "index_variant": variante, "start_date": f"{inicio:%Y%m%d}",
                "end_date": f"{fim:%Y%m%d}", "data_frequency": "DAILY", "index_codes": cod},
        headers=UA, timeout=60,
    )
    r.raise_for_status()
    niveis = r.json()["indexes"]["INDEX_LEVELS"]
    s = pd.Series({pd.to_datetime(str(x["calc_date"]), format="%Y%m%d"): float(x["level_eod"]) for x in niveis})
    if s.empty:
        raise ErroDados("MSCI não retornou dados")
    return s.sort_index()




@st.cache_data(ttl=6 * 3600, show_spinner=False)
def serie_indice(codigo: str, inicio: date, fim: date) -> tuple[str, pd.DataFrame]:
    """Tenta as fontes em ordem; devolve (fonte usada, DataFrame)."""
    erros = []
    for fonte, cod in INDICES[codigo][2]:
        try:
            s = {"b3": _serie_b3, "msci": _serie_msci, "yf": _serie_yf}[fonte](cod, inicio, fim)
            s = s.loc[pd.Timestamp(inicio): pd.Timestamp(fim)]
            if len(s) > 1:
                rot = NOME_FONTE[fonte] + ("" if fonte != "yf" else f" ({cod})")
                return rot, s.to_frame("valor")
        except Exception as e:  # noqa: BLE001
            msg = str(e)[:60] if isinstance(e, ErroDados) else type(e).__name__
            erros.append(f"{NOME_FONTE[fonte]} ({cod}): {msg}")
    raise ErroDados("nenhuma fonte respondeu (" + "; ".join(erros) + ")")


CPI_DISCO = os.path.join(PASTA, "cpi_eua.csv")


def _segredo(nome: str) -> str | None:
    try:
        v = st.secrets.get(nome)
    except Exception:  # sem secrets configurados
        v = None
    return v or os.environ.get(nome)


def _cpi_oecd() -> pd.Series:
    """OECD (SDMX, sem chave): CPI total dos EUA, índice, sem ajuste sazonal."""
    ini = date.today().year - 20
    url = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/"
           "USA.M.N.CPI..._T.N.")
    r = requests.get(url, params={"startPeriod": f"{ini}-01", "format": "csvfile"}, headers=UA, timeout=25)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), dtype=str)
    for coluna, valor in (("UNIT_MEASURE", "IX"), ("TRANSFORMATION", "_Z"), ("ADJUSTMENT", "N"),
                          ("METHODOLOGY", "N"), ("EXPENDITURE", "_T")):
        if coluna in df and (df[coluna] == valor).any():
            df = df[df[coluna] == valor]
    if "BASE_PER" in df and df["BASE_PER"].nunique() > 1:  # se houver mais de uma base, fica com a mais longa
        df = df[df["BASE_PER"] == df["BASE_PER"].value_counts().idxmax()]
    return pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values,
                     index=pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m"))


def _cpi_fred_api() -> pd.Series:
    """API oficial do FRED (precisa de chave grátis em FRED_API_KEY nos Secrets)."""
    chave = _segredo("FRED_API_KEY")
    if not chave:
        raise ErroDados("sem chave")
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                     params={"series_id": "CPIAUCNS", "api_key": chave, "file_type": "json",
                             "observation_start": f"{date.today().year - 20}-01-01"},
                     headers=UA, timeout=25)
    r.raise_for_status()
    obs = r.json()["observations"]
    return pd.Series({pd.Timestamp(o["date"]): _num_br(o["value"]) for o in obs if o["value"] != "."})


def _cpi_bls() -> pd.Series:
    """API pública do BLS. Sem chave (v1): 10 anos por consulta e limite diário por IP.
    Com chave grátis em BLS_API_KEY (v2): limite bem maior."""
    chave = _segredo("BLS_API_KEY")
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/" if chave else "https://api.bls.gov/publicAPI/v1/timeseries/data/"
    pontos, fim = {}, date.today().year
    for ini in (fim - 19, fim - 9):
        corpo = {"seriesid": ["CUUR0000SA0"], "startyear": str(ini), "endyear": str(ini + 9)}
        if chave:
            corpo["registrationkey"] = chave
        r = requests.post(url, json=corpo, headers={**UA, "Content-Type": "application/json"}, timeout=25)
        r.raise_for_status()
        j = r.json()
        if j.get("status") != "REQUEST_SUCCEEDED":
            raise ErroDados("; ".join(j.get("message", [])) or "recusado")
        for x in j["Results"]["series"][0]["data"]:
            if x["period"].startswith("M") and x["period"] != "M13":
                pontos[pd.Timestamp(int(x["year"]), int(x["period"][1:]), 1)] = _num_br(x["value"])
    return pd.Series(pontos)


def _cpi_fred() -> pd.Series:
    r = requests.get(FRED_CSV, params={"id": "CPIAUCNS"}, headers=UA, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(df.iloc[:, 0]))


def _cpi_dbnomics() -> pd.Series:
    r = requests.get("https://api.db.nomics.world/v22/series/BLS/cu/CUUR0000SA0",
                     params={"observations": 1}, headers=UA, timeout=20)
    r.raise_for_status()
    doc = r.json()["series"]["docs"][0]
    return pd.Series(pd.to_numeric(pd.Series(doc["value"]), errors="coerce").values,
                     index=pd.to_datetime(doc["period"], format="%Y-%m"))


FONTES_CPI = (("FRED (API)", _cpi_fred_api), ("BLS", _cpi_bls), ("OECD", _cpi_oecd),
              ("FRED", _cpi_fred), ("DBnomics", _cpi_dbnomics))


@st.cache_resource(show_spinner=False)
def _memoria_cpi() -> dict:
    """Guarda, por fonte, o último resultado (sucesso ou falha) para não repetir consultas lentas."""
    return {}


def _cpi_eua() -> tuple[str, pd.Series, list[str]]:
    """CPI-U dos EUA (sem ajuste sazonal). Tenta várias fontes; usa a primeira que estiver em dia.
    Sucesso fica guardado por 12 h; falha, por 30 min."""
    memoria, agora = _memoria_cpi(), time.time()
    em_dia = pd.Timestamp(date.today()) - pd.Timedelta(days=80)  # CPI sai ~15 dias após o fim do mês
    erros, candidatos = [], []
    for nome, func in FONTES_CPI:
        quando, s, erro = memoria.get(nome, (0, None, None))
        validade = 12 * 3600 if s is not None else 30 * 60
        if agora - quando > validade:
            try:
                s, erro = func().dropna().sort_index(), None
                if len(s) < 24:
                    s, erro = None, "poucos dados"
            except Exception as e:  # noqa: BLE001
                s, erro = None, (str(e)[:80] if isinstance(e, ErroDados) else type(e).__name__)
            memoria[nome] = (agora, s, erro)
        if s is None:
            erros.append(f"{nome}: {erro}")
            continue
        if s.index[-1] >= em_dia:
            try:
                s.to_frame("cpi").to_csv(CPI_DISCO)
            except OSError:
                pass
            return nome, s, erros
        erros.append(f"{nome}: desatualizado (último {s.index[-1]:%m/%Y})")
        candidatos.append((s.index[-1], nome, s))
    if os.path.exists(CPI_DISCO):
        s = pd.read_csv(CPI_DISCO, index_col=0, parse_dates=True)["cpi"].sort_index()
        candidatos.append((s.index[-1], "cópia salva", s))
    if candidatos:  # nenhuma fonte em dia: usa a mais recente que houver
        _, nome, s = max(candidatos, key=lambda c: c[0])
        return f"{nome} (último dado {s.index[-1]:%m/%Y})", s, erros
    raise ErroDados("CPI dos EUA indisponível agora (" + "; ".join(erros) + ").")


DETALHES_INF: dict[str, dict] = {}


def serie_inflacao_mais(base: str, spread: float, inicio: date, fim: date) -> tuple[str, pd.DataFrame, dict]:
    """Índice mensal (fim de mês) de inflação + spread % a.a.

    Ponto-base = último fim de mês ≤ início do período (ex.: período a partir de 20/10/2025 → base set/2025).
    Cada mês M recebe a inflação de M e 1/12 do spread composto: (1 + spread)^(meses/12).
    A série termina no último mês com inflação publicada (CPI sai ~dia 15 do mês seguinte; IPCA ~dia 10).
    """
    b = pd.offsets.MonthEnd().rollback(pd.Timestamp(inicio))
    limite = pd.Timestamp(fim) + pd.offsets.MonthEnd(0)
    if base == "ipca":
        idx = serie_bcb("433", inicio, fim)["valor"]
        fonte, nivel = "Banco Central (IPCA)", None
    else:
        fonte_cpi, cpi_total, erros = _cpi_eua()
        cpi_total = cpi_total.copy()
        cpi_total.index = cpi_total.index + pd.offsets.MonthEnd(0)  # CPI do mês M no fim de M
        antes = cpi_total[cpi_total.index <= b]
        base_pt = antes.index[-1] if len(antes) else cpi_total.index[0]
        cpi = cpi_total[(cpi_total.index >= base_pt) & (cpi_total.index <= limite)]
        if len(cpi) < 2:
            raise ErroDados(
                f"CPI sem dados para o período: a fonte disponível ({fonte_cpi}) vai só até "
                f"{cpi_total.index[-1]:%m/%Y}. Outras fontes: " + "; ".join(erros) + ". "
                "Dica: cadastre uma chave grátis do FRED (FRED_API_KEY) nos Secrets do app."
            )
        idx = cpi / cpi.iloc[0]
        fonte, nivel = f"CPI-U dos EUA via {fonte_cpi}", cpi
    # meses decorridos pela data (há meses sem CPI, ex.: out/2025, não publicado no shutdown dos EUA)
    meses = np.asarray((idx.index.year - idx.index[0].year) * 12 + (idx.index.month - idx.index[0].month))
    out = idx.values * (1 + spread / 100) ** (meses / 12)
    faltando = [m.strftime("%m/%Y") for m in pd.date_range(idx.index[0], idx.index[-1], freq="ME")
                if m not in idx.index]
    detalhe = {
        "Inflação": "IPCA" if base == "ipca" else "CPI-U (EUA, sem ajuste sazonal)",
        "Fonte": fonte,
        "Mês-base": f"{idx.index[0]:%m/%Y}" + (f" (índice {nivel.iloc[0]:,.3f})" if nivel is not None else ""),
        "Último mês publicado": f"{idx.index[-1]:%m/%Y}" + (f" (índice {nivel.iloc[-1]:,.3f})" if nivel is not None else ""),
        "Meses": int(meses[-1]),
        "Inflação acumulada": float(idx.iloc[-1] - 1),
        "Spread acumulado": float((1 + spread / 100) ** (meses[-1] / 12) - 1),
        "Total (inflação + spread)": float(out[-1] - 1),
        "Meses sem dado publicado": ", ".join(faltando) or "nenhum",
    }
    return fonte, pd.DataFrame({"valor": out}, index=idx.index), detalhe


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def serie_ptax(inicio: date, fim: date) -> pd.Series:
    """Dólar PTAX venda (R$/US$): SGS do Banco Central → API PTAX (Olinda) do BC → Yahoo (BRL=X)."""
    erros = []
    try:
        s = serie_bcb("1", inicio, fim)["valor"]
        if len(s) > 1:
            return s
    except Exception as e:  # noqa: BLE001
        erros.append(f"SGS: {e}")
    try:
        url = (
            "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
            "CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)"
            f"?@dataInicial='{inicio:%m-%d-%Y}'&@dataFinalCotacao='{fim:%m-%d-%Y}'"
            "&$top=100000&$format=json&$select=cotacaoVenda,dataHoraCotacao"
        )
        r = requests.get(url, headers=UA, timeout=45)
        r.raise_for_status()
        v = r.json()["value"]
        s = pd.Series({pd.Timestamp(x["dataHoraCotacao"][:10]): float(x["cotacaoVenda"]) for x in v}).sort_index()
        if len(s) > 1:
            return s
    except Exception as e:  # noqa: BLE001
        erros.append(f"PTAX/Olinda: {type(e).__name__}")
    try:
        return _serie_yf("BRL=X", inicio, fim)
    except Exception as e:  # noqa: BLE001
        erros.append(f"Yahoo: {type(e).__name__}")
    raise ErroDados("; ".join(erros))


# ──────────────────────────────────────────────────────────────────────────
# Ponto único de acesso:
#   "fi:<cnpj>" | "bcb:<código>" | "yf:<ticker>" | "idx:<código>" | "inf:<ipca|cpi>:<spread>"
# ──────────────────────────────────────────────────────────────────────────
def moeda(ident: str) -> str:
    fonte, cod = ident.split(":", 1)
    if fonte == "idx":
        return INDICES[cod][1]
    if fonte == "yf":
        return "BRL" if cod.endswith(".SA") or cod == "^BVSP" else "USD"
    if fonte == "inf":
        return "USD" if cod.startswith("cpi") else "BRL"
    return "BRL"


def nome_padrao(ident: str) -> str:
    fonte, cod = ident.split(":", 1)
    if fonte == "bcb":
        return BCB.get(cod, (cod,))[0]
    if fonte == "fi":
        cad = cadastro()
        return cad.at[cod, "nome"] if cod in cad.index else fmt_cnpj(cod)
    if fonte == "idx":
        return INDICES[cod][0]
    if fonte == "inf":
        base, spread = cod.split(":")
        return f"{'IPCA' if base == 'ipca' else 'CPI (EUA)'} + {fmt_num(float(spread), 2).rstrip('0').rstrip(',')}%"
    return {"^BVSP": "Ibovespa", "^GSPC": "S&P 500"}.get(cod, cod.replace(".SA", ""))


def carregar(ident: str, inicio: date, fim: date) -> pd.DataFrame:
    fonte, cod = ident.split(":", 1)
    if fonte == "fi":
        FONTES_USADAS[ident] = "CVM"
        return serie_fundo(cod, inicio, fim)
    if fonte == "bcb":
        FONTES_USADAS[ident] = "Banco Central"
        return serie_bcb(cod, inicio, fim)
    if fonte == "yf":
        FONTES_USADAS[ident] = "Yahoo Finance"
        return serie_yahoo(cod, inicio, fim)[1]
    if fonte == "idx":
        FONTES_USADAS[ident], df = serie_indice(cod, inicio, fim)
        return df
    if fonte == "inf":
        base, spread = cod.split(":")
        FONTES_USADAS[ident], df, DETALHES_INF[ident] = serie_inflacao_mais(base, float(spread), inicio, fim)
        return df
    raise ErroDados(f"Identificador desconhecido: {ident}")


# ──────────────────────────────────────────────────────────────────────────
# Cálculos
# ──────────────────────────────────────────────────────────────────────────
def periodos_por_ano(s: pd.Series) -> float:
    if len(s) < 3:
        return 252.0
    passo = np.median(np.diff(s.index.values).astype("timedelta64[D]").astype(float))
    return 252.0 if passo <= 5 else (52.0 if passo <= 10 else 12.0)


def max_drawdown(s: pd.Series) -> float:
    return float((s / s.cummax() - 1).min()) if len(s) else np.nan


def retorno_anualizado(s: pd.Series) -> float:
    dias = (s.index[-1] - s.index[0]).days
    if dias < 365:
        return np.nan
    return float((s.iloc[-1] / s.iloc[0]) ** (365.25 / dias) - 1)


def retornos_mensais(s: pd.Series) -> pd.Series:
    fim_mes = s.resample("ME").last().dropna()
    base = pd.concat([s.iloc[:1], fim_mes])
    base = base[~base.index.duplicated(keep="last")]
    return base.pct_change().dropna()


def metricas(s: pd.Series, bench: pd.Series | None, extra: pd.DataFrame) -> dict:
    s = s.dropna()
    if len(s) < 2:
        return {}
    ret = s.pct_change().dropna()
    vol = float(ret.std() * np.sqrt(periodos_por_ano(s))) if len(ret) > 2 else np.nan
    acum = s.iloc[-1] / s.iloc[0] - 1
    anual = retorno_anualizado(s)
    mens = retornos_mensais(s)
    linha = {
        "Início": s.index[0].date(),
        "Fim": s.index[-1].date(),
        "Retorno acumulado": acum,
        "Retorno anualizado": anual,
        "Volatilidade (a.a.)": vol,
        "Máx. drawdown": max_drawdown(s),
        "% meses positivos": float((mens > 0).mean()) if len(mens) else np.nan,
        "Melhor mês": float(mens.max()) if len(mens) else np.nan,
        "Pior mês": float(mens.min()) if len(mens) else np.nan,
    }
    if bench is not None:
        b = bench.dropna()
        b = b[(b.index >= s.index[0] - pd.Timedelta(days=3)) & (b.index <= s.index[-1])]
        if len(b) > 1:
            b_acum = b.iloc[-1] / b.iloc[0] - 1
            linha["% do benchmark"] = acum / b_acum if b_acum else np.nan
            usar_anual = not np.isnan(anual)
            r, rb = (anual, retorno_anualizado(b)) if usar_anual else (acum, b_acum)
            linha["Sharpe (vs bench.)"] = (r - rb) / vol if vol and not np.isnan(vol) else np.nan
    for c, rot, div in (("pl", "PL (R$ mi)", 1e6), ("cotistas", "Cotistas", 1)):
        if c in extra and extra[c].notna().any():
            linha[rot] = float(extra[c].dropna().iloc[-1]) / div
    if "captacao" in extra and extra["captacao"].notna().any():
        linha["Captação líq. (R$ mi)"] = float((extra["captacao"] - extra["resgate"]).sum()) / 1e6
    return linha


def tabela_calendario(s: pd.Series, bench: pd.Series | None) -> pd.DataFrame:
    r = retornos_mensais(s.dropna())
    if r.empty:
        return pd.DataFrame()
    t = pd.DataFrame({"ano": r.index.year, "mes": r.index.month, "r": r.values})
    tab = t.pivot(index="ano", columns="mes", values="r").reindex(columns=range(1, 13))
    tab.columns = MESES
    tab["Ano"] = t.groupby("ano")["r"].apply(lambda x: (1 + x).prod() - 1)
    if bench is not None and len(bench.dropna()) > 1:
        rb = retornos_mensais(bench.dropna())
        rb = rb[rb.index.isin(r.index)]
        if not rb.empty:
            tab["Bench."] = (1 + rb).groupby(rb.index.year).prod() - 1
            tab["% Bench."] = tab["Ano"] / tab["Bench."]
    tab.index.name = None
    return tab.sort_index(ascending=False)


def cor_retorno(x) -> str:
    """Verde para mês positivo, vermelho para negativo; intensidade até ±5% (sem matplotlib)."""
    if pd.isna(x) or x == 0:
        return ""
    alfa = min(abs(float(x)) / 0.05, 1.0) * 0.55 + 0.08
    rgb = "34, 160, 90" if x > 0 else "215, 60, 60"
    return f"background-color: rgba({rgb}, {alfa:.2f})"


def fmt_pct(x):
    return "–" if pd.isna(x) else f"{x:.2%}".replace(".", ",")


def fmt_num(x, casas=1):
    if pd.isna(x):
        return "–"
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def alinhar(ref: pd.Series, datas: pd.Index) -> pd.Series:
    """Valor de `ref` em cada data (último disponível até a data)."""
    return ref.reindex(ref.index.union(datas)).ffill().reindex(datas)


def em_moeda(s: pd.Series, origem: str, destino: str, ptax: pd.Series | None) -> pd.Series | None:
    if origem == destino:
        return s
    if ptax is None:
        return None
    cambio = alinhar(ptax, s.index)
    return (s * cambio if destino == "BRL" else s / cambio).dropna()


def retorno_janela(s: pd.Series, de: pd.Timestamp, ate: pd.Timestamp) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Retorno entre o último dado até `de` (fim do mês anterior ao 1º mês escolhido) e o último até `ate`."""
    s = s.dropna()
    antes = s[s.index <= de]
    base = antes.index[-1] if len(antes) else (s.index[0] if len(s) else None)
    final = s[s.index <= ate]
    if base is None or not len(final) or final.index[-1] <= base:
        return np.nan, None, None
    return float(final.iloc[-1] / s.loc[base] - 1), base, final.index[-1]


# ──────────────────────────────────────────────────────────────────────────
# Estado (ativos escolhidos ficam também na URL)
# ──────────────────────────────────────────────────────────────────────────
if "fi_ativos" not in st.session_state:
    st.session_state.fi_ativos = {}
    for ident in [a.strip() for a in st.query_params.get("ativos", "").split(",") if ":" in a]:
        st.session_state.fi_ativos[ident] = None  # nome resolvido depois


def adicionar(ident: str, nome: str | None = None):
    st.session_state.fi_ativos[ident] = nome
    st.query_params["ativos"] = ",".join(st.session_state.fi_ativos)


# ──────────────────────────────────────────────────────────────────────────
# Interface
# ──────────────────────────────────────────────────────────────────────────
st.title("Fundos & Indicadores")
st.caption(
    "Dados públicos e gratuitos: CVM (fundos), Banco Central (CDI, Selic, IPCA, IGP-M, Dólar) e "
    "Yahoo Finance (bolsa). Rentabilidade passada não garante rentabilidade futura."
)

with st.container(border=True):
    st.markdown("**1. Escolha os ativos**")
    c1, c2 = st.container(), st.container()  # busca em cima, atalhos embaixo (largura total)
    with c1:
        tipo = st.radio("Tipo", ["Fundo (CVM)", "Ação / ETF / FII / índice (Yahoo)"], horizontal=True)
        if tipo.startswith("Fundo"):
            termo = st.text_input("Nome ou CNPJ do fundo", placeholder="Ex.: Verde, Kapitalo, 22.187.946/0001-41")
            if termo and len(termo.strip()) >= 3:
                try:
                    res = buscar_fundos(termo)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Não consegui carregar o cadastro da CVM: {e}")
                    res = pd.DataFrame()
                if len(res):
                    opcoes = {
                        f"{r.nome}  ·  {fmt_cnpj(c)}" + ("" if r.ativo else "  (cancelado)"): c
                        for c, r in res.iterrows()
                    }
                    escolha = st.selectbox("Resultados", list(opcoes))
                    if st.button("Adicionar fundo", type="primary"):
                        cnpj = opcoes[escolha]
                        adicionar(f"fi:{cnpj}", res.at[cnpj, "nome"])
                        st.rerun()
                elif len(so_digitos(termo)) == 14:
                    st.info("CNPJ não está no cadastro, mas posso tentar buscar as cotas mesmo assim.")
                    if st.button("Adicionar CNPJ", type="primary"):
                        adicionar(f"fi:{so_digitos(termo)}", fmt_cnpj(termo))
                        st.rerun()
                else:
                    st.info("Nenhum fundo encontrado.")
        else:
            termo = st.text_input("Ticker", placeholder="Ex.: PETR4, BOVA11, HGLG11, AAPL, ^BVSP")
            if termo and st.button("Adicionar ticker", type="primary"):
                t = ticker_yahoo(termo)
                adicionar(f"yf:{t}", nome_padrao(f"yf:{t}"))
                st.rerun()
    with c2:
        mapa_atalhos = dict(ATALHOS)
        grupos = [
            ("Brasil", ["CDI", "Selic", "IPCA", "IGP-M", "IPCA + 5%", "Dólar", "Ibovespa"]),
            ("Exterior", ["S&P 500 TR", "MSCI World", "MSCI ACWI", "CPI + 5%"]),
        ]
        if hasattr(st, "pills"):  # botões compactos que mostram o nome inteiro
            def _clicou_atalho(chave):
                rot = st.session_state.get(chave)
                if rot:
                    adicionar(mapa_atalhos[rot], nome_padrao(mapa_atalhos[rot]))
                    st.session_state[chave] = None

            for titulo, rotulos_grupo in grupos:
                st.pills(f"Atalhos · {titulo}", rotulos_grupo, key=f"pills_{titulo}",
                         on_change=_clicou_atalho, args=(f"pills_{titulo}",))
        else:  # versões antigas do Streamlit
            for titulo, rotulos_grupo in grupos:
                st.caption(f"Atalhos · {titulo}")
                cols = st.columns(2)
                for k, rot in enumerate(rotulos_grupo):
                    if cols[k % 2].button(rot, key=f"at_{rot}", use_container_width=True):
                        adicionar(mapa_atalhos[rot], nome_padrao(mapa_atalhos[rot]))
                        st.rerun()
        with st.expander("Inflação + spread personalizado"):
            e1, e2 = st.columns(2)
            base_inf = e1.selectbox("Inflação", ["IPCA", "CPI (EUA)"], key="inf_base")
            spread = e2.number_input("Spread (% a.a.)", value=5.0, step=0.5, key="inf_spread")
            if st.button("Adicionar", key="inf_add"):
                ident = f"inf:{'ipca' if base_inf == 'IPCA' else 'cpi'}:{spread:g}"
                adicionar(ident, nome_padrao(ident))
                st.rerun()
    lugar_selecionados = st.container()

if not st.session_state.fi_ativos:
    st.info("Busque um fundo, digite um ticker ou use os atalhos para começar.")
    st.stop()

# Período --------------------------------------------------------------------
hoje = date.today()
c1, c2, c3 = st.columns([2, 2, 3])
INICIO_GESTAO = date(2021, 12, 31)  # início da gestão
ROTULO_GESTAO = f"Desde o início da gestão ({INICIO_GESTAO:%d/%m/%Y})"
janela = c1.selectbox("Período", ["12 meses", "No ano", "24 meses", "36 meses", "60 meses", ROTULO_GESTAO, "Personalizado"])
if janela == "Personalizado":
    datas = c2.date_input("De / até", value=(hoje - timedelta(days=365), hoje), format="DD/MM/YYYY")
    inicio, fim = (datas if len(datas) == 2 else (datas[0], hoje))
elif janela == "No ano":
    inicio, fim = date(hoje.year - 1, 12, 31), hoje
elif janela == ROTULO_GESTAO:
    inicio, fim = INICIO_GESTAO, hoje
else:
    inicio, fim = (pd.Timestamp(hoje) - pd.DateOffset(months=int(janela.split()[0]))).date(), hoje
if janela != "Personalizado":
    c2.text_input("De / até", f"{inicio:%d/%m/%Y} – {fim:%d/%m/%Y}", disabled=True)
tem_usd = any(moeda(i) == "USD" for i in st.session_state.fi_ativos)
em_reais = st.toggle(
    "Converter ativos em dólar para reais (PTAX)", value=False, disabled=not tem_usd,
    help="S&P 500 TR, MSCI e ações dos EUA são cotados em USD. Ligue para comparar com fundos em R$ sem hedge.",
)

# Download -------------------------------------------------------------------
ids = list(st.session_state.fi_ativos)
if any(i.startswith("fi:") for i in ids):
    preparar_meses_cvm(inicio, fim)

series, extras, nativas, falhas = {}, {}, {}, []
ptax, erro_ptax = None, None  # dólar PTAX: usado na conversão do gráfico e na tabela R$ × US$
with st.spinner("Buscando dados…"):
    try:
        ptax = serie_ptax(inicio - timedelta(days=45), fim)  # folga p/ o ponto-base de séries mensais
    except Exception as e:  # noqa: BLE001
        erro_ptax = str(e)
    for ident in ids:
        if not st.session_state.fi_ativos.get(ident):
            st.session_state.fi_ativos[ident] = nome_padrao(ident)
        nome = st.session_state.fi_ativos[ident]
        try:
            df = carregar(ident, inicio, fim)
        except Exception as e:  # noqa: BLE001 — uma fonte fora do ar não derruba as outras
            falhas.append(f"**{nome}**: {e}")
            continue
        if df.empty or df["valor"].dropna().empty:
            falhas.append(f"**{nome}**: sem dados no período.")
            continue
        nativas[ident] = df["valor"].dropna()
        if em_reais and moeda(ident) == "USD":
            if ptax is None:
                falhas.append(f"**{nome}**: não consegui a PTAX para converter ({erro_ptax}).")
                continue
            df = df.copy()
            df["valor"] = df["valor"] * alinhar(ptax, df.index)
        series[ident] = df["valor"].dropna()
        extras[ident] = df

with lugar_selecionados:
    rotulos = {st.session_state.fi_ativos[i]: i for i in st.session_state.fi_ativos}
    mantidos = [
        rotulos[r] for r in st.multiselect(
            "Ativos selecionados (clique no × para remover)", list(rotulos), default=list(rotulos)
        )
    ]
    if set(mantidos) != set(st.session_state.fi_ativos):
        st.session_state.fi_ativos = {i: st.session_state.fi_ativos[i] for i in mantidos}
        st.query_params["ativos"] = ",".join(mantidos)
        st.rerun()

for f in falhas:
    st.warning(f)
if not series:
    st.stop()

nomes = {
    i: st.session_state.fi_ativos[i].replace("(USD)", "(R$)") + ("" if "(USD)" in st.session_state.fi_ativos[i] else " em R$")
    if em_reais and moeda(i) == "USD" else st.session_state.fi_ativos[i]
    for i in series
}
rotulos_bench = {"Nenhum": None, **{nomes[i]: i for i in series}}
padrao = next((k for k, v in enumerate(rotulos_bench.values()) if v == "bcb:12"), 0)
bench_id = rotulos_bench[c3.selectbox("Benchmark (para % do bench. e Sharpe)", list(rotulos_bench), index=padrao)]
bench = series.get(bench_id) if bench_id else None

# Abas -----------------------------------------------------------------------
aba_rent, aba_comp, aba_mens, aba_dados = st.tabs(
    ["📈 Rentabilidade", "⚖️ Comparador", "🗓️ Retornos mensais", "⬇️ Dados"]
)
painel = pd.concat(series, axis=1, sort=True).ffill()

with aba_rent:
    acum = painel.apply(lambda c: c / c.dropna().iloc[0] - 1) * 100
    acum.columns = [nomes[c] for c in acum.columns]
    st.markdown("**Retorno acumulado no período (%)**")
    st.line_chart(acum, height=420)

    # ── Rentabilidade em R$ e em US$ nos meses escolhidos (independe do botão de conversão do gráfico)
    st.markdown("**Rentabilidade em R\\$ e em US\\$ nos meses escolhidos**")
    # o 1º mês é o do dia seguinte ao início (ex.: início 31/12/2021 → primeiro mês jan/22)
    meses_periodo = list(pd.period_range(pd.Timestamp(inicio) + pd.Timedelta(days=1), pd.Timestamp(fim), freq="M"))
    rot_mes = {m: f"{MESES[m.month - 1].lower()}/{m.year % 100:02d}" for m in meses_periodo}
    if len(meses_periodo) > 1:
        por_rot = {v: k for k, v in rot_mes.items()}
        r_ini, r_fim = st.select_slider(
            "Meses", options=list(por_rot), value=(rot_mes[meses_periodo[0]], rot_mes[meses_periodo[-1]]),
            key=f"fi_janela_{inicio:%Y%m%d}_{fim:%Y%m%d}",  # muda o período → volta a mostrar tudo
        )
        m_ini, m_fim = por_rot[r_ini], por_rot[r_fim]
    else:
        m_ini = m_fim = meses_periodo[0]
    visao = st.radio("Mostrar", ["Total do intervalo", "Mês a mês"], horizontal=True, key="fi_visao",
                     label_visibility="collapsed")
    de = (m_ini.to_timestamp() - pd.Timedelta(days=1)).normalize()   # fim do mês anterior
    ate = m_fim.to_timestamp(how="end").normalize()

    por_moeda = {}  # ident -> {"BRL": série, "USD": série}
    for i, s_nat in nativas.items():
        if i in series:
            por_moeda[i] = {m: em_moeda(s_nat, moeda(i), m, ptax) for m in ("BRL", "USD")}
    nome_base = {i: st.session_state.fi_ativos[i] for i in por_moeda}

    if visao == "Total do intervalo":
        linhas = {}
        for i, d in por_moeda.items():
            r_brl, b0, b1 = retorno_janela(d["BRL"], de, ate) if d["BRL"] is not None else (np.nan, None, None)
            r_usd, u0, u1 = retorno_janela(d["USD"], de, ate) if d["USD"] is not None else (np.nan, None, None)
            d0, d1 = (b0 or u0), (b1 or u1)
            linhas[nome_base[i]] = {
                "Em R$": r_brl, "Em US$": r_usd,
                "De": f"{d0:%d/%m/%Y}" if d0 is not None else "–",
                "Até": f"{d1:%d/%m/%Y}" if d1 is not None else "–",
                "Moeda original": "R$" if moeda(i) == "BRL" else "US$",
            }
        tab_moeda = pd.DataFrame(linhas).T
        st.dataframe(tab_moeda.style.format({"Em R$": fmt_pct, "Em US$": fmt_pct})
                     .map(cor_retorno, subset=["Em R$", "Em US$"]))
    else:
        for m_rot, chave in (("Em R$", "BRL"), ("Em US$", "USD")):
            mensal_m = {}
            for i, d in por_moeda.items():
                if d[chave] is None:
                    continue
                mensal_m[nome_base[i]] = {
                    rot_mes[m]: retorno_janela(d[chave], (m.to_timestamp() - pd.Timedelta(days=1)).normalize(),
                                               m.to_timestamp(how="end").normalize())[0]
                    for m in meses_periodo if m_ini <= m <= m_fim
                }
            if mensal_m:
                st.caption(m_rot.replace("$", "\\$"))
                st.dataframe(pd.DataFrame(mensal_m).T.style.format(fmt_pct).map(cor_retorno))
    if ptax is not None:
        r_dolar = retorno_janela(ptax, de, ate)[0]
        st.caption(
            f"Dólar PTAX no intervalo: {fmt_pct(r_dolar)}. Ativos em US\\$ (S&P 500, MSCI, CPI + X%, ações dos EUA) "
            "são convertidos para R\\$ pela PTAX, e ativos em R\\$ para US\\$ do mesmo jeito. "
            "Meses incompletos usam o dado disponível mais recente (veja as colunas De/Até)."
        )
    else:
        st.caption(f"Sem PTAX agora ({erro_ptax}): cada ativo aparece só na moeda original.")

    infl = {nome_base[i]: d for i, d in DETALHES_INF.items() if i in series}
    if infl:
        with st.expander("Como o CPI/IPCA + X% é calculado (conferência)"):
            st.dataframe(pd.DataFrame(infl).map(lambda v: fmt_pct(v) if isinstance(v, float) else str(v)))
            st.caption(
                "Índice = (inflação do mês-base até o último mês publicado) × (1 + spread)^(meses/12). "
                "O mês-base é o último fim de mês antes do início do período. A série para no último mês com "
                "inflação divulgada (CPI sai por volta do dia 15 do mês seguinte), então pode terminar antes dos "
                "outros ativos."
            )

    st.markdown("**Rentabilidade mês a mês**")
    por_nome = {nomes[i]: i for i in series}
    alvo = por_nome[st.selectbox("Ativo", list(por_nome), key="fi_cal")]
    cal = tabela_calendario(series[alvo], bench if alvo != bench_id else None)
    if not cal.empty:
        st.dataframe(cal.style.format(fmt_pct).map(cor_retorno, subset=MESES))

with aba_comp:
    tab = pd.DataFrame({nomes[i]: metricas(series[i], bench if i != bench_id else None, extras[i]) for i in series}).T
    formatos = {c: fmt_pct for c in tab if c in (
        "Retorno acumulado", "Retorno anualizado", "Volatilidade (a.a.)", "Máx. drawdown",
        "% meses positivos", "Melhor mês", "Pior mês", "% do benchmark")}
    formatos.update({c: (lambda x: fmt_num(x, 2)) for c in tab if c == "Sharpe (vs bench.)"})
    formatos.update({c: (lambda x: fmt_num(x, 1)) for c in tab if "R$ mi" in c})
    formatos.update({c: (lambda x: fmt_num(x, 0)) for c in tab if c == "Cotistas"})
    st.dataframe(tab.style.format(formatos))
    st.caption(
        "Volatilidade anualizada pela frequência da série (252 dias úteis ou 12 meses). "
        "Sharpe = (retorno anualizado − benchmark anualizado) ÷ volatilidade; em períodos < 1 ano usa o "
        "retorno do período. PL, cotistas e captação líquida (no período) vêm do informe diário da CVM."
    )
    st.markdown("**Drawdown (%)**")
    dd = painel.apply(lambda c: c / c.cummax() - 1) * 100
    dd.columns = [nomes[c] for c in dd.columns]
    st.area_chart(dd, height=300)

    fundos = [i for i in series if i.startswith("fi:")]
    if fundos:
        with st.expander("Ficha cadastral dos fundos (CVM)"):
            fichas = {nomes[i]: f for i in fundos if (f := ficha(i[3:]))}
            if fichas:
                st.dataframe(pd.DataFrame(fichas).fillna("").astype(str))
            else:
                st.info("Fundos não encontrados no cadastro.")

with aba_mens:
    mensal = pd.DataFrame({nomes[i]: retornos_mensais(s) for i, s in series.items()}).sort_index(ascending=False)
    mensal.index = mensal.index.strftime("%m/%Y")
    st.markdown("**Retorno mensal de todos os ativos**")
    st.dataframe(mensal.style.format(fmt_pct), height=480)

with aba_dados:
    bruto = painel.copy()
    bruto.columns = [nomes[c] for c in bruto.columns]
    st.dataframe(bruto.tail(30))
    st.caption("Fontes: " + " · ".join(f"{nomes[i]}: {FONTES_USADAS.get(i, '–')}" for i in series))
    st.download_button(
        "Baixar séries (CSV)",
        bruto.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
        file_name=f"fundos_indicadores_{inicio:%Y%m%d}_{fim:%Y%m%d}.csv",
        mime="text/csv",
    )
    st.caption(
        "Fundos: cota da CVM (já líquida de taxas). CDI/Selic/IPCA/IGP-M: índice acumulado a partir das taxas "
        "do Banco Central. Bolsa: preço ajustado por proventos (Yahoo). O link desta página guarda os ativos escolhidos."
    )
