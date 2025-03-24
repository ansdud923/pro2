import pandas as pd
import streamlit as st
import mysql.connector
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import plotly.colors as pc
import folium
import random


DB_HOST = "localhost"
DB_NAME = "online"
DB_USER = "root"
DB_PASS = "rnjsans0"
DB_PORT = "3306"

CAMPAIN_TABLE = "campaintbl"
OFF_TABLE = "offlinetbl"
ONLINE_TABLE = "onlinetbl"

# 한글 및 마이너스 깨짐
from matplotlib import rc

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 페이지 설정
st.set_page_config(
    page_title="캠페인 데이터 분석", page_icon=":bar_chart:", layout="wide"
)
st.title("🕵 환경 캠페인 데이터 분석")

# SQLAlchemy 엔진 생성
engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# SQLAlchemy 엔진 사용
def fetch_data(query):
    return pd.read_sql(query, engine)


### 사이드바 날짜 조회 추가
with st.sidebar:
    st.title("📅 날짜별 조회")

    # 캠페인 테이블에서 날짜 목록 가져오기
    date_query = f"SELECT DISTINCT 날짜 FROM {OFF_TABLE};"
    date_df = fetch_data(date_query)

    if not date_df.empty:
        date_options = date_df["날짜"].astype(str).tolist()  # 날짜 리스트 변환
        date_options.insert(0, "전체조회")  # 전체조회 옵션 추가

        selected_dates = st.multiselect("", options=date_options, default=["전체조회"])
    else:
        selected_dates = []  # 데이터가 없을 경우 빈 리스트


### 데이터셋 불러오기
def get_dataset(HEADER, TABLE_NAME, KEYNAME):
    st.subheader(HEADER)

    if st.button("데이터 조회", key=KEYNAME):
        query = f"SELECT * FROM {TABLE_NAME}"

        if selected_dates:
            if "전체조회" in selected_dates:
                pass  # 전체조회 시 WHERE 조건 제거 → 모든 데이터를 가져옴
            else:
                formatted_dates = ", ".join(f"'{date}'" for date in selected_dates)
                query += f" WHERE 날짜 IN ({formatted_dates})"

        df = fetch_data(query)
        st.dataframe(df.set_index("id"), use_container_width=True)


# 오프라인 - 데이터 시각화 옵션 선택
def input_off_column():
    st.markdown("#### 🔍유형별 시각화")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("##### ✔️X 축 선택")
        offline_x = st.radio(
            "",
            ["지역", "연령대", "성별", "이벤트 종류"],
            key="offline_key_x",
            horizontal=False,
        )

    with col2:
        st.markdown("##### ✔️Y 축 선택")
        offline_y = st.radio(
            "",
            ["방문자수", "참여자수", "참여비율"],
            key="offline_key_y",
            horizontal=False,
        )

    return offline_x, offline_y


# 온라인 - 데이터 시각화 옵션 선택
def input_online_column():
    st.markdown("#### 🔍유형별 시각화")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("##### ✔️X 축 선택")
        online_x = st.radio(
            "",
            ["디바이스", "유입경로", "키워드"],
            key="online_key_x",
            horizontal=False,
        )

    with col2:
        st.markdown("##### ✔️Y 축 선택")
        online_y = st.radio(
            "",
            [
                "노출수",
                "유입수",
                "체류시간(min)",
                "페이지뷰",
                "이탈수",
                "회원가입",
                "앱 다운",
                "구독",
            ],
            key="online_key_y",
            horizontal=False,
        )

    return online_x, online_y


palette = pc.qualitative.Pastel


# X=문자 Y=숫자 별 평균 참여율 비교 (막대 그래프)
def barplot(TABLE_NAME, x, y):
    query = f"SELECT * FROM {TABLE_NAME}"
    if selected_dates and "전체조회" not in selected_dates:
        formatted_dates = ", ".join(f"'{date}'" for date in selected_dates)
        query += f" WHERE 날짜 IN ({formatted_dates})"
    df = fetch_data(query)
    if not df.empty:
        df_grouped = df.groupby(x)[y].mean().reset_index()
        fig = px.bar(
            df_grouped,
            x=x,
            y=y,
            color=x,  # x 기준으로 색상 → legend 생길
            color_discrete_sequence=palette,
            title=f"{x}별 평균 {y} 비교",
            text_auto=True,
            hover_data=[y],
        )
        fig.update_layout(legend_title_text=x)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("조회된 데이터가 없습니다.")


# X=문자 Y=숫자 별 참여율 비교 (박스플롯)
def baxplot(TABLE_NAME, x, y):
    query = f"SELECT * FROM {TABLE_NAME}"
    if selected_dates and "전체조회" not in selected_dates:
        formatted_dates = ", ".join(f"'{date}'" for date in selected_dates)
        query += f" WHERE 날짜 IN ({formatted_dates})"
    df = fetch_data(query)
    if not df.empty:
        fig = px.box(
            df,
            x=x,
            y=y,
            title=f"{x}별 {y} 분포",
            points="outliers",
            color=x,
            color_discrete_sequence=palette,
            hover_data=[x, y],
        )
        fig.update_layout(legend_title_text=x)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("조회된 데이터가 없습니다.")


# X=문자 Y=숫자 별 평균 비교 (선그래프)
def linechart(TABLE_NAME, x, y):
    query = f"SELECT * FROM {TABLE_NAME}"
    if selected_dates and "전체조회" not in selected_dates:
        formatted_dates = ", ".join(f"'{date}'" for date in selected_dates)
        query += f" WHERE 날짜 IN ({formatted_dates})"

    df = fetch_data(query)
    if not df.empty:
        df_grouped = df.groupby(x)[y].mean().reset_index()

        fig = px.line(
            df_grouped,
            x=x,
            y=y,
            markers=True,
            color_discrete_sequence=palette,
            title=f"{x}별 평균 {y} 비교",
            hover_data=[y],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("조회된 데이터가 없습니다.")


# 산점도 그래프 (scatterplot)
def scatterplot(TABLE_NAME, x, y):
    query = f"SELECT * FROM {TABLE_NAME}"
    if selected_dates and "전체조회" not in selected_dates:
        formatted_dates = ", ".join(f"'{date}'" for date in selected_dates)
        query += f" WHERE 날짜 IN ({formatted_dates})"
    df = fetch_data(query)
    if not df.empty:
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=x,
            color_discrete_sequence=palette,
            title=f"{x} vs {y} 의 상관관계",
            hover_data=[x, y],
        )
        fig.update_layout(legend_title_text=x)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("조회된 데이터가 없습니다.")


# X=문자 Y=숫자 (파이플롯)
def piechart(TABLE_NAME, x, y):
    query = f"SELECT * FROM {TABLE_NAME}"
    if selected_dates and "전체조회" not in selected_dates:
        formatted_dates = ", ".join(f"'{date}'" for date in selected_dates)
        query += f" WHERE 날짜 IN ({formatted_dates})"
    df = fetch_data(query)
    if not df.empty:
        if x not in df.columns or y not in df.columns:
            st.error(f"⚠️ 컬럼 '{x}' 또는 '{y}'가 존재하지 않습니다.")
            return
        df_grouped = df.groupby(x)[y].sum().reset_index()
        fig = px.pie(
            df_grouped,
            names=x,
            values=y,
            title=f"{x}별 {y} 비율 비교",
            hole=0.3,
            color=x,
            color_discrete_sequence=palette,
            hover_data=[y],
        )
        fig.update_layout(
            legend_title_text=x, width=900, height=700  # 👉 가로 크기  # 👉 세로 크기
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 선택한 조건에 해당하는 데이터가 없습니다.")


def map_campain():
    query = f"SELECT * FROM {OFF_TABLE}"
    if selected_dates and "전체조회" not in selected_dates:
        formatted_dates = ", ".join(f"'{date}'" for date in selected_dates)
        query += f" WHERE 날짜 IN ({formatted_dates})"
    off_df = fetch_data(query)

    coordinates = {
        "인천": (37.4563, 126.7052),
        "강원": (37.8228, 128.1555),
        "충북": (36.6351, 127.4915),
        "경기": (37.4138, 127.5183),
        "울산": (35.5373, 129.3167),
        "제주": (33.4997, 126.5318),
        "전북": (35.7210, 127.1454),
        "대전": (36.3504, 127.3845),
        "대구": (35.8714, 128.6014),
        "서울": (37.5665, 126.9780),
        "충남": (36.6887, 126.7732),
        "경남": (35.2345, 128.6880),
        "세종": (36.4805, 127.2898),
        "경북": (36.1002, 128.6295),
        "부산": (35.1796, 129.0756),
        "광주": (35.1595, 126.8526),
        "전남": (34.7802, 126.1322),
    }

    if (
        not off_df.empty
        and "방문자수" in off_df.columns
        and "참여자수" in off_df.columns
    ):
        off_data_by_city = (
            off_df.groupby("지역")
            .agg({"방문자수": "sum", "참여자수": "sum"})
            .reset_index()
        )
        off_data_by_city = off_data_by_city.dropna(subset=["방문자수", "참여자수"])
        off_data_by_city["참여율"] = off_data_by_city.apply(
            lambda row: (
                (row["참여자수"] / row["방문자수"] * 100) if row["방문자수"] > 0 else 0
            ),
            axis=1,
        )
        off_data_by_city["위도"] = off_data_by_city["지역"].map(
            lambda x: coordinates.get(x, (None, None))[0]
        )
        off_data_by_city["경도"] = off_data_by_city["지역"].map(
            lambda x: coordinates.get(x, (None, None))[1]
        )
        valid_data = off_data_by_city.dropna(subset=["위도", "경도"])

        fig = px.scatter_geo(
            valid_data,
            lat="위도",
            lon="경도",
            size="참여율",
            color="지역",
            text="지역",  # ✅ 마커 안에 지역명 표시
            hover_name="지역",
            size_max=30,
            projection="natural earth",
            title="🗺️ 지역별 참여율 (Plotly 지도)",
        )
        fig.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
        fig.update_layout(
            legend_title_text="지역",
            height=650,
            geo=dict(
                center={"lat": 36.5, "lon": 127.8},  # ✅ 한국 중심으로 확대
                projection_scale=30,  # 확대 정도 (작을수록 확대됨)
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("지도에 표시할 데이터가 없습니다.")


# 예측 데이터셋 결측치 처리
def clean_df(df):
    df = df.copy()  # 원본 손상 방지
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df


tab1, tab2, tab3 = st.tabs(["오프라인", "온라인", "예상 결과"])


with tab1:
    get_dataset("1️⃣ OFFLINE DATA", OFF_TABLE, "off_btn")
    map_campain()
    tab1_x, tab1_y = input_off_column()
    col1, col2 = st.columns([1, 1])
    with col1:
        barplot(OFF_TABLE, tab1_x, tab1_y)
        linechart(OFF_TABLE, tab1_x, tab1_y)
    with col2:
        baxplot(OFF_TABLE, tab1_x, tab1_y)
        scatterplot(OFF_TABLE, tab1_x, tab1_y)
    piechart(OFF_TABLE, tab1_x, tab1_y)


with tab2:
    get_dataset("2️⃣ ONLINE DATA", ONLINE_TABLE, "online_btn")
    tab2_x, tab2_y = input_online_column()
    col1, col2 = st.columns([1, 1])
    with col1:
        barplot(ONLINE_TABLE, tab2_x, tab2_y)
        linechart(ONLINE_TABLE, tab2_x, tab2_y)
    with col2:
        baxplot(ONLINE_TABLE, tab2_x, tab2_y)
        scatterplot(ONLINE_TABLE, tab2_x, tab2_y)
    piechart(ONLINE_TABLE, tab2_x, tab2_y)


with tab3:
    offline_df = clean_df(fetch_data(f"SELECT * FROM {OFF_TABLE}"))
    online_df = clean_df(fetch_data(f"SELECT * FROM {ONLINE_TABLE}"))

    # 🔹 오프라인 모델 학습
    X_off = offline_df[
        ["지역", "연령대", "성별", "이벤트 종류", "방문자수", "참여자수"]
    ]
    y_off = offline_df["참여비율"]

    num_cols_off = ["방문자수", "참여자수"]
    cat_cols_off = ["지역", "연령대", "성별", "이벤트 종류"]

    pre_off = ColumnTransformer(
        [
            ("num", "passthrough", num_cols_off),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_off),
        ]
    )
    model_off = Pipeline([("pre", pre_off), ("reg", LinearRegression())])
    model_off.fit(X_off, y_off)

    # 🔹 온라인 모델 학습
    X_on = online_df[
        [
            "디바이스",
            "유입경로",
            "키워드",
            "노출수",
            "유입수",
            "회원가입",
            "앱 다운",
            "구독",
        ]
    ]
    y_on = online_df["전환수"]

    num_cols_on = ["노출수", "유입수", "회원가입", "앱 다운", "구독"]
    cat_cols_on = ["디바이스", "유입경로", "키워드"]

    pre_on = ColumnTransformer(
        [
            ("num", "passthrough", num_cols_on),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_on),
        ]
    )
    model_on = Pipeline([("pre", pre_on), ("reg", LinearRegression())])
    model_on.fit(X_on, y_on)

    # ✅ Streamlit UI
    st.title("🤖 캠페인 데이터 예측기")

    # ✅ 모드 선택
    mode = st.radio(
        "모드 선택",
        ["🧍‍♀️ 오프라인 캠페인 (참여비율 예측)", "📱 온라인 캠페인 (전환수 예측)"],
    )

    # 🔸 오프라인 예측기
    if "오프라인" in mode:
        st.subheader("🧍‍♀️ 오프라인 캠페인 참여비율 예측기")
        col1, col2 = st.columns(2)
        with col1:
            지역 = st.selectbox("지역", offline_df["지역"].unique())
            연령대 = st.selectbox("연령대", offline_df["연령대"].unique())
            성별 = st.selectbox("성별", offline_df["성별"].unique())
            이벤트 = st.selectbox("이벤트 종류", offline_df["이벤트 종류"].unique())
        with col2:
            방문자수 = st.number_input("방문자 수", min_value=0, value=100)
            참여자수 = st.number_input("참여자 수", min_value=0, value=50)

        input_df = pd.DataFrame(
            {
                "지역": [지역],
                "연령대": [연령대],
                "성별": [성별],
                "이벤트 종류": [이벤트],
                "방문자수": [방문자수],
                "참여자수": [참여자수],
            }
        )

        pred = model_off.predict(input_df)[0]
        st.success(f"✅ 예측된 참여비율: **{pred:.2f}%**")

    # 🔹 온라인 예측기
    else:
        st.subheader("📱 온라인 캠페인 전환수 예측기")
        col1, col2 = st.columns(2)
        with col1:
            디바이스 = st.selectbox("디바이스", online_df["디바이스"].unique())
            유입경로 = st.selectbox("유입경로", online_df["유입경로"].unique())
            키워드 = st.selectbox("키워드", online_df["키워드"].unique())
        with col2:
            노출수 = st.number_input("노출수", min_value=0, value=1000)
            유입수 = st.number_input("유입수", min_value=0, value=300)
            회원가입 = st.number_input("회원가입 수", min_value=0, value=100)
            앱다운 = st.number_input("앱 다운 수", min_value=0, value=50)
            구독 = st.number_input("구독 수", min_value=0, value=20)

        input_df = pd.DataFrame(
            {
                "디바이스": [디바이스],
                "유입경로": [유입경로],
                "키워드": [키워드],
                "노출수": [노출수],
                "유입수": [유입수],
                "회원가입": [회원가입],
                "앱 다운": [앱다운],
                "구독": [구독],
            }
        )

        pred = model_on.predict(input_df)[0]
        st.success(f"✅ 예측된 전환수: **{pred:.0f}건**")
