import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import plotly.colors as pc

CSV_BASE_PATH = "https://raw.githubusercontent.com/ansdud923/pro2/main/data/"
OFF_TABLE = "recycling_off.csv"
ONLINE_TABLE = "recycling_online.csv"


def fetch_data(file_name):
    return pd.read_csv(CSV_BASE_PATH + file_name, encoding="cp949")


st.set_page_config(
    page_title="캠페인 데이터 분석", page_icon=":bar_chart:", layout="wide"
)
st.title("🕵 환경 캠페인 데이터 분석")

palette = pc.qualitative.Pastel
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

offline_all = fetch_data(OFF_TABLE)
online_all = fetch_data(ONLINE_TABLE)

with st.sidebar:
    st.title("📅 날짜별 조회")
    if "날짜" in offline_all.columns:
        date_options = offline_all["날짜"].astype(str).dropna().unique().tolist()
        date_options.insert(0, "전체조회")
        selected_dates = st.multiselect("", options=date_options, default=["전체조회"])
    else:
        selected_dates = []


def filter_by_date(df):
    if selected_dates and "전체조회" not in selected_dates:
        return df[df["날짜"].astype(str).isin(selected_dates)]
    return df


def clean_df(df):
    df = df.copy()
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def barplot(df, x, y):
    if not df.empty:
        df_grouped = df.groupby(x)[y].mean().reset_index()
        fig = px.bar(
            df_grouped,
            x=x,
            y=y,
            color=x,
            color_discrete_sequence=palette,
            title=f"{x}별 평균 {y} 비교",
            text_auto=True,
            hover_data=[y],
        )
        fig.update_layout(legend_title_text=x)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("조회된 데이터가 없습니다.")


def baxplot(df, x, y):
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


def linechart(df, x, y):
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


def scatterplot(df, x, y):
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


def piechart(df, x, y):
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
        fig.update_layout(legend_title_text=x, width=900, height=700)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 선택한 조건에 해당하는 데이터가 없습니다.")


def map_campain():
    off_df = filter_by_date(fetch_data(OFF_TABLE))
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
    if not off_df.empty:
        off_data_by_city = (
            off_df.groupby("지역")
            .agg({"방문자수": "sum", "참여자수": "sum"})
            .reset_index()
        )
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
            text="지역",
            hover_name="지역",
            size_max=30,
            projection="natural earth",
            title="🗺️ 지역별 참여율 (Plotly 지도)",
        )
        fig.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
        fig.update_layout(
            legend_title_text="지역",
            height=650,
            geo=dict(center={"lat": 36.5, "lon": 127.8}, projection_scale=30),
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


# ----- 탭 구성 -----
tab1, tab2, tab3 = st.tabs(["오프라인", "온라인", "예상 결과"])

with tab1:
    st.subheader("1️⃣ 오프라인 캠페인 분석")
    map_campain()
    offline_df = filter_by_date(fetch_data(OFF_TABLE))
    if not offline_df.empty:
        col1, col2 = st.columns([1, 1])
        with col1:
            offline_x = st.selectbox(
                "X축 선택", ["지역", "연령대", "성별", "이벤트 종류"], key="offline_x"
            )
        with col2:
            offline_y = st.selectbox(
                "Y축 선택", ["방문자수", "참여자수", "참여비율"], key="offline_y"
            )

        barplot(offline_df, offline_x, offline_y)
        linechart(offline_df, offline_x, offline_y)
        baxplot(offline_df, offline_x, offline_y)
        scatterplot(offline_df, offline_x, offline_y)
        piechart(offline_df, offline_x, offline_y)

with tab2:
    st.subheader("2️⃣ 온라인 캠페인 분석")
    online_df = filter_by_date(fetch_data(ONLINE_TABLE))
    if not online_df.empty:
        col1, col2 = st.columns([1, 1])
        with col1:
            online_x = st.selectbox(
                "X축 선택", ["디바이스", "유입경로", "키워드"], key="online_x"
            )
        with col2:
            online_y = st.selectbox(
                "Y축 선택",
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
                key="online_y",
            )

        barplot(online_df, online_x, online_y)
        linechart(online_df, online_x, online_y)
        baxplot(online_df, online_x, online_y)
        scatterplot(online_df, online_x, online_y)
        piechart(online_df, online_x, online_y)

with tab3:
    st.subheader("🤖 캠페인 예측기")
    offline_df = clean_df(filter_by_date(fetch_data(OFF_TABLE)))
    online_df = clean_df(filter_by_date(fetch_data(ONLINE_TABLE)))

    mode = st.radio(
        "예측할 캠페인을 선택하세요",
        ["🧍‍♀️ 오프라인 참여비율 예측", "📱 온라인 전환수 예측"],
    )

    if "오프라인" in mode:
        st.markdown("#### 오프라인 캠페인 참여비율 예측기")
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

        X = offline_df[
            ["지역", "연령대", "성별", "이벤트 종류", "방문자수", "참여자수"]
        ]
        y = offline_df["참여비율"]

        pre = ColumnTransformer(
            [
                ("num", "passthrough", ["방문자수", "참여자수"]),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["지역", "연령대", "성별", "이벤트 종류"],
                ),
            ]
        )
        model = Pipeline([("pre", pre), ("reg", LinearRegression())])
        model.fit(X, y)
        pred = model.predict(input_df)[0]
        st.success(f"✅ 예측된 참여비율: {pred:.2f}%")

    else:
        st.markdown("#### 온라인 캠페인 전환수 예측기")
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

        X = online_df[
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
        y = online_df["전환수"]

        pre = ColumnTransformer(
            [
                (
                    "num",
                    "passthrough",
                    ["노출수", "유입수", "회원가입", "앱 다운", "구독"],
                ),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["디바이스", "유입경로", "키워드"],
                ),
            ]
        )
        model = Pipeline([("pre", pre), ("reg", LinearRegression())])
        model.fit(X, y)
        pred = model.predict(input_df)[0]
        st.success(f"✅ 예측된 전환수: {pred:.0f}건")
