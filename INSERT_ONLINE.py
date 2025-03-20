import pandas as pd
import mysql.connector
from mysql.connector import Error
import pymysql
import streamlit as st
from sqlalchemy import create_engine
import numpy as np

# CSV 파일 경로
CSV_FILE_PATH = "./data/recycling_online.csv"

# MySQL 연결 정보
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "online"
DB_USER = "root"
DB_PASS = "rnjsans0"
DB_TABLE = "onlinetbl"


def create_database_connection():  # DB와 연결
    try:  # 팅기지 않게 하기위해
        connection = mysql.connector.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS
        )

        if connection.is_connected():
            print("MySQL 데이터베이스에 성공적으로 연결되었습니다.")
        return connection
    except Error as e:
        print(f"데이터베이스 연결 중 오류 발생:{e}")
        return None


def create_table(connection):
    try:
        cursor = connection.cursor()
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DB_TABLE} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            날짜 DATE,
            디바이스 VARCHAR(20),
            유입경로 VARCHAR(20),
            키워드 VARCHAR(20),
            노출수 INT,
            유입수 INT,
            `유입률(%)` FLOAT,
            `체류시간(min)` INT,
            `평균체류시간(min)` FLOAT,
            페이지뷰 INT,
            평균페이지뷰 FLOAT,
            이탈수 INT,
            `이탈률(%)` FLOAT,
            회원가입 INT,
            `전환율(가입)` FLOAT,
            `앱 다운` INT,
            `전환율(앱)` FLOAT,
            구독 INT,
            `전환율(구독)` FLOAT,
            총전환수 INT,
            총전환율 FLOAT
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        print(f"'{DB_TABLE}' 테이블이 성공적으로 생성되었습니다.")
    except Error as e:
        print(f"테이블 생성 중 오류 발생: {e}")
    finally:
        cursor.close()


def insert_data(connection, df):
    try:
        cursor = connection.cursor()
        insert_query = f"""
        INSERT INTO `{DB_TABLE}` (
            날짜, 디바이스, 유입경로, 키워드, 노출수, 유입수, `유입률(%)`, 
            `체류시간(min)`, `평균체류시간(min)`, 페이지뷰, 평균페이지뷰, 이탈수, `이탈률(%)`, 회원가입, `전환율(가입)`,
            `앱 다운`, `전환율(앱)`, 구독, `전환율(구독)`, 총전환수, 총전환율
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        data = [tuple(row) for row in df.values]
        cursor.executemany(insert_query, data)
        connection.commit()
        print(f"{cursor.rowcount}개의 레코드가 '{DB_TABLE}' 테이블에 삽입되었습니다.")
    except Error as e:
        print(f"데이터 삽입 중 오류 발생: {e}")
    finally:
        cursor.close()


def main():
    # CSV 파일 읽기 (index_col=0으로 첫 번째 열을 인덱스로 설정)
    try:
        df = pd.read_csv(CSV_FILE_PATH, index_col=0, encoding="cp949")
        print("CSV 파일을 성공적으로 읽었습니다.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    except FileNotFoundError:
        print(f"파일 '{CSV_FILE_PATH}'을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"CSV 파일 읽기 중 오류 발생: {e}")
        return

    # 열 이름 조정
    expected_columns = [
        "날짜",
        "디바이스",
        "유입경로",
        "키워드",
        "노출수",
        "유입수",
        "유입률(%)",
        "체류시간(min)",
        "평균체류시간(min)",
        "페이지뷰",
        "평균페이지뷰",
        "이탈수",
        "이탈률(%)",
        "회원가입",
        "전환율(가입)",
        "앱 다운",
        "전환율(앱)",
        "구독",
        "전환율(구독)",
        "전환수",
        "전환율",
    ]
    df.columns = expected_columns  # 열 이름 매핑

    # 데이터베이스 연결
    connection = create_database_connection()
    if connection is None:
        return
    # 테이블 생성
    create_table(connection)

    # 데이터 삽입
    insert_data(connection, df)

    # 연결 종료
    if connection.is_connected():
        connection.close()
        print("MySQL 연결이 종료되었습니다.")


if __name__ == "__main__":
    main()


# SQLAlchemy 엔진 생성 (MySQL용)
engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# 데이터 가져오기 함수
def fetch_data():
    query = "SELECT * FROM onlinetbl;"
    df = pd.read_sql(query, engine)  # SQLAlchemy 엔진 사용
    return df


# Streamlit 앱
st.title("환경 캠패인 Online 테이블 조회")

if st.button("데이터 블러오기"):
    df = fetch_data()
    st.dataframe(df.set_index("id"))

    # st.dataframe(df)  # 데이터 출력
