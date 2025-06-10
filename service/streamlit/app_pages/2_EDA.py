import base64
import logging
import math

import client_funcs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots
from settings import Settings

settings = Settings()


def pdf_clicked():
    '''
    переход на страницу с PDF-файлом
    '''
    st.session_state.eda_type = 1


def realtime_clicked():
    '''
    переход на страницу с real-time EDA
    '''
    st.session_state.eda_type = 2


def back_clicked():
    '''
    возвращение на стартовую страницу
    '''
    st.session_state.eda_type = 0
    st.cache_data.clear()


def bars_and_donut(df, col, h=500, w=800):
    '''
    распределене целевого признака
    '''
    fig = make_subplots(rows=1, cols=2, specs=[
                        [{"type": "domain"}, {"type": "xy"}]])
    x = df[col].value_counts(sort=False).index.tolist()
    y = df[col].value_counts(sort=False).tolist()
    fig.add_trace(
        go.Pie(
            values=y,
            labels=x,
            hole=0.3,
            textinfo="label+percent",
            title=f"Признак: {col}",
            marker=dict(colors=["darkturquoise", "darkgoldenrod"]),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            text=y,
            textposition="outside",
            marker_color=["darkturquoise", "darkgoldenrod"],
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        height=h,
        width=w,
        showlegend=False,
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        title=dict(
            text=f"Распределение целевого признака {col}", x=0.5, y=0.95),
    )
    return fig


@st.cache_data
def hist(df, cols, bins, ncols=3):
    '''
    распределение остальных признаков (для трейна и теста)
    '''
    nrows = math.ceil(len(cols) / ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(
        5 * ncols, 4.2 * nrows), sharey=False)
    for i in range(len(cols)):
        sns.histplot(
            data=df,
            x=cols[i],
            bins=bins,
            hue="dataset",
            palette=["indigo", "grey"],
            ax=ax[i // ncols, i % ncols],
        )
        ax[i // ncols, i % ncols].set_xlabel(cols[i])
        if i % ncols != 0:
            ax[i // ncols, i % ncols].set_ylabel(" ")
    plt.tight_layout()
    return fig


@st.cache_data
def hist_target(df, cols, target, bins, ncols=3):
    '''
    распределение признаков в зависимости от целевой переменной
    '''
    nrows = math.ceil(len(cols) / ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(
        5 * ncols, 4.2 * nrows), sharey=False)
    for i in range(len(cols)):
        sns.histplot(
            data=df,
            x=cols[i],
            bins=bins,
            hue=target,
            palette=["darkturquoise", "darkgoldenrod"],
            ax=ax[i // ncols, i % ncols],
        )
        ax[i // ncols, i % ncols].set_xlabel(cols[i])
        if i % ncols != 0:
            ax[i // ncols, i % ncols].set_ylabel(" ")
    plt.tight_layout()
    return fig


def donut_custom(df1, df2, col, text1, text2, title_text, h, w):
    '''
    распределение целочисленных признаков в зависимости
    от целевой переменной
    '''
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.5],
        specs=[[{"type": "pie"}, {"type": "pie"}]],
    )
    fig.add_trace(
        go.Pie(
            labels=df1[col].value_counts().index,
            values=df1[col].value_counts(),
            legendgroup="group",
            textinfo="percent",
            hole=0.3,
            title=dict(text=text1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Pie(
            labels=df2[col].value_counts().index,
            values=df2[col].value_counts(),
            legendgroup="group",
            textinfo="percent",
            hole=0.3,
            title=dict(text=text2),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        height=h,
        width=w,
        title=dict(text=title_text, y=0.9, x=0.5,
                   xanchor="center", yanchor="top"),
    )
    return fig


def start_page(placeholder):
    '''
    стартовая страница
    '''
    with placeholder.container():
        st.write("Выберите тип отображения разведочного анализа данных.")
        st_cols = st.columns(2)
        st_cols[0].button("Скачать PDF", on_click=pdf_clicked,
                          use_container_width=True)
        st_cols[1].button(
            "EDA в реальном времени",
            on_click=realtime_clicked,
            use_container_width=True,
        )


def pdf_page(placeholder):
    '''
    страница для скачивания PDF-файла
    '''
    with placeholder.container():
        st_cols = st.columns(3)
        st_cols[1].button("Назад", on_click=back_clicked,
                          use_container_width=True)
        response = client_funcs.get_pdf()
        pdf = base64.b64encode(response.content).decode("utf-8")
        pdf_display = f'<div style="text-align:center">\
            <iframe src="data:application/pdf;base64,\
            {pdf}" width="600" height="750" type="application/pdf"></iframe>\
            </div>'
        st.markdown(pdf_display, unsafe_allow_html=True)


@st.cache_data
def read_train_test(train_csv, test_csv):
    '''
    чтение датасетов из файлов
    '''
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    return train_df, test_df


@st.cache_data
def get_df_data(train_df, test_df):
    '''
    получение датасета с количеством
    событий, дубликатов и проусков
    '''
    df_obs = pd.DataFrame(
        index=["Количество событий"],
        columns=["Тренировочный набор", "Тестовый набор"]
    )
    df_obs["Тренировочный набор"] = len(train_df)
    df_obs["Тестовый набор"] = len(test_df)
    df_duplicate_rows = pd.DataFrame(
        index=["Количество дубликатов"],
        columns=["Тренировочный набор", "Тестовый набор"],
    )
    df_duplicate_rows["Тренировочный набор"] = train_df.duplicated().sum()
    df_duplicate_rows["Тестовый набор"] = test_df.duplicated().sum()
    df_missing = pd.DataFrame(
        index=["Количество пропущеных значений"],
        columns=["Тренировочный набор", "Тестовый набор"],
    )
    df_missing["Тренировочный набор"] = len(
        train_df.isna().sum()[train_df.isna().sum() != 0]
    )
    df_missing["Тестовый набор"] = len(
        test_df.isna().sum()[test_df.isna().sum() != 0])
    return pd.concat([df_obs, df_duplicate_rows, df_missing],
                     ignore_index=False)


@st.cache_data
def get_merged_df(train_df, test_df):
    '''
    слияние тренировочного и тестового датасетов
    '''
    df_train_temp = train_df.copy(deep=True)
    df_train_temp["dataset"] = "train"
    df_test_temp = test_df.copy(deep=True)
    df_test_temp["dataset"] = "test"
    df_all = pd.concat([df_train_temp, df_test_temp],
                       axis=0, ignore_index=True)
    return df_all


def draw_plots(train_df, test_df, target_col):
    '''
    построение графиков
    '''
    st.header("Дубликаты и пропущенные значения")
    st.dataframe(get_df_data(train_df, test_df), use_container_width=True)
    st.header("Описательные характеристики данных")
    st.subheader("Тренировочный датасет:")
    st.dataframe(train_df.describe(), use_container_width=True)
    st.subheader("Тестовый датасет:")
    st.dataframe(test_df.describe(), use_container_width=True)
    st.header("Анализ целевого признака")
    st.write(
        """
             Целевой признак `Label` является \
             бинарным и принимает значения `b` или \
             `s`, указывающих на природу события (фоновое или сигнал).
             """
    )
    st.plotly_chart(
        bars_and_donut(train_df, target_col),
        use_container_width=True,
        theme="streamlit",
    )
    st.header("Распределение остальных признаков")
    df_all = get_merged_df(train_df, test_df)
    st.pyplot(
        hist(
            df_all.replace(-999, np.nan),
            list(train_df.columns[train_df.dtypes == "float64"]),
            bins=max(
                math.floor(len(train_df) ** (1 / 3)),
                math.floor(len(test_df) ** (1 / 3)),
            ),
            ncols=3,
        )
    )
    st.header(
        """
              Сравнение распределений признаков
              по целевому классу в обучающей выборке
        """
    )
    st.markdown(
        """
                Далее мы сравниваем одномерные распределения признаков
                для фоновых событий и сигнальных событий в обучающем наборе.

                Если признак имеет разные распределения для фоновых и
                сигнальных событий, то это значит, что признак важен в
                задаче классификации событий, когда метка неизвестна.

                Аналогично, если признак имеет очень похожие распределения
                для двух целевых классов, то он вряд ли поможет в задаче
                классификации.
                """
    )
    df_train_b = train_df[train_df[target_col] == "b"]
    df_train_s = train_df[train_df[target_col] == "s"]
    st.pyplot(
        hist_target(
            train_df.replace(-999, np.nan),
            list(test_df.columns[test_df.dtypes == "float64"]),
            target="Label",
            bins=max(
                math.floor(len(df_train_b) ** (1 / 3)),
                math.floor(len(df_train_s) ** (1 / 3)),
            ),
            ncols=3,
        )
    )
    st.header(
        """
        Распределение целочисленных признаков \
        (количество струй PRI_jet_num)
        """
    )
    st.plotly_chart(
        donut_custom(
            train_df,
            test_df,
            col="PRI_jet_num",
            title_text="PRI_jet_num",
            text1="Тренировочная выборка",
            text2="Тестовая выборка",
            h=600,
            w=1000,
        )
    )
    st.header(
        """
        Частота распределения PRI_jet_num в тренировочном \
        наборе в соотвествии с целевой переменной
        """
    )
    st.plotly_chart(
        donut_custom(
            df_train_b,
            df_train_s,
            col="PRI_jet_num",
            title_text="PRI_jet_num",
            text1="Фоновые события",
            text2="Сигнальные события",
            h=600,
            w=1000,
        )
    )


def realtime_page(placeholder):
    '''
    страница с real-time EDA
    '''
    response = client_funcs.get_columns()
    df_cols_data = response.json()
    with placeholder.container():
        st_cols = st.columns(3)
        st_cols[1].button("Назад", on_click=back_clicked,
                          use_container_width=True)
        st.markdown(
            f"""
            Загрузите любые данные. Датасеты, используемые \
            для обучения моделей в рамках проекта, можно \
            скачать из [репозитория]({settings.GITHUB_URL}) \
            (в формате ZIP).
            """
        )
        train_csv = st.file_uploader(
            "Загрузите тренировочный датасет", type=["csv"])
        test_csv = st.file_uploader("Загрузите тестовый датасет", type=["csv"])
        if train_csv is None or test_csv is None:
            st.error("Загрузите и тренировочный, и тестовый датасеты!")
        else:
            train_df, test_df = read_train_test(train_csv, test_csv)
            check_train = client_funcs.check_dataset(train_df, df_cols_data)
            check_test = client_funcs.check_dataset(
                test_df, df_cols_data, "test")
            if not check_train or not check_test:
                if not check_train:
                    st.error("Ошибка данных в тренировочном датасете!")
                if not check_test:
                    st.error("Ошибка данных в тестовом датасете!")
            else:
                draw_plots(train_df, test_df, df_cols_data["target"])
                st.markdown(
                    """
                    **Примечание**: разведочный анализ данных для \
                    датасетов, используемых в проекте, также \
                    доступен в этом Streamtil-приложении \
                    в виде файла PDF.
                    """
                )


logging.info("EDA opened")

# переменная с текущим типом EDA (PDF или real-time EDA)
if "eda_type" not in st.session_state:
    st.session_state.eda_type = 0

placeholder = st.empty()

if st.session_state.eda_type == 0:
    logging.info('Start page opened')
    start_page(placeholder)
elif st.session_state.eda_type == 1:
    logging.info('PDF page opened')
    pdf_page(placeholder)
elif st.session_state.eda_type == 2:
    logging.info('Real-tile EDA page opened')
    realtime_page(placeholder)
