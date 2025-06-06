import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler

def load_data():
    return pd.read_csv("cleaned_city_air.csv")

def model_training_page():
    st.title("SVM: Классификация безопасности воздуха")
    
    # Загрузка данных
    df = load_data()
    
    # Показываем первые 5 строк датасета
    st.subheader("Просмотр данных")
    st.write("Первые 5 строк датасета:")
    st.dataframe(df.head())
    
    # Разделение выборки (удаляем AQI)
    X = df.drop(columns=["City", "Date", "Air_Safe", "AQI"])  # Исключаем нечисловые, целевой признак и AQI
    y = df["Air_Safe"]
    
    # Разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    
    # Масштабирование данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Настройка гиперпараметров
    st.subheader("Настройка гиперпараметров SVM")
    col1, col2 = st.columns(2)
    
    with col1:
        kernel = st.selectbox("Ядро", options=["linear", "poly", "rbf", "sigmoid"])
    
    with col2:
        C = st.slider("Параметр регуляризации (C)", 
                      min_value=0.01, max_value=10.0, 
                      value=1.0, step=0.01)
    
    # Обучение модели
    model = SVC(kernel=kernel, C=C, probability=True, random_state=1)
    model.fit(X_train_scaled, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Расчет метрик
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    # Отображение метрик
    metrics_data = {
        "Метрика": ["Точность (Accuracy)", "Точность (Precision)", "Полнота (Recall)", "F1-мера"],
        "Значение": [
            f"{accuracy * 100:.2f}%",
            f"{precision * 100:.2f}%",
            f"{recall * 100:.2f}%",
            f"{f1 * 100:.2f}%",
        ],
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    st.subheader("Оценка качества модели")
    st.table(metrics_df.set_index("Метрика"))
    
    # Визуализация важности признаков (коэффициенты для линейного ядра)
    if kernel == "linear":
        st.subheader("Важность признаков (коэффициенты SVM)")
        feature_importance = pd.DataFrame({
            "Признак": X.columns,
            "Важность": model.coef_[0]
        }).sort_values("Важность", ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Важность", y="Признак", data=feature_importance, ax=ax)
        ax.set_title("Важность признаков в модели")
        st.pyplot(fig)
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Матрица ошибок")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Небезопасно", "Безопасно"],
                yticklabels=["Небезопасно", "Безопасно"])
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    st.pyplot(fig)
    
    # ROC-кривая
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    st.subheader("ROC-кривая")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC кривая (площадь = {roc_auc:.2f})"
    )
    ax2.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC-кривая")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

def main():
    model_training_page()

if __name__ == "__main__":
    main()