import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import seaborn as sns

#Metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

#Model Select
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

st.write("""
# Prediksi Breast Cancer
### KNN, Random Forest, Decicion Tree, Gaussian Naive Bayes
"""
)

img = Image.open('cancer.jpg')
st.image(img, use_column_width=False)

st.sidebar.write("""
            # Penjelasan Untuk Pengisi Form"""
            )
st.sidebar.write("""
            ####  1. Usia: Diisi dengan angka usia calon pasien yang akan di prediksi
            """)
st.sidebar.write("""
            ####  2. BMI: Diisi dengan jumlah BMI yang ada dalam anda. Angka BMI normal berada pada kisaran 18,5-25.
            """)
st.sidebar.write("""
            ####  3. Glukosa: Diisi dengan jumlah glukosa dalam tubuh anda. Glukosa Normal berkisar < 100mg/dL, jika berpuasa 70-130 mg/dL, < 180 mg/dL (setelah makan), 100-140 mg/dL (sebelum tidur)
            """)
st.sidebar.write("""
            ####  4. Insulin: Diisi dengan jumlah insulin dalam tubuh anda. Insulin normal berkisar di bawah 100 mg/dL.
            """)
st.sidebar.write("""
            ####  5. HOMA: Diisi dengan jumlah HOMA dalam tubuh anda. homeostasis model aseessment (HOMA)
            """)
st.sidebar.write("""
            ####  6. Leptin: Diisi dengan jumlah Leptin dalam tubuh anda. Leptin adalah suatu protein yang berasal dari 167 asam amino,merupakan hormon yang di produksi oleh jaringan adiposa. Biasa ditentukan dalam bentuk (ng/mL)
            """)
st.sidebar.write("""
            ####  7. Adiponectin: Diisi dengan jumlah Adiponectin dalam tubuh anda.
            """)           
st.sidebar.write("""
            ####  8. Resistin: Diisi dengan jumlah resistin dalam tubuh anda. Biasa ditentukan dalam bentuk (ng/mL)
            """)
st.sidebar.write("""
            ####  9. MCP: Diisi dengan jumlah MCP dalam tubuh anda. MCP (Monocyte Chemoattracttant Protein-1). Biasa ditentukan dalam bentuk (pg/dL)
            """)
st.sidebar.write("""
            ####  10. Setelah semuanya terisi silahkan klik prediksi untuk mengetahui hasil dari prediksi tersebut
            """)


tab_titles = [
    "Akurasi",
    "Identifikasi Penyakit",
    "Preprocessing Data",
    "Github dan Dataset",]

tabs = st.tabs(tab_titles)

with tabs[0]:
    cancer = pd.read_csv('https://raw.githubusercontent.com/DiahDSyntia/breastcancer/main/dataR2.csv')

    #cancer['Classification'].unique()
    
    X=cancer.iloc[:,0:9].values 
    y=cancer.iloc[:,9].values

    st.write('Jumlah baris dan kolom :', X.shape)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    #split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    #st.write("Data Training", X_train)
    #st.write("Data Testing", X_test)

    #KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    Y_pred = knn.predict(X_test) 
    accuracy_knn=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_knn = round(knn.score(X_train, y_train) * 100, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for KNN\n',cm)
    print('accuracy_KNN : %.3f' %accuracy)
    print('precision_KNN : %.3f' %precision)
    print('recall_KNN: %.3f' %recall)
    print('f1-score_KNN : %.3f' %f1)

    #NAIVE BAYES
    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    Y_pred = gaussian.predict(X_test) 
    accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for Naive Bayes\n',cm)
    print('accuracy_Naive Bayes: %.3f' %accuracy)
    print('precision_Naive Bayes: %.3f' %precision)
    print('recall_Naive Bayes: %.3f' %recall)
    print('f1-score_Naive Bayes : %.3f' %f1)

    #DECISION TREE
    decision_tree = DecisionTreeClassifier() 
    decision_tree.fit(X_train, y_train)  
    Y_pred = decision_tree.predict(X_test) 
    accuracy_dt=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for DecisionTree\n',cm)
    print('accuracy_DecisionTree: %.3f' %accuracy)
    print('precision_DecisionTree: %.3f' %precision)
    print('recall_DecisionTree: %.3f' %recall)
    print('f1-score_DecisionTree : %.3f' %f1)

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    Y_prediction = random_forest.predict(X_test)
    accuracy_rf=round(accuracy_score(y_test,Y_prediction)* 100, 2)
    acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

    cm = confusion_matrix(y_test, Y_prediction)
    accuracy = accuracy_score(y_test,Y_prediction)
    precision =precision_score(y_test, Y_prediction,average='micro')
    recall =  recall_score(y_test, Y_prediction,average='micro')
    f1 = f1_score(y_test,Y_prediction,average='micro')
    print('Confusion matrix for Random Forest\n',cm)
    print('accuracy_random_Forest : %.3f' %accuracy)
    print('precision_random_Forest : %.3f' %precision)
    print('recall_random_Forest : %.3f' %recall)
    print('f1-score_random_Forest : %.3f' %f1)

    st.write("""
                #### Akurasi:"""
                )

    results = pd.DataFrame({
        'Model': ['K-Nearest Neighbor','Naive Bayes','Decision Tree','Random Forest'],
        'Score': [ acc_knn,acc_gaussian,acc_decision_tree, acc_random_forest ],
        "Accuracy_score":[accuracy_knn,accuracy_nb,accuracy_dt,accuracy_rf
                        ]})
    result_df = results.sort_values(by='Accuracy_score', ascending=False)
    result_df = result_df.reset_index(drop=True)
    result_df.head(9)
    st.write(result_df)

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(['K-Nearest Neighbor','Naive Bayes','Decision Tree','Random Forest'],[accuracy_knn,accuracy_nb,accuracy_dt,accuracy_rf])
    plt.show()
    st.pyplot(fig)


with tabs[1]:
    col1,col2 = st.columns([2,2])
    model=st.selectbox(
            'Metode Prediksi', ('K-Nearest Neighbor','Naive Bayes','Decision Tree','Random Forest'))
    with col1:
        usia = st.number_input("Usia",0)
        bmi = st.number_input("BMI",0.00)
        glukosa = st.number_input("Glukosa",0)
        insulin = st.number_input("Insulin",0.00)
    with col2:
        homa = st.number_input("HOMA",0.00)
        leptin = st.number_input("Leptin",0.00)
        adiponectin = st.number_input("Adiponectin",0.00)
        resistin = st.number_input("Resistin",0.00)
        mcp = st.number_input("MCP.1",0.00)
    submit = st.button('Prediksi')

    if submit:
        if model == 'K-Nearest Neighbor':
            X_new = np.array([[usia, bmi, glukosa, insulin, homa, leptin, adiponectin, resistin, mcp]])
            predict = knn.predict(X_new)
            if predict == 1 :
                st.write("""# Anda Negative Breast Cancer""")
            else : 
                st.write("""# Anda Positive Breast Cancer, Segera Ke Dokter""")

        elif model == 'Naive Bayes':
            X_new = np.array([[usia,bmi,glukosa,insulin, homa, leptin, adiponectin, resistin, mcp]])
            predict = gaussian.predict(X_new)
            if predict == 1 :
                st.write("""# Anda Negative Breast Cancer""")
            else : 
                st.write("""# Anda Positive Breast Cancer, Segera Ke Dokter""")

        elif model == 'Decision Tree':
            X_new = np.array([[usia, bmi, glukosa, insulin, homa, leptin, adiponectin, resistin, mcp]])
            predict = decision_tree.predict(X_new)
            if predict == 1 :
                st.write("""# Anda Negative Breast Cancer""")
            else : 
                st.write("""# Anda Positive Breast Cancer, Segera Ke Dokter""")

        else:
            X_new = np.array([[usia, bmi, glukosa, insulin, homa, leptin, adiponectin, resistin, mcp]])
            predict = random_forest.predict(X_new)
            if predict == 1 :
                st.write("""# Anda Negative Breast Cancer""")
            else : 
                st.write("""# Anda Positive Breast Cancer, Segera Ke Dokter""")

with tabs[2]:
    st.write("Jumlah Data Training:", len(X_train))
    st.write("Jumlah Data Testing:", len(X_test))
    st.write("Data Cancer (https://raw.githubusercontent.com/DiahDSyntia/Data-Mining/main/dataR2.csv) ",cancer)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    st.write("Hasil Preprocesing : ", scaled)
    
    st.write("Data Training", X_train)
    st.write("Data Testing", X_test)

with tabs[3]:
    dataset = f"https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra"
    github = f"https://github.com/DiahDSyntia/breastcancer"
    st.write(f"Dataset yang digunakan dalam web ini adalah dataset yang diambil dari situs UCI Machine Learning Repository. [Klik Disini Untuk Dataset]({dataset}).")
    st.write(f"Untuk Code dan Repository web ini bisa dilihat pada github saya. [Klik Disini Untuk Github]({github}).")
