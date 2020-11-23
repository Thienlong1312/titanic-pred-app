import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,\
classification_report,roc_auc_score,roc_curve
import base64
from PIL import Image

############################
def xu_ly_du_lieu(data):
    data=data.interpolate()
    data=data.drop(['cabin','home.dest'],axis=1)
    data=data.dropna()
    data=pd.get_dummies(data,columns=['sex','embarked'],drop_first=True)

    #Chọn những thuộc tính quan trọng cần trong việc xây dựng model và dự đoán
    data=data[['pclass','sex_male','age','sibsp','parch',
                'fare','embarked_Q','embarked_S','survived']]
    return data

def chuan_doan_du_lieu(data,model):
    X=data[['pclass','sex_male','age','sibsp','parch','fare','embarked_Q','embarked_S']]
    y=data['survived']
    test_size=.2

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)

    probs_train=model.predict_proba(X_train)[:,1]
    probs_test=model.predict_proba(X_test)[:,1]
    fpr_train,tpr_train,thresholds_train=roc_curve(y_train,probs_train)
    fpr_test,tpr_test,thresholds_test=roc_curve(y_test,probs_test)
    st.write('Đối với dữ liệu training:')
    st.write('* Accuracy score: %.4f'%accuracy_score(y_train,model.predict(X_train)))
    st.write('* AUROC: %.4f'%roc_auc_score(y_train,probs_train))
    st.write('* Confusion matrix:')
    df_a=pd.DataFrame(confusion_matrix(y_train,model.predict(X_train)))
    col_ix=pd.MultiIndex.from_product([['Predicted'],['"Not survived"','"Survived"']])
    row_ix=pd.MultiIndex.from_product([['Actual'],['"Not survived"','"Survived"']])
    df_a.index=row_ix
    df_a.columns=col_ix
    st.dataframe(df_a)
    st.write('* Classification report:')
    st.dataframe(pd.DataFrame(classification_report(y_train,model.predict(X_train),output_dict=True)).\
    transpose().rename(index={'0':'"Not survived"','1':'"Survived"'}))

    st.write('Đối với dữ liệu testing:')
    st.write('* Accuracy score: %.4f'%accuracy_score(y_test,model.predict(X_test)))
    st.write('* AUROC: %.4f'%roc_auc_score(y_test,probs_test))
    st.write('* Confusion matrix:')
    df_b=pd.DataFrame(confusion_matrix(y_test,model.predict(X_test)))
    col_ix=pd.MultiIndex.from_product([['Predicted'],['"Not survived"','"Survived"']])
    row_ix=pd.MultiIndex.from_product([['Actual'],['"Not survived"','"Survived"']])
    df_b.index=row_ix
    df_b.columns=col_ix
    st.dataframe(df_b)
    st.write('* Classification report:')
    st.dataframe(pd.DataFrame(classification_report(y_test,model.predict(X_test),output_dict=True)).\
    transpose().rename(index={'0':'"Not survived"','1':'"Survived"'}))
    return X_train,X_test,y_train,y_test,fpr_train,tpr_train,fpr_test,tpr_test,probs_train,probs_test

def ve_bieu_do(model,X_train,X_test,y_train,y_test,fpr_train,tpr_train,\
                    fpr_test,tpr_test,probs_train,probs_test):
    fig, ax = plt.subplots(figsize=(12,12))

    plt.subplot(221)
    sns.kdeplot(y_train,label='actual',color='b')
    sns.kdeplot(model.predict(X_train),label='predict',color='r')
    plt.title('Training data')

    plt.subplot(222)
    sns.kdeplot(y_test,label='actual',color='b')
    sns.kdeplot(model.predict(X_test),label='predict',color='r')
    plt.title('Testing data')

    plt.subplot(223)
    plt.plot([0,1],[0,1],linestyle='--')
    plt.plot(fpr_train,tpr_train,marker='.')
    plt.title('ROC Curve - Training')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.subplot(224)
    plt.plot([0,1],[0,1],linestyle='--')
    plt.plot(fpr_test,tpr_test,marker='.')
    plt.title('ROC Curve - Testing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    st.pyplot(fig)

    st.write('Nhận xét:')
    st.write('* Chỉ số Accuracy score cao ở cả training và testing, chỉ số AUROC ở testing đạt %.3f'\
                %roc_auc_score(y_test,probs_test))
    st.write('* Model phù hợp với dữ liệu')
    st.write('* Có thể sử dụng model để dự báo cho các hành khách chưa biết khác')
    return
############################

############################
def gioi_tinh():
    #Giới tính: Nam, Nữ --> sex_male
    gender=st.radio('- *Giới tính người đó:',['Nam','Nữ'])
    if gender=='Nữ':
        SEX_MALE=0
    else:
        SEX_MALE=1
    return gender,SEX_MALE

def tuoi():
    #Tuổi --> 0-80
    age=st.text_input('- *Nhập tuổi:')
    AGE=1
    if age!='':
        if age.isdigit():
            if int(age)>=1 and int(age)<=100:
                st.success('Ok!')
                AGE=int(age)
            else:
                st.warning('Tuổi nằm trong khoảng từ 1 đến 100')
        else:
            st.warning('Chỉ nhập số!')
    else:
        st.write('Chỉ nhập số')
    return AGE

def sibsp():
    #Có vợ/chồng đi cùng không
    #Có người thân đi cùng không
    st.write('*Đánh dấu tích nếu có')
    VO_CHONG=st.checkbox('Có vợ/chồng đi cùng')
    nguoi_than=st.checkbox('Có anh chị em ruột đi cùng')
    if nguoi_than==True:
        NGUOI_THAN=st.slider('*Mấy người đi cùng:',min_value=0,max_value=8,step=1)
    else:
        NGUOI_THAN=0
    return VO_CHONG,NGUOI_THAN

def parch():
    #Bố mẹ có đi cùng không
    #Có dẫn trẻ em đi cùng không
    BO=st.checkbox('Có bố đi cùng')
    ME=st.checkbox('Có mẹ đi cùng')
    tre_em=st.checkbox('Có dẫn trẻ em đi cùng')
    if tre_em==True:
        TRE_EM=st.slider('*Mấy trẻ em đi cùng:',min_value=0,max_value=8,step=1)
    else:
        TRE_EM=0
    return BO,ME,TRE_EM

def pclass():
    #Mua vé thuộc hạng nào
    ve=st.radio('- *Mua vé thuộc hạng nào:',['1st','2nd','3rd'])
    if ve=='1st':
        PCLASS=1
    elif ve=='2nd':
        PCLASS=2
    else:
        PCLASS=3
    return ve,PCLASS

def fare():
    #Giá vé bao nhiêu (Tính theo tiền đô) --> 0-512
    FARE=st.slider('*Giá vé bao nhiêu:',min_value=0,max_value=512,step=1)
    return FARE

def Embarked():
    #Lên tàu ở cảng nào
        #Cảng Southampton --> [0,1]
        #Cảng Queenstown  --> [1,0]
        #Cảng Cherbourg   --> [0,0]
    embarked=st.selectbox('- *Lên tàu ở cảng nào:',
    ['Cảng Queenstown','Cảng Cherbourg','Cảng Southampton'])
    if embarked=='Cảng Queenstown':
        EMBARKED_Q=1
        EMBARKED_S=0
    elif embarked=='Cảng Cherbourg':
        EMBARKED_Q=0
        EMBARKED_S=0
    else:
        EMBARKED_Q=0
        EMBARKED_S=1
    return embarked,EMBARKED_Q,EMBARKED_S

def Name():
    # Họ tên nếu có
    name=st.text_input('- Họ tên (Nếu có):')
    return name

def Quan_he():
    #Mối quan hệ với người dùng
    quan_he=st.radio('- Người đó có quan hệ gì với bạn',
    ['Là chính bạn','Người quen/bạn bè...','Không có quan hệ gì'])
    return quan_he

def nhap_thong_tin():
    gender,SEX_MALE=gioi_tinh()
    AGE=tuoi()

    VO_CHONG,NGUOI_THAN=sibsp()
    #--> Cộng tổng lại xem có bao nhiêu người --> sibsp (0-8)
    SIBSP=VO_CHONG+NGUOI_THAN

    BO,ME,TRE_EM=parch()
    #--> Cộng tổng lại xem có bao nhiêu người --> parch (0-9)
    PARCH=BO+ME+TRE_EM

    ve,PCLASS=pclass()
    FARE=fare()
    embarked,EMBARKED_Q,EMBARKED_S=Embarked()

    #Tên
    name=Name()
    #Mối quan hệ
    quan_he=Quan_he()
    return gender,SEX_MALE,AGE,VO_CHONG,NGUOI_THAN,\
           SIBSP,BO,ME,TRE_EM,PARCH,ve,PCLASS,FARE,embarked,\
           EMBARKED_Q,EMBARKED_S,name,quan_he
############################

############################
def thong_tin_huu_ich():
    list_a=['Bạn có biết: ','Có thể bạn chưa biết: ','Thông tin hữu ích: ','Sự thật thú vị rằng: ']
    list_b=['Trên chuyến tàu Titanic, chỉ có khoảng 20% nam giới sống sót và nữ lên tới 75%!',
            'Giới tính quyết định đến 25% trong việc dự đoán của model',
            'Tuổi quyết định đến tận 32% trong việc dự đoán và chiếm vị trí Thuộc tính quan trọng nhất!!',
            'Giá vé chiếm tận 25% độ quan trọng khi dự đoán!!',
            'Chỉ có khoảng 6% hành khách mua vé với giá trên 100$ và 70% trong số họ sống sót!',
            'Nếu bạn là nam và bạn chịu mua vé với giá trên 100$, bạn có tới tận 32% cơ hội sống sót!!!',
            'Thảm họa Titanic đã cướp đi sinh mạng gần 2/3 số hành khách trên tàu chỉ 1/3 số hành khách sống sót',
            'Đa số hành khách mua vé với giá dưới 100$, và chỉ 35% trong số họ sống sót',
            'Nếu bạn là nam và bạn mua vé rẻ (dưới 100$), xin chia buồn, bạn chỉ có khoảng 18% cơ hội sống sót :(',
            'Khoảng 60% trẻ em dưới 10 tuổi và khoảng 45% người già trong độ tuổi 50-60 được cứu sống',
            'Độ tuổi từ 20 đến 45 tuổi là bấp bênh nhất, chỉ có 35% cơ hội sống sót',
            'Nếu bạn là nam và tuổi của bạn trong khoảng 20 đến 40, bạn sẽ phải 1 chọi 6 nếu muốn có cơ hội sống sót :(',
            'Nhóm nữ có cơ hội sống sót thấp nhất theo phân tích là nhóm nữ có độ tuổi 25-30, mua vé dưới 100$ với 63% cơ hội sống sót',]
    return list_a[np.random.randint(len(list_a))]+\
             list_b[np.random.randint(len(list_b))]

def speak_sorry(name,gender,quan_he):
    list_sorry_1=['Rất tiếc! ','Xin chia buồn! ','Tin buồn! ']
    list_sorry_2=[' không thể sống sót sau vụ Titanic :(',' sẽ hi sinh anh dũng :(',' sẽ ra đi :(']
    if quan_he=='Là chính bạn':
        speak=list_sorry_1[np.random.randint(len(list_sorry_1))]+'Bạn'+list_sorry_2[np.random.randint(len(list_sorry_2))]
    else:
        if name=='':
            if quan_he=='Người quen':
                if gender=='Nam':
                    list_name=['Người bạn quen biết','Họ','Anh ấy']
                else:
                    list_name=['Người bạn quen biết','Họ','Cô ấy']
                speak=list_sorry_1[np.random.randint(len(list_sorry_1))]+list_name[np.random.randint(len(list_name))]+list_sorry_2[np.random.randint(len(list_sorry_2))]
            else:
                list_sorry_2=[' không thể sống sót sau vụ Titanic :(',' rất khó để sống sót :(']
                if gender=='Nam':
                    list_name=['Người này','Hành khách này','Họ','Anh ấy']
                else:
                    list_name=['Người này','Hành khách này','Họ','Cô ấy']
                speak=list_sorry_1[np.random.randint(len(list_sorry_1))]+list_name[np.random.randint(len(list_name))]+list_sorry_2[np.random.randint(len(list_sorry_2))]
        else:
            speak=list_sorry_1[np.random.randint(len(list_sorry_1))]+name.capitalize()+list_sorry_2[np.random.randint(len(list_sorry_2))]

    return speak

def speak_good(name,gender,quan_he):
    list_good_1=['Rất may mắn! ','Tuyệt vời! ','Tin tốt! ']
    list_good_2=[' sẽ sống sót sau vụ Titanic!',' sẽ sống sót trở về!',' sẽ còn nguyên vẹn trở về sau chuyến đi!']
    if quan_he=='Là chính bạn':
        speak=list_good_1[np.random.randint(len(list_good_1))]+'Bạn'+list_good_2[np.random.randint(len(list_good_2))]
    else:
        if name=='':
            if quan_he=='Người quen':
                if gender=='Nam':
                    list_name=['Người bạn quen biết','Họ','Anh ấy']
                else:
                    list_name=['Người bạn quen biết','Họ','Cô ấy']
                speak=list_good_1[np.random.randint(len(list_good_1))]+list_name[np.random.randint(len(list_name))]+list_good_2[np.random.randint(len(list_good_2))]
            else:
                list_good_2=[' có khả năng sống sót cao trong vụ Titanic!',' có thể sống sót trở về!']
                if gender=='Nam':
                    list_name=['Người này','Hành khách này','Họ','Anh ấy']
                else:
                    list_name=['Người này','Hành khách này','Họ','Cô ấy']
                speak=list_good_1[np.random.randint(len(list_good_1))]+list_name[np.random.randint(len(list_name))]+list_good_2[np.random.randint(len(list_good_2))]
        else:
            speak=list_good_1[np.random.randint(len(list_good_1))]+name.capitalize()+list_good_2[np.random.randint(len(list_good_2))]
    return speak

def thong_bao(model,gender,SEX_MALE,AGE,VO_CHONG,NGUOI_THAN,
              SIBSP,BO,ME,TRE_EM,PARCH,ve,PCLASS,FARE,embarked,
              EMBARKED_Q,EMBARKED_S,name,quan_he):
    result=model.predict([[PCLASS,SEX_MALE,AGE,SIBSP,PARCH,FARE,EMBARKED_Q,EMBARKED_S]])
    if quan_he=='Là chính bạn':
        st.subheader('Bạn đã chọn:')
        st.write('- Giới tính của bạn:',gender)
        st.write('- Tuổi của bạn:',AGE,'tuổi')
        if VO_CHONG==0 and NGUOI_THAN==0 and BO==0 and ME==0 and TRE_EM==0:
            st.write('- Bạn đi một mình')
        else:
            if VO_CHONG==1:
                st.write('- Có vợ/chồng đi cùng')
            else:
                pass
            if NGUOI_THAN!=0:
                st.write('- Có',NGUOI_THAN,'anh chị em ruột đi cùng')
            else:
                pass
            if TRE_EM!=0:
                st.write('- Có dẫn theo',TRE_EM,'trẻ em đi cùng')
            else:
                pass
            if BO==1 and ME==1:
                st.write('- Bạn đi cùng bố mẹ')
            else:
                if BO==1:
                    st.write('- Bạn đi cùng bố')
                elif ME==1:
                    st.write('- Bạn đi cùng mẹ')
                else:
                    pass
        st.write('- Mua vé hạng',ve,'với giá tiền',FARE,'đô và lên tàu ở',embarked)
        st.write('...')
        if result==0:
            st.write('-->',speak_sorry(name,gender,quan_he))
        else:
            st.write('-->',speak_good(name,gender,quan_he))
    else:
        st.subheader('Người đó có những đặc điểm sau:')
        st.write('- Giới tính:',gender)
        st.write('- Tuổi:',AGE,'tuổi')
        if name=='':
            if VO_CHONG==0 and NGUOI_THAN==0 and BO==0 and ME==0 and TRE_EM==0:
                st.write('- Người đó đi một mình')
            else:
                if VO_CHONG==1:
                    st.write('- Có vợ/chồng đi cùng')
                else:
                    pass
                if NGUOI_THAN!=0:
                    st.write('- Có',NGUOI_THAN,'anh chị em ruột đi cùng')
                else:
                    pass
                if TRE_EM!=0:
                    st.write('- Có dẫn theo',TRE_EM,'trẻ em đi cùng')
                else:
                    pass
                if BO==1 and ME==1:
                    st.write('- Người đó đi cùng bố mẹ')
                else:
                    if BO==1:
                        st.write('- Người đó đi cùng bố')
                    elif ME==1:
                        st.write('- Người đó đi cùng mẹ')
                    else:
                        pass
            st.write('- Mua vé hạng',ve,'với giá tiền',FARE,'đô và lên tàu ở',embarked)
            st.write('...')
        else:
            if VO_CHONG==0 and NGUOI_THAN==0 and BO==0 and ME==0 and TRE_EM==0:
                st.write('-',name.capitalize(),'đi một mình')
            else:
                if VO_CHONG==1:
                    st.write('- Có vợ/chồng đi cùng')
                else:
                    pass
                if NGUOI_THAN!=0:
                    st.write('- Có',NGUOI_THAN,'anh chị em ruột đi cùng')
                else:
                    pass
                if TRE_EM!=0:
                    st.write('- Có dẫn theo',TRE_EM,'trẻ em đi cùng')
                else:
                    pass
                if BO==1 and ME==1:
                    st.write('-',name.capitalize(),'đi cùng bố mẹ')
                else:
                    if BO==1:
                        st.write('-',name.capitalize(),'đi cùng bố')
                    elif ME==1:
                        st.write('-',name.capitalize(),'đi cùng mẹ')
                    else:
                        pass
            st.write('-',name.capitalize(),'mua vé hạng',ve,'với giá tiền',FARE,'đô và lên tàu ở',embarked)
            st.write('...')
        if result==0:
            st.write('-->',speak_sorry(name,gender,quan_he))
        else:
            st.write('-->',speak_good(name,gender,quan_he))
    return
############################

############################
def xu_ly_du_lieu_2(X):
    X_pre=X.interpolate()
    X_pre=X_pre.drop(['cabin','home.dest'],axis=1)
    X_pre=X_pre.dropna()
    X_pre=pd.get_dummies(X_pre,columns=['sex','embarked'],drop_first=True)
    X_pre=X_pre[['pclass','sex_male','age','sibsp','parch','fare','embarked_Q','embarked_S']]
    return X_pre

def download_csv(df):
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href =f'Ok! Nhấn<a href="data:file/csv;base64,{b64}"\
          download="labeled_titanic.csv"   > vào đây </a>để tải file csv xuống'
    st.markdown(href, unsafe_allow_html=True)
    return
############################

############################
def main():
    st.title('Titanic - Dự đoán còn sống hay đã chết')

    #Load model
    with open('titanic_model.pkl','rb') as f:
        model=pk.load(f)
    
    menu=['Kiểm tra model','Dự đoán hành khách','Gán nhãn và lưu file']
    choice=st.sidebar.selectbox('Menu',menu)
    st.subheader(choice)

    if choice=='Kiểm tra model':
        #Load dữ liệu
        st.subheader('1. Load dữ liệu có chứa label')

        file_csv = st.file_uploader("Upload file csv tại đây", type=([".csv"]))
        if st.button('Nhấn vào đây nếu bạn không có mẫu!')
            file_csv='data/titanic.csv'
        if file_csv:
            data=pd.read_csv(file_csv)
            if 'survived' not in data.columns:
                st.write('Dữ liệu không có nhãn, vui lòng chọn dữ liệu khác')
            else:
                #Xử lý dữ liệu
                st.subheader('2. Xử lý dữ liệu')
                data=xu_ly_du_lieu(data)
                st.write('Dữ liệu của bạn đã được xử lý!')
                st.dataframe(data.head())

                #Chuẩn đoán dữ liệu
                st.subheader('3. Chuẩn đoán dữ liệu')
                X_train,X_test,y_train,y_test,fpr_train,tpr_train,\
                fpr_test,tpr_test,probs_train,probs_test=chuan_doan_du_lieu(data,model)

                #Vẽ biểu đồ
                st.subheader('4. Vẽ biểu đồ')
                ve_bieu_do(model,X_train,X_test,y_train,y_test,fpr_train,tpr_train,\
                           fpr_test,tpr_test,probs_train,probs_test)
        else:
            image = Image.open('image/titanic_image.jpg')
            st.image(image, caption=None,use_column_width=True,width=None)
    elif choice=='Dự đoán hành khách':
        st.write('- Dự đoán hành khách sau chuyến titanic còn sống hay đã chết hoặc bạn có thể thử vận may của bạn')
        st.write('(Câu hỏi * là những câu hỏi bắt buộc)')
        st.subheader('Lựa chọn thuộc tính')

        gender, SEX_MALE, AGE, VO_CHONG, NGUOI_THAN,\
        SIBSP, BO, ME, TRE_EM, PARCH, ve, PCLASS, FARE, embarked,\
        EMBARKED_Q, EMBARKED_S, name, quan_he=nhap_thong_tin()

        if st.button('Dự đoán'):
            thong_bao(model,gender,SEX_MALE,AGE,VO_CHONG,NGUOI_THAN,
                      SIBSP,BO,ME,TRE_EM,PARCH,ve,PCLASS,FARE,embarked,
                      EMBARKED_Q,EMBARKED_S,name,quan_he)
            st.write(thong_tin_huu_ich())
        else:
            st.write('- ...')
            st.write('- ...')
            st.write('- ...')
            st.write('- ...')
            st.write('- ...')
            st.write('- ...')
            st.write('- ...')
            st.write('- ...')
    else:
        #Load dữ liệu
        st.subheader('1. Load dữ liệu muốn gán nhãn')

        file_csv = st.file_uploader("Upload a csv file", type=([".csv"]))
        if st.button('Nhấn vào đây nếu bạn không có mẫu!')
            file_csv='data/titanic_new.csv'
        if file_csv:
            X=pd.read_csv(file_csv)
            if 'survived' in X.columns:
                st.write('Dữ liệu đã có nhãn, vui lòng chọn dữ liệu khác')
            
            else:
                #Xử lý dữ liệu
                st.subheader('2. Xử lý dữ liệu')

                X_pre=xu_ly_du_lieu_2(X)
                st.write('Dữ liệu của bạn đã được xử lý!')
                st.dataframe(X_pre.head())

                #Gán nhãn cho dữ liệu
                st.subheader('3. Gán nhãn cho dữ liệu')
                X_pre['survived']=model.predict(X_pre)
                st.write('Đã có nhãn cho dữ liệu tiền xử lý!')
                st.dataframe(X_pre.head())

                #Kiểm tra xem có dữ liệu nào model không đoán được
                X_pre=X_pre[['survived']]
                list_a=list(set(X.index)-set(X_pre.index))
                X_result_1=pd.merge(X,X_pre,'left',left_index=True,right_index=True)
                X_result_2=pd.merge(X,X_pre,'right',left_index=True,right_index=True)
                X_result_3=pd.merge(X,X_pre,'outer',left_index=True,right_index=True)

                #Tải dữ liệu xuống
                if len(list_a)!=0:
                    st.write('Có',len(list_a),'hành khách model không đoán được do bị thiếu một số thuộc tính quan trọng')
                    st.dataframe(X.iloc[list_a,:])
                    giu_lai=st.radio('Bạn có muốn dữ lại những hành khách này trong bảng?',['Có','Không'])
                    if giu_lai=='Có':
                        download_csv(X_result_1)
                    else:
                        download_csv(X_result_2)
                else:
                    download_csv(X_result_3)
        else:
            image = Image.open('image/titanic_image_2.jpg')
            st.image(image, caption=None,use_column_width=True,width=None)
    return

if __name__=='__main__':
    main()





































































