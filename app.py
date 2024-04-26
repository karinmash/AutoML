import streamlit as st
import pandas as pd
import numpy as np
import ydata_profiling
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, LabelEncoder
# from streamlit_pandas_profiling import st_profile_report
from streamlit_ydata_profiling import st_profile_report
import plotly.express as px
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


from pycaret.classification import *

plot_colors = px.colors.sequential.YlOrRd[::-2]

# Page Configuration
st.set_page_config(page_title="AutoML", page_icon="🔍", layout="centered")

st.markdown(
    "<div style='background-color:#17BBEC; border-radius:50px;'><h1 style='text-align:center; color:white;'>Auto Machine Learning Web Application</h1></div>",
    unsafe_allow_html=True)

st.markdown("<h3 style='text-align:center; color:black;'>Classification</h3>", unsafe_allow_html=True)







df = None
if 'df' not in st.session_state:
    st.session_state.df = df
else:
    df = st.session_state.df

if 'clicked' not in st.session_state:
    st.session_state.clicked = False


def click_button():
    st.session_state.clicked = True


def random_imputation(new_data2, missing_columns):
    for feature in missing_columns:
        number_missing = new_data2[feature].isnull().sum()
        observed_values = new_data2.loc[new_data2[feature].notnull(), feature]
        new_data2.loc[new_data2[feature].isnull(), feature] = np.random.choice(observed_values, number_missing,
                                                                               replace=True)
    return new_data2


st.subheader("Upload a dataset")
file = st.file_uploader("Upload your dataset", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    st.write(df)
    st.write("Dataset shape:", df.shape)
    chosen_target = st.selectbox("Choose your target variable", df.columns.tolist())
    st.write("chosen target is ", chosen_target)


with st.expander("Exploratory Data Analysis"):
    if df is not None:
        eda_choise = st.selectbox(label="Pick the operation you want",
                                  options=['Show shape', 'Show data type', 'Show missing values', 'Description',
                                           'Show columns', 'Show selected columns', 'Show Correlation Heatmap',
                                           'Show Value Counts', 'Show Unique Values', 'Show ydata profiling'],
                                  index=None)
        if eda_choise == None:
            st.write("Please select an Option...")

        elif eda_choise == 'Show shape':
            st.write("Shape of the Dataset :", df.shape)

        elif eda_choise == 'Show data type':
            st.write("Data type of the Columns :", df.dtypes.astype(str))

        elif eda_choise == 'Show missing values':
            st.write("Number of Nan values across columns:", df.isna().sum())
            st.write("Total number of Nan values:", df.isna().sum().sum())

        elif eda_choise == 'Description':
            st.write("Description of the Dataset :")
            st.dataframe(df.describe(), use_container_width=True)

        elif eda_choise == 'Show columns':
            st.write("Columns in the Dataset :")
            st.dataframe(df.columns, use_container_width=True, hide_index=True)

        elif eda_choise == 'Show selected columns':
            selected_columns = st.multiselect('Select desired columns', df.columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)

        elif eda_choise == 'Show Correlation Heatmap':
            st.subheader("Correlation Heatmap")
            fig = px.imshow(df.corr(), width=980)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig)

        elif eda_choise == 'Show Value Counts':
            st.header('Value counts in a :blue[_bar chart_]', divider='rainbow')
            # st.set_option('deprecation.showPyplotGlobalUse', False)

            all_cols_less_40 = [col for col in df.columns if df[col].nunique() < 40]
            cols_to_show = st.multiselect("Select columns to show", all_cols_less_40, default=all_cols_less_40[0])
            for col in cols_to_show:
                st.bar_chart(df[col].value_counts(dropna=False))
                st.subheader('_In Percentage(%):_')
                st.bar_chart(round(df[col].value_counts(dropna=False, normalize=True) * 100, 2))

            st.header('Value counts in a :blue[_chart_]', divider='rainbow')
            selected_columns = st.multiselect('Select desired columns', df.columns.tolist(), default=cols_to_show)
            new_df = df[selected_columns]
            st.write(new_df.value_counts(dropna=False).rename(index='Value'))
            all_cols_more_40 = [col for col in df.columns if df[col].nunique() >= 40]
            if len(all_cols_more_40) > 1:
                st.subheader("Scatter plot between two columns")
                x_col = st.selectbox("Select x column", all_cols_more_40, index=0)
                y_col = st.selectbox("Select y column", all_cols_more_40, index=1)
                fig = px.scatter(df, x=x_col, y=y_col, width=980, color_discrete_sequence=plot_colors[1:])
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig)

        elif eda_choise == 'Show Unique Values':
            st.header('The Unique Values', divider='rainbow')
            all_cols_less_40 = [col for col in df.columns if df[col].nunique() < 40]
            for col in all_cols_less_40:
                st.subheader(col)
                st.write('unique values: ', df[col].unique())
                st.write('number of unique values: ', df[col].nunique(dropna=False))

        elif eda_choise == 'Show ydata profiling':
            profile_df = ydata_profiling.ProfileReport(df)
            st_profile_report(profile_df)

with st.expander("Data Cleaning"):
    if df is not None:
        st.write("Select cleaning options")
        drop_duplicates = st.checkbox("Remove duplicates")
        drop_colmuns = st.checkbox("Drop specific columns")
        Handling_missing_data = st.checkbox("Handle missing data")

        if drop_colmuns:
            columns = st.multiselect("Select columns to drop", df.columns)
        if Handling_missing_data:
            Handling_missing_data_operation = st.selectbox('Please pick the operation you want',
                                                           ['', 'Drop the missing data', 'Fill the missing data'])
            if 'Drop the missing data' in Handling_missing_data_operation:
                drop_na0 = st.checkbox("Drop all rows with Nan values")
                drop_na1 = st.checkbox("Drop all columns with Nan values")
            if 'Fill the missing data' in Handling_missing_data_operation:
                cat_col = [col for col in df.columns if df[col].dtype == 'object']
                num_col = [col for col in df.columns if df[col].dtype != 'object']
                columns_with_empty_values = df.columns[df.isnull().any()].tolist()
                numeric_columns_with_empty_values = df[num_col].columns[df[num_col].isnull().any()].tolist()
                categorial_columns_with_empty_values = df[cat_col].columns[df[cat_col].isnull().any()].tolist()
                Imputation_options = st.multiselect("Select the imputations you want", options=(
                    "Mean", "Median", "Most frequent", "Random Imputation"))

                if "Mean" in Imputation_options:
                    columns_for_Mean_imputation = st.multiselect(
                        "Please select the columns that you want to perform Mean imputation on",
                        numeric_columns_with_empty_values)

                if "Median" in Imputation_options:
                    columns_for_Median_imputation = st.multiselect(
                        "Please select the columns that you want to perform Median imputation on",
                        numeric_columns_with_empty_values)

                if "Most frequent" in Imputation_options:
                    columns_for_Most_frequent_imputation = st.multiselect(
                        "Please select the columns that you want to perform Most frequent imputation on",
                        columns_with_empty_values)

                if "Random Imputation" in Imputation_options:
                    columns_for_Random_imputation = st.multiselect(
                        "Please select the columns that you want to perform Random Imputation imputation on",
                        columns_with_empty_values)

        # st.button('Apply data cleaning', on_click=click_button)
        if st.button('Apply data cleaning'):
        # if st.session_state.clicked:
            if drop_duplicates:
                df = df.drop_duplicates()
            if drop_colmuns:
                df = df.drop(columns, axis=1)
            if Handling_missing_data:
                if 'Drop the missing data' in Handling_missing_data_operation:
                    if drop_na0:
                        df = df.dropna(axis=0)
                    if drop_na1:
                        df = df.dropna(axis=1)
                if 'Fill the missing data' in Handling_missing_data_operation:
                    if "Mean" in Imputation_options:
                        imputer = SimpleImputer(strategy='mean')
                        df[columns_for_Mean_imputation] = np.round(
                            imputer.fit_transform(df[columns_for_Mean_imputation]), 1)

                    if "Median" in Imputation_options:
                        imputer = SimpleImputer(strategy='median')
                        df[columns_for_Median_imputation] = np.round(
                            imputer.fit_transform(df[columns_for_Median_imputation]), 1)

                    if "Most frequent" in Imputation_options:
                        imputer = SimpleImputer(strategy='most_frequent')
                        df[columns_for_Most_frequent_imputation] = imputer.fit_transform(
                            df[columns_for_Most_frequent_imputation])

                    if "Random Imputation" in Imputation_options:
                        df = random_imputation(df, columns_for_Random_imputation)

            st.session_state.df = df
            st.write(df)
            st.write("Dataset shape:", df.shape)
            st.write("Total number of Nan values remaining:", df.isna().sum().sum())

       

with st.expander("Data Preprocessing"):
    if df is not None:
        if st.session_state.df is not None:
            df = st.session_state.df
        st.write("Select preprocessing options")
        Scale = st.checkbox("Scale your dataset")
        Encode = st.checkbox("Encode your dataset")
        balance = st.checkbox("Balance your dataset")
        if Scale:
            num_col_ = [col for col in df.columns if df[col].dtype != 'object']
            columns_for_scaling = st.multiselect("Please pick the columns you want to scale", num_col_)

        if Encode:
            encoding_options = st.multiselect("Select encoding type", options=(
                "Label Encoder", "One Hot Encoder", "Ordinal Encoder"))
            if "Label Encoder" in encoding_options:
                columns_for_label = st.multiselect("Please pick the columns you want to encode(label encoder)",
                                                   df.columns.tolist())

            if "One Hot Encoder" in encoding_options:
                columns_for_onehot = st.multiselect("Please pick the columns you want to encode(one hot encoder)",
                                                    df.columns.tolist())

            if "Ordinal Encoder" in encoding_options:
                columns_for_Ordinal = st.multiselect("Please pick the columns you want to encode(Ordinal encoder)",
                                                     df.columns.tolist())
        if balance:
            sample_tech = st.selectbox("Select sampling technique", ["Over Sampling", "Under Sampling", "Combined"])
        # st.button('Apply data preprocessing', on_click=click_button)
        if st.button('Apply data preprocessing'):
        # if st.session_state.clicked:
            if Scale:
                scaler = MinMaxScaler()
                df[columns_for_scaling] = scaler.fit_transform(df[columns_for_scaling])


            if Encode:
                if "Label Encoder" in encoding_options:
                    encoder = LabelEncoder()
                    df[columns_for_label] = df[columns_for_label].apply(encoder.fit_transform)


                if "One Hot Encoder" in encoding_options:
                    df = pd.get_dummies(df, columns=columns_for_onehot, prefix=columns_for_onehot, drop_first=True)


                if "Ordinal Encoder" in encoding_options:
                    encoder = OrdinalEncoder()
                    df[columns_for_Ordinal] = encoder.fit_transform(df[columns_for_Ordinal])


            if balance:
                X = df.drop(chosen_target, axis=1)
                y = df[chosen_target]
                if sample_tech == "Over Sampling":
                    X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(X, y)
                elif sample_tech == "Under Sampling":
                    X_resampled, y_resampled = RandomUnderSampler(random_state=0).fit_resample(X, y)
                elif sample_tech == "Combined":
                    X_resampled, y_resampled = SMOTEENN(random_state=0).fit_resample(X, y)
                df = pd.DataFrame(X_resampled, columns=df.columns[:-1])
                df[chosen_target] = y_resampled
                st.bar_chart(df[chosen_target].value_counts())

            st.session_state.df = df
            st.write(df)
            st.write("Dataset shape:", df.shape)

with st.expander("Model Training"):
    if df is not None:
        if st.session_state.df is not None:
            df = st.session_state.df
        st.write("Dataset to be used for training")
        st.write(df)
        st.write("Dataset shape:", df.shape)
        N=5
 
        if st.button('Run Modelling'):
            setup(df, target=chosen_target, verbose=False)
            setup_df = pull()
            st.info("This is the ML experiment settings")
            st.dataframe(setup_df)
            best_N = compare_models(n_select=N)
            compare_best_N = pull()
            best_model = best_N[0]
            st.info("The performance of all the estimators available in the model library")
            st.dataframe(compare_best_N)
            tuned_models = [tune_model(model) for model in best_N]
            best_model = compare_models(include=tuned_models)
            compare_df = pull()
            st.info("This is the ML model After Tuning the hyperparameters of 5 best estimators")
            st.dataframe(compare_df)
            fig = px.bar(
                compare_df.set_index("Model"),
                orientation='h',
                width=980
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis={'categoryorder': 'total ascending'})
            st.write(fig)
            st.info("The winner algorithm after Hyperparameter tuning is:")
            best_model
            plot_model(best_model, plot='confusion_matrix', plot_kwargs={'percent': True},
                        display_format='streamlit')

            plot_model(best_model, plot='parameter', display_format='streamlit')
            pl = ['confusion_matrix', 'auc', 'threshold', 'pr', 'class_report', 'learning', 'manifold']
            for plot in pl:
                try:
                    plot_model(best_model, plot=plot, display_format='streamlit')
                except:
                    pass


            save_model(best_model, 'best_model')

            with open('best_model.pkl', 'rb') as model_file:
                st.download_button('Download the model', model_file, 'best_model.pkl')

