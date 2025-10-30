# ml model for regressive evolution in fishes
import pandas as pd
from pandas.api.types import is_numeric_dtype

import streamlit as st
# import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

# title
st.title("Prediction Model")

DEFAULT_XLSX_PATH = "astyanax_ml_dataset_expanded.xlsx"

uploaded_file = st.file_uploader(
    "Upload your Excel or CSV file",
    type=["xlsx", "csv"]
)

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")

    # Load file appropriately
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info(f"Using default file: `{DEFAULT_XLSX_PATH}`")
    df = pd.read_excel(DEFAULT_XLSX_PATH)


# shift index, start from 1 rather 0   
df.index = range(1,len(df)+1)
    
# display dataframe
st.dataframe(df)

# headers in file
columns_in_file = df.columns.to_list()

# remove identifier columns like Population name in our default datset
columns_in_file.remove("population_name")

# define default features for model
default_features = [
    "environment_light_level_lux",
    "pigmentation_score_0_1",
    "eye_area_mm2",
    "body_length_mm",
    "neuromast_count_superficial",
    "generation_time_months",
]
# select input features
features = st.multiselect("Select feature you want to include:",columns_in_file,default = default_features)
if (features):
    st.info("Got features!")
else:
    st.warning("Not any feature selected!")
    st.info("Using default features")
    features = default_features

# default target
default_target = "regression_index"
# select target or ground truth
target = st.selectbox("Select ground truth, ensure it is number!",columns_in_file,index=columns_in_file.index(default_target) if default_target in columns_in_file else len(columns_in_file)-1)

if not is_numeric_dtype(df[target]):
    st.warning("Target is not of number type!")
    target = default_target
    st.info("Reverted back to default!")
    


# Keep only columns that actually exist in file
features = [c for c in features if c in df.columns]

# input and target segregation
X = df[features].copy()
y = df[target].astype(float).copy()


button_input = st.button("Train Model",type="primary")
if(button_input):
    # perform train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # pipeline: Impute -> Scale -> Ridge
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=42)),
    ])
    
    # fit model in data
    pipe.fit(X_train, y_train)
    st.success("Model trained successfully!")
    y_pred = pipe.predict(X_test)

    # evaluate model
    r2 = r2_score(y_test, y_pred)*100
    mae = mean_absolute_error(y_test, y_pred)
    
    # Create 2 equal-width columns
    col1, col2 = st.columns(2)

    # Place metrics inside each column
    col1.metric("R2", f"{r2:.2f}%")
    col2.metric("MAE",f"{mae:.3f}")
    
    # coefficients
    coefficient = pipe.named_steps["model"].coef_
    coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficient
    })
    
    coef_df.index = range(1,len(coef_df)+1)
    
    st.header("Model Coefficients")
    st.table(coef_df)

    # log results
    print("Features used:", features)
    print(f"Test R^2:  {r2:.3f}")
    print(f"Test MAE:  {mae:.3f}")
    print("Coefficient",pipe.named_steps["model"].coef_)
    # save the trained model
    # joblib.dump(pipe, "simple_regression_model.joblib")


    
    

