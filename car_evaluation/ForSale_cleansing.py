import pandas as pd
import numpy as np
import openpyxl

#parse the excel sheet, rename columns, and clean the data
column_names = ["Ad_number","Ad_description","Price","Car_brand","Car_kind"]
df_sheet3 = pd.read_excel('extracted_data-1.xlsx', sheet_name='Sheet3', names=column_names)
df_sheet3["Ad_number"] = df_sheet3["Ad_number"].str[13:]

column_names = ["Ad_number","discrptions_values","description_headers"]
df_sheet4 = pd.read_excel('extracted_data-1.xlsx', sheet_name='Sheet4', names=column_names)
df_sheet4["Ad_number"] = df_sheet4["Ad_number"].str[13:]


#start creating the structured dataframe
# df_cleaned = pd.DataFrame(columns = ["Model_year","Car_color","Millage","Car_brand","Car_kind"])
df_cleaned = pd.DataFrame()
df_cleaned.insert(0,'Ad_number',df_sheet3[["Ad_number"]])

# Filtering by 'سنة الصنع' and renaming columns
filtered_df_model_year = df_sheet4[df_sheet4['description_headers'] == 'سنة الصنع'].copy()
filtered_df_model_year.rename(columns = {'discrptions_values':'Model_year', 'YOUR_VALUE_COLUMN_NAME':'Model_year_value'}, inplace=True)
print(filtered_df_model_year)
# Filtering by 'العداد' and renaming columns
filtered_df_millage = df_sheet4[df_sheet4['description_headers'] == 'العداد'].copy()
filtered_df_millage.rename(columns = {'discrptions_values':'Millage', 'YOUR_VALUE_COLUMN_NAME':'Millage_value'}, inplace=True)
print(filtered_df_millage)

#do the inner left joins for all structured data frames where Ad_number is a primary key
df_cleaned = df_cleaned.merge(df_sheet3[['Ad_number','Car_brand']], on='Ad_number', how='left')
df_cleaned = df_cleaned.merge(df_sheet3[['Ad_number','Price']], on='Ad_number', how='left')
df_cleaned = df_cleaned.merge(df_sheet3[['Ad_number','Car_kind']], on='Ad_number', how='left')
df_cleaned = df_cleaned.merge(filtered_df_model_year[['Ad_number','Model_year']],on='Ad_number',how='left')
df_cleaned = df_cleaned.merge(filtered_df_millage[['Ad_number','Millage']],on='Ad_number',how='left')


df_cleaned.to_csv("scrapped_4sale_data.csv", index=False, encoding='utf-8-sig')
