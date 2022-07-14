import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import scipy.stats as sci
import numpy as np

st.set_page_config(layout="wide")

st.title('Welcome!')
st.write("We are Team 9 (Vatsal Bora, Shivani Chidella, Soohyun Kim, Louis Liu). The primary objective of our project is to help people with an interest in subjective well-being (such as politicians and bureaucrats) understand how different factors impact a Happiness Score so that they can try improving it.")
st.write("In the scatter plots below, we showcase the relationships between 6 factors affecting the Happiness Score (denoted as 'Ladder').")

data = pd.read_csv('https://raw.githubusercontent.com/s00hyunkim/world-happiness-regression/main/world-happiness-score-2020.csv')

data.drop(columns = ['Country name', 'Regional indicator', 'Standard error of ladder score',
'upperwhisker', 'lowerwhisker', 'Ladder score in Dystopia', 'Explained by: Log GDP per capita',
'Explained by: Social support', 'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices',
'Explained by: Generosity', 'Explained by: Perceptions of corruption', 'Dystopia + residual'],
axis = 1, inplace = True)

data.columns = ['Ladder', 'GDP', 'Social Support', 'Healthy Life Expectancy', 'Freedom', 'Generosity', 'Corruption']

plot_gdp = px.scatter(data_frame=data, x= 'GDP', y='Ladder')
plot_ss = px.scatter(data_frame=data, x= 'Social Support', y='Ladder')
plot_hle = px.scatter(data_frame=data, x= 'Healthy Life Expectancy', y='Ladder')
plot_fre = px.scatter(data_frame=data, x= 'Freedom', y='Ladder')
plot_gen = px.scatter(data_frame=data, x= 'Generosity', y='Ladder')
plot_cor = px.scatter(data_frame=data, x= 'Corruption', y='Ladder')

col1, col2, col3 = st.columns(3)
plot_gdp.update_layout(height=250, width=500)
plot_ss.update_layout(height=250, width=500)
plot_hle.update_layout(height=250, width=500)
plot_fre.update_layout(height=250, width=500)
plot_gen.update_layout(height=250, width=500)
plot_cor.update_layout(height=250, width=500)

col1.markdown("<h5 style='text-align: center;'>Effect of GDP</h5>", unsafe_allow_html=True)
col1.plotly_chart(plot_gdp)

col1.markdown("<h5 style='text-align: center;'>Effect of Social Support</h5>", unsafe_allow_html=True)
col1.plotly_chart(plot_ss)

col2.markdown("<h5 style='text-align: center;'>Effect of Healthy Life Expectancy</h5>", unsafe_allow_html=True)
col2.plotly_chart(plot_hle)

col2.markdown("<h5 style='text-align: center;'>Effect of Freedom</h5>", unsafe_allow_html=True)
col2.plotly_chart(plot_fre)

col3.markdown("<h5 style='text-align: center;'>Effect of Generosity</h5>", unsafe_allow_html=True)
col3.plotly_chart(plot_gen,use_column_width=True, width=800, height=400)

col3.markdown("<h5 style='text-align: center;'>Effect of Corruption</h5>", unsafe_allow_html=True)
col3.plotly_chart(plot_cor,use_column_width=True, width=800, height=400)

c1, c2, c3, c4 = st.columns(4)

c2.subheader('Rankings of Features')
c2.write("View how much each factor impacts the Happiness Score based on the correlation coefficients.")
#Removing the y value from pairwise correlation and sorting it in descending order based on magnitude and not sign
correlation = data.corr()['Ladder'].iloc[1:].abs().sort_values(ascending=False)
#Corruption has negative correlation
correlation['Corruption'] *= -1
c2.write(correlation)

c1.subheader('Pick a Regression Model')
reg_model = c1.radio(
     "Choose a regression model to predict the Happiness Score, and then input values for each factor using the sliders. A high Happiness Score is desirable!",
     ('Standard Linear Regression', 'Ridge Linear Regression'))

X = data.iloc[:,1:]
y = data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
coef = reg.coef_

ridge_reg = Ridge().fit(X_train, y_train)
ridge_y_pred = ridge_reg.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_y_pred)
ridge_coef = ridge_reg.coef_

gdp_confint = sci.t.interval(alpha=0.95, df=len(data['GDP'])-1, loc=np.mean(data['GDP']), scale=sci.sem(data['GDP']))
ss_confint = sci.t.interval(alpha=0.95, df=len(data['Social Support'])-1, loc=np.mean(data['Social Support']), scale=sci.sem(data['Social Support']))
hle_confint = sci.t.interval(alpha=0.95, df=len(data['Healthy Life Expectancy'])-1, loc=np.mean(data['Healthy Life Expectancy']), scale=sci.sem(data['Healthy Life Expectancy']))
fre_confint = sci.t.interval(alpha=0.95, df=len(data['Freedom'])-1, loc=np.mean(data['Freedom']), scale=sci.sem(data['Freedom']))
gen_confint = sci.t.interval(alpha=0.95, df=len(data['Generosity'])-1, loc=np.mean(data['Generosity']), scale=sci.sem(data['Generosity']))
cor_confint = sci.t.interval(alpha=0.95, df=len(data['Corruption'])-1, loc=np.mean(data['Corruption']), scale=sci.sem(data['Corruption']))

gdp = c3.number_input('Insert the value for GDP. 95% conf. int: {}'.format(gdp_confint))
ss = c3.number_input('Insert the value for Social Support. 95% conf. int: {}'.format(ss_confint))
hle = c3.number_input('Insert the value for Healthy Life Expectancy. 95% conf. int: {}'.format(hle_confint))
fre = c4.number_input('Insert the value for Freedom. 95% conf. int: {}'.format(fre_confint))
gen = c4.number_input('Insert the value for Generosity. 95% conf. int: {}'.format(gen_confint))
cor = c4.number_input('Insert the value for Corruption. 95% conf. int: {}'.format(cor_confint))

if reg_model == "Standard Linear Regression":
     prediction = reg.intercept_ + coef[0]*gdp + coef[1]*ss + coef[2]*hle + coef[3]*fre + coef[4]*gen + coef[5]*cor
     st.subheader('Predicted Happiness Score: {}'.format(prediction))
     st.subheader("Model's Coefficient of Determination: {}".format(r2))
else:
     prediction = ridge_reg.intercept_ + ridge_coef[0]*gdp + ridge_coef[1]*ss + ridge_coef[2]*hle + ridge_coef[3]*fre + ridge_coef[4]*gen + ridge_coef[5]*cor
     st.subheader('Predicted Happiness Score: {}'.format(prediction))
     st.subheader("Model's Coefficient of Determination: {}".format(ridge_r2))

st.button("Re-run")

st.markdown("<h5 style='text-align: center;'>Important notes on units and meaning of features (note that GWP refers to the Gallup World Poll (GWP)):</h5>", unsafe_allow_html=True)
st.write("- 'GDP' refers to logged GDP per capita in purchasing power parity (PPP) at constant 2011 international dollar prices.")
st.write("- Social support (or having someone to count on in times of trouble) is the national average of the binary responses (either 0 or 1) to the GWP question “If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them, or not?'")
st.write("- Healthy life expectancy refers to life expectancy in years.")
st.write("- Freedom to make life choices is the national average of responses to the GWP question “Are you satisfied or dissatisfied with your freedom to choose what you do with your life?”")
st.write("- Generosity is the residual of regressing national average of response to the GWP question “Have you donated money to a charity in the past month?” on GDP per capita.")
st.write("- The Corruption measure is the national average of the survey responses to two questions in the GWP: “Is corruption widespread throughout the government or not” and “Is corruption widespread within businesses or not?” The overall perception is just the average of the two 0-or-1 responses.")