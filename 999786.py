# %% [markdown]
# # Investigating periods of increasing interest rates for the S&P 1500
# ### by Luca Reichelt, 999786

# %% [markdown]
# ## 1. Outline

# %% [markdown]
# This project aims to investigate periods of increasing interest rates by the federal reserve based on the constituents of the S&P 1500. As an end result we attempt to implement our findings into a portfolio strategy.
#
# The first month of observation is defined as the month of the first hike. The last date of observation is defined as the last month before a decreasing or stagnant interest rate (no change). Company data is based on the latest data available which have been published before the first day.
# We do not adjust for changes in the SP1500, but will include all the data available for its constituents as of the first month of the respective period.
#
# All of the data that isn't retrieved while executing the code has been exported from Bloomberg Terminal or investing.com and is available within the folder "data_raw" in xlsx format. The data which is retrieved during the code execution is retrieved from Yahoo Finance or the FRED API, which is a API provided by the Federal Reserve of St.Lewis (https://fred.stlouisfed.org/docs/api/fred/).
#
# You should be able to run this notebook after creating a local environment using anconda/miniconda and executing following command while referring to the provided yaml file:
#
# `conda env create --file=env.yml`

# %% [markdown]
# ## 2. Data

# %%
# importing the main packages
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import pickle as pkl
import dataframe_image as dfi


# %% [markdown]
# ### 2.1 Data Retrieval

# %%
# getting the last periods of increasing interest rates (Federal Funds Effective MONTHLY Rate) from 1965 onwards
fed_rates = web.DataReader("FEDFUNDS", "fred", 1965)
fed_rates.set_index(fed_rates.index.astype("datetime64[ns]"), inplace=True)

periods1965 = pd.DataFrame(columns=["Name", "Start", "Last"])
periods1965.loc[periods1965.shape[0]] = [None, None, None]

period = 0
j = 0

# we search for periods that are at least 9 months (eight fed rate decisions)
for i in range(0, len(fed_rates) - 1):
    if (fed_rates.iloc[i + 1]["FEDFUNDS"] <= fed_rates.iloc[i]["FEDFUNDS"]) and (
        i - j >= 13
    ):
        periods1965.loc[period]["Last"] = datetime.strftime(
            fed_rates.index[i], "%Y-%m-%d"
        )
        period += 1
        if (
            datetime.date(fed_rates.index[-1]) - datetime.date(fed_rates.index[i])
        ).days >= 365:
            periods1965.loc[periods1965.shape[0]] = [None, None, None]
        j = i
    elif (fed_rates.iloc[i + 1]["FEDFUNDS"] <= fed_rates.iloc[i]["FEDFUNDS"]) and (
        i - j < 12
    ):
        periods1965.loc[period]["Name"] = "Period " + str(period + 1)
        periods1965.loc[period]["Start"] = datetime.strftime(
            fed_rates.index[i], "%Y-%m-%d"
        )
        j = i

# adding the last date for the current one
periods1965.loc[period]["Last"] = datetime.strftime(fed_rates.index[-1], "%Y-%m-%d")
periods1965["Duration"] = (
    (pd.to_datetime(periods1965["Last"]) - pd.to_datetime(periods1965["Start"]))
    / np.timedelta64(1, "M")
).astype(int)
dfi.export(
    periods1965.style.set_properties(
        **{"background-color": "white", "color": "black", "border-color": "#948b8b"}
    ),
    "periods1965.png",
)
periods1965

# unluckily, bloomberg doesn't provide constituent data much earlier than 1995
# note that this approach would've focused on the sp500 constituents, as the sp1500 was founded only in 1995

# %%
# getting the last periods of increasing interest rates (Federal Funds Effective MONTHLY Rate) from 1995 onwards
fed_rates = web.DataReader("FEDFUNDS", "fred", 1995)
fed_rates.set_index(fed_rates.index.astype("datetime64[ns]"), inplace=True)

periods = pd.DataFrame(columns=["Name", "Start", "Last"])
periods.loc[periods.shape[0]] = [None, None, None]

period = 0
j = 0

# we search for periods that are at least 9 months (eight fed rate decisions)
for i in range(0, len(fed_rates) - 1):
    if (fed_rates.iloc[i + 1]["FEDFUNDS"] <= fed_rates.iloc[i]["FEDFUNDS"]) and (
        i - j >= 10
    ):
        periods.loc[period]["Last"] = datetime.strftime(fed_rates.index[i], "%Y-%m-%d")
        period += 1
        if (
            datetime.date(fed_rates.index[-1]) - datetime.date(fed_rates.index[i])
        ).days >= 365:
            periods.loc[periods.shape[0]] = [None, None, None]
        j = i
    elif (fed_rates.iloc[i + 1]["FEDFUNDS"] <= fed_rates.iloc[i]["FEDFUNDS"]) and (
        i - j < 9
    ):
        periods.loc[period]["Name"] = "Period " + str(period + 1)
        periods.loc[period]["Start"] = datetime.strftime(fed_rates.index[i], "%Y-%m-%d")
        j = i

# adding the last date for the current one
periods.loc[period]["Last"] = datetime.strftime(fed_rates.index[-1], "%Y-%m-%d")
periods["Duration"] = (
    (pd.to_datetime(periods["Last"]) - pd.to_datetime(periods["Start"]))
    / np.timedelta64(1, "M")
).astype(int)
dfi.export(
    periods.style.set_properties(
        **{"background-color": "white", "color": "black", "border-color": "#948b8b"}
    ),
    "periods.png",
)
periods


# %%
def web_import(rate, start, end):
    df = web.DataReader(rate, "fred", start, end)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# %%
def yf_import(ticker, start, end):
    data = yf.download(ticker, start, end)
    monthly = data.groupby(pd.PeriodIndex(data.index, freq="M"))["Close"].mean()

    if data.size > 0:
        if (
            not data.empty == True
            and len(monthly.values)
            == round((data.index[-1] - data.index[0]) / np.timedelta64(1, "M"))
            and len(monthly.values) >= 9
        ):
            df = pd.DataFrame(
                index=pd.date_range(start, end, freq="MS"),
                data=np.append(monthly.values, data.iloc[-1]["Close"]),
            )
            df.rename(columns={df.columns[0]: ticker}, inplace=True)
            df.drop(columns=df.columns.difference([ticker]), inplace=True)
            df.index = pd.to_datetime(df.index)
            return df
        elif len(monthly.values) < 9:
            empty = np.empty((1, 11 - (len(monthly))))
            empty[:] = np.nan
            df_data = np.append(empty, monthly.values)
            df = pd.DataFrame(index=pd.date_range(start, end, freq="MS"), data=df_data)
            df.rename(columns={df.columns[0]: ticker}, inplace=True)
            df.drop(columns=df.columns.difference([ticker]), inplace=True)
            df.index = pd.to_datetime(df.index)
            return df
    else:
        df = pd.DataFrame(index=pd.date_range(start, end, freq="MS"), columns=[ticker])
        df.index = pd.to_datetime(df.index)
        return df


# %%
raw = "data_raw/"

gold = pd.read_csv(raw + "Gold_Futures.csv").set_index("Date")
gold["Gold"] = gold["Price"].str.replace(",", "").astype(float)
gold.drop(columns=gold.columns.difference(["Gold"]), inplace=True)
gold.index = pd.to_datetime(gold.index)
gold.sort_index(inplace=True)

# %% [markdown]
# ### 2.2 Introductory Visualizations

# %%
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

tickers = ["^SP1500", "^IXIC"]
rates = ["FEDFUNDS", "CORESTICKM159SFRBATL", "UNRATE"]


def comparison(tickers, start, end):
    df = gold.loc[start:end]
    for ticker in tickers:
        df = pd.concat([df, yf_import(ticker, start, end)], axis=1)
    df = pd.concat([df, web.DataReader("WTISPLC", "fred", start, end)], axis=1)

    df_chg = df.pct_change() * 100

    legend = {
        "Gold": "Gold",
        "^SP1500": "S&P 1500",
        "^IXIC": "Nasdaq",
        "WTISPLC": "Spot Crude Oil Price WTI",
        "FEDFUNDS": "FED Rate",
        "CORESTICKM159SFRBATL": "CPI",
        "UNRATE": "Rate of Unemployment",
    }

    fig = px.line(
        df_chg,
        y=["Gold", "^SP1500", "^IXIC", "WTISPLC"],
        labels=legend,
        title="Indices/Assets monthly change in %",
    )
    fig.update_yaxes(title="Change in % compared to month before")
    fig.update_xaxes(title="Date")
    fig.for_each_trace(
        lambda t: t.update(
            name=legend[t.name],
            legendgroup=legend[t.name],
            hovertemplate=t.hovertemplate.replace(t.name, legend[t.name]),
        )
    )

    for index, row in periods.iterrows():
        fig.add_vline(
            x=row["Start"], line_width=2, line_dash="dash", line_color="green"
        )
        fig.add_vline(x=row["Last"], line_width=2, line_dash="dash", line_color="red")

    fig.update_layout(legend_title="Legend", autosize=False, width=1200, height=600)
    fig.show()

    df_perf = df.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))

    for rate in rates:
        df_perf = pd.concat([df_perf, web.DataReader(rate, "fred", start, end)], axis=1)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    x = df_perf.index

    for ticker in tickers:
        fig.add_trace(
            go.Scatter(x=x, y=df_perf[ticker], name=legend[ticker]),
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(x=x, y=df_perf["Gold"], name="Gold"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=x, y=df_perf["WTISPLC"], name="Spot Crude Oil Price WTI"),
        secondary_y=False,
    )

    for rate in rates:
        fig.add_trace(
            go.Scatter(x=x, y=df_perf[rate], name=legend[rate]),
            secondary_y=True,
        )

    for index, row in periods.iterrows():
        fig.add_vline(
            x=row["Start"], line_width=2, line_dash="dash", line_color="green"
        )
        fig.add_vline(x=row["Last"], line_width=2, line_dash="dash", line_color="red")

    fig.update_yaxes(title_text="Total Indices/Asset change in %", secondary_y=False)
    fig.update_yaxes(title_text="Rates for FED/CPI/Unemployment", secondary_y=True)

    fig.update_xaxes(title="Date")

    fig.update_layout(
        title="Total performance to FED/CPI/Unemployment Rate",
        legend_title="Legend",
        autosize=False,
        width=1200,
        height=600,
    )
    fig.show()


# %%
comparison(tickers, periods.iloc[0]["Start"], periods.iloc[-1]["Last"])

# %% [markdown]
# ### 2.3 Pre-Processing the features data
# #### This data has been retrieved beforehand via the Bloomberg Excel Add-ins and is imported from the files in data-raw

# %%
# the periods refer to the four timeframes from the outline

period_data = [
    "SPR_Period_1.xlsx",
    "SPR_Period_2.xlsx",
    "SPR_Period_3.xlsx",
    "SPR_Period_4.xlsx",
]

# %%
import matplotlib.pyplot as plt
import seaborn as sns


def heatmap(df):
    df = df.corr(numeric_only=True)

    f, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(df, annot=True, linewidths=0.5, fmt=".1f", ax=ax)
    sns.set(font_scale=2)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()


# %%
# this function cleans the datasets and provides first tabular and visual descriptions of the data
# its also calls the heatmap visualization function above to give us an idea on how to structure the model


def df(filename):
    df = pd.read_excel(open(raw + filename, "rb"))

    string_cols = ["Ticker", "Name", "GICS Sector"]

    for col in set(df.columns) - set(string_cols):
        df.loc[:, col] = pd.to_numeric(df.loc[:, col], errors="coerce")

    df.dropna(inplace=True)

    sector_dummies = pd.get_dummies(df[["GICS Sector"]])
    df = pd.concat([df, sector_dummies], axis=1)

    df["Revenue per Employee"] = (
        df["Revenue T12M"] / df["Number of Employees:Y"]
    ).astype(float)

    df["Market Cap_perf"] = (
        (df["Market Cap_last"] - df["Market Cap"]) / df["Market Cap"]
    ).astype(float)

    df.drop(["Market Cap_last", "Price"], axis=1, inplace=True)

    df.set_index("Ticker", inplace=True)
    df.sort_values("GICS Sector", inplace=True)

    print("\nData decription for cleaned " + filename)
    perf = df.pop("Market Cap_perf")
    df.insert(0, "Market Cap_perf", perf)

    rpe = df.pop("Revenue per Employee")
    df.insert(1, "Revenue per Employee", rpe)

    print(df["GICS Sector"].value_counts(ascending=False))
    print(df.iloc[:, :12].describe())

    fig = px.scatter(
        df,
        x="Beta:M-1",
        y="Market Cap_perf",
        size="Market Cap",
        title="Risk/Compared Volatility to SP1500 compared to Return/Performance for single companies",
        color="GICS Sector",
        hover_name="Name",
        log_x=True,
        size_max=100,
        width=1200,
        height=600,
    )

    fig.show()

    grouped = df.groupby("GICS Sector").mean()
    fig = px.scatter(
        grouped,
        x="Beta:M-1",
        y="Market Cap_perf",
        size="Market Cap",
        title="Risk/Compared Volatility to SP1500 compared to Return/Performance by GICS Sector",
        color="Market Cap_perf",
        hover_name=grouped.index,
        log_x=True,
        size_max=100,
        width=1200,
        height=600,
    )

    fig.show()

    heatmap(df)

    return df


# %%
P1 = df("SPR_Period_1.xlsx")
P1.to_pickle("data/P1.pkl")
P2 = df("SPR_Period_2.xlsx")
P2.to_pickle("data/P2.pkl")
P3 = df("SPR_Period_3.xlsx")
P3.to_pickle("data/P3.pkl")
CrP = df("SPR_Period_4.xlsx")
CrP.to_pickle("data/CrP.pkl")

# %%
# saving the pre-processed dfs for easy access

P1 = pd.read_pickle("data/P1.pkl")
P2 = pd.read_pickle("data/P2.pkl")
P3 = pd.read_pickle("data/P3.pkl")
CrP = pd.read_pickle("data/CrP.pkl")

# %%
# creating a dataframe consisting of all data across periods
all_dfs = [P1, P2, P3, CrP]
all_df = pd.concat(all_dfs)

print("\nData description for all periods (including current period)")
perf = all_df.pop("Market Cap_perf")
all_df.insert(0, "Market Cap_perf", perf)

rpe = all_df.pop("Revenue per Employee")
all_df.insert(1, "Revenue per Employee", rpe)

print(all_df["GICS Sector"].value_counts(ascending=False))
print(all_df.iloc[:, :12].describe())

fig = px.scatter(
    all_df,
    x="Beta:M-1",
    y="Market Cap_perf",
    size="Market Cap",
    color="GICS Sector",
    title="ALL DATA: Risk/Compared Volatility to SP1500 compared to Return/Performance for single companies",
    hover_name="Name",
    log_x=True,
    size_max=100,
    width=1200,
    height=600,
)

fig.show()

grouped = all_df.groupby("GICS Sector").mean()
fig = px.scatter(
    grouped,
    x="Beta:M-1",
    y="Market Cap_perf",
    size="Market Cap",
    title="ALL DATA: Risk/Compared Volatility to SP1500 compared to Return/Performance by GICS Sector",
    color="Market Cap_perf",
    hover_name=grouped.index,
    log_x=True,
    size_max=100,
    width=1200,
    height=600,
)
fig.show()

heatmap(all_df)
heatmap(all_df.iloc[:, :12])

# %%
# creating a df consisting only of concluded periods

concluded_dfs = [P1, P2, P3]
concluded_df = pd.concat(concluded_dfs)

print("\nData decription for only concluded periods")
perf = concluded_df.pop("Market Cap_perf")
concluded_df.insert(0, "Market Cap_perf", perf)

rpe = concluded_df.pop("Revenue per Employee")
concluded_df.insert(1, "Revenue per Employee", rpe)

print(concluded_df["GICS Sector"].value_counts(ascending=False))
print(concluded_df.iloc[:, :12].describe())

fig = px.scatter(
    concluded_df,
    x="Beta:M-1",
    y="Market Cap_perf",
    size="Market Cap",
    title="CONCLUDED DATA: Risk/Compared Volatility to SP1500 compared to Return/Performance for single companies",
    color="GICS Sector",
    hover_name="Name",
    log_x=True,
    size_max=100,
    width=1200,
    height=600,
)

fig.show()

grouped = concluded_df.groupby("GICS Sector").mean()
fig = px.scatter(
    grouped,
    x="Beta:M-1",
    y="Market Cap_perf",
    size="Market Cap",
    title="CONCLUDED DATA: Risk/Compared Volatility to SP1500 compared to Return/Performance by GICS Sector",
    color="Market Cap_perf",
    hover_name=grouped.index,
    log_x=True,
    size_max=100,
    width=1200,
    height=600,
)
fig.show()

heatmap(concluded_df)

# %%
# insight into the characteristics of the best performing observations

top10 = pd.DataFrame()

for df in concluded_dfs:
    top10 = pd.concat(
        [top10, df.sort_values("Market Cap_perf", ascending=False).head(10)]
    )

perf = top10.pop("Market Cap_perf")
top10.insert(0, "Market Cap_perf", perf)

rpe = top10.pop("Revenue per Employee")
top10.insert(1, "Revenue per Employee", rpe)

print(top10["GICS Sector"].value_counts(ascending=False))
print(top10.iloc[:, :12].describe())

top10.sort_values("Market Cap_perf", ascending=False)

heatmap(top10)

# %% [markdown]
# ## 3. Machine learning

# %% [markdown]
# ### Basic Decision Tree

# %%
from sklearn.metrics import confusion_matrix, accuracy_score

# function to quickly retrieve minimal evaluation data for the classifier


def evaluate(clf, X_train, X_test, y_train, y_test):
    print("Train Accuracy :", accuracy_score(y_train, clf.predict(X_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, clf.predict(X_train)))
    print("Test Accuracy :", accuracy_score(y_test, clf.predict(X_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, clf.predict(X_test)))


# %%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


def decision_tree(df, title):
    df["Compared Performance"] = np.where(
        (df["Market Cap_perf"] > df["Market Cap_perf"].mean()),
        "Outperformed",
        "Not Outperformed",
    )
    y = df["Compared Performance"]
    X = df.drop(
        columns=["Compared Performance", "Market Cap_perf", "Name", "GICS Sector"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=222
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Compared Performance", data=df)
    plt.title("Target distribution")
    plt.show()

    print(f"X_train : {X_train.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_test : {y_test.shape}")

    print(title)
    dt = DecisionTreeClassifier(max_depth=2)

    dt.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(30, 30))
    plot_tree(
        dt,
        feature_names=X.columns,
        class_names=["Outperformed", "Not Outperformed"],
        filled=True,
        proportion=False,
        fontsize=35,
    )

    evaluate(dt, X_train, X_test, y_train, y_test)


# %%
decision_tree(
    all_df, "Using the data of all periods (including the current, ongoing one)"
)

# %%
decision_tree(concluded_df, "Using the data of only concluded periods")

# %%
concluded_df["Compared Performance"] = np.where(
    (concluded_df["Market Cap_perf"] > concluded_df["Market Cap_perf"].mean()),
    "Outperformed",
    "Not Outperformed",
)
y_train = concluded_df["Compared Performance"]
X_train = concluded_df.drop(
    columns=["Compared Performance", "Market Cap_perf", "Name", "GICS Sector"]
)

CrP["Compared Performance"] = np.where(
    (CrP["Market Cap_perf"] > CrP["Market Cap_perf"].mean()),
    "Outperformed",
    "Not Outperformed",
)
y_test = CrP["Compared Performance"]
X_test = CrP.drop(
    columns=["Compared Performance", "Market Cap_perf", "Name", "GICS Sector"]
)

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Compared Performance", data=all_df)
plt.title("Target distribution")
plt.show()

print(f"X_train : {X_train.shape}")
print(f"y_train : {y_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_test : {y_test.shape}")

print(
    "Using the data of concluded periods as training data and the current, ongoing period data, as test data"
)
dt = DecisionTreeClassifier(max_depth=2)

dt.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(30, 30))
plot_tree(
    dt,
    feature_names=X_train.columns,
    class_names=["Outperformed", "Not Outperformed"],
    filled=True,
    proportion=False,
    fontsize=35,
)

evaluate(dt, X_train, X_test, y_train, y_test)

# %% [markdown]
# ### 3.2 Random Forest with Hyperparametertuning

# %%
n_estimators = np.arange(50, 250, 50)
max_features = ["auto", "sqrt"]
max_depth = np.arange(2, 20, 1)
min_samples_leaf = [1, 5, 25, 50]
min_samples_split = [2, 5, 25, 50]
max_leaf_nodes = [50, 100, 250, 500]
bootstrap = [True, False]

params_arr = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_leaf": min_samples_leaf,
    "min_samples_split": min_samples_split,
    "max_leaf_nodes": max_leaf_nodes,
    "bootstrap": bootstrap,
}

# %%
from treeinterpreter import treeinterpreter as ti
import os
from sklearn.tree import export_graphviz
import six
import pydot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()


def visualize_best(gridname, grid, X_test, X_train, y_test, y_train):
    params = {
        "n_estimators": [grid.best_params_["n_estimators"]],
        "max_features": [grid.best_params_["max_features"]],
        "max_depth": [grid.best_params_["max_depth"]],
        "min_samples_leaf": [grid.best_params_["min_samples_leaf"]],
        "min_samples_split": [grid.best_params_["min_samples_split"]],
        "max_leaf_nodes": [grid.best_params_["max_leaf_nodes"]],
        "bootstrap": [grid.best_params_["bootstrap"]],
    }

    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"][0],
        max_features=params["max_features"][0],
        max_depth=params["max_depth"][0],
        min_samples_leaf=params["min_samples_leaf"][0],
        min_samples_split=params["min_samples_split"][0],
        max_leaf_nodes=params["max_leaf_nodes"][0],
        bootstrap=params["bootstrap"][0],
    )

    clf.fit(X_train.values, y_train.values)

    prediction, bias, contributions = ti.predict(clf, X_test.values)

    N = len(X_test.columns)

    outperformed = []
    not_outperformed = []

    for j in range(2):
        list_ = [outperformed, not_outperformed]
        for i in range(N - 1):
            val = contributions[0, i, j]
            list_[j].append(val)

    outperformed.append(prediction[0, 0] / N)
    not_outperformed.append(prediction[0, 1] / N)

    fig, ax = plt.subplots()
    ind = np.arange(N)

    width = 0.5

    p1 = ax.bar(ind, outperformed, width, color="green", bottom=0)
    p2 = ax.bar(ind + width, not_outperformed, width, color="red", bottom=0)

    ax.set_title("Feature importance for performance result")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(X_train.columns, rotation=90)
    ax.legend((p1[0], p2[0]), ("Outperformed", "Not Outperformed"), loc="upper left")
    ax.autoscale_view()

    fig.set_figwidth(15)
    plt.show()

    dotfile = six.StringIO()

    i = 0
    for tree_in_forest in clf.estimators_:
        export_graphviz(
            tree_in_forest,
            out_file="trees/" + gridname + "tree.dot",
            feature_names=X_train.columns,
            filled=True,
        )
        (graph,) = pydot.graph_from_dot_file("trees/" + gridname + "tree.dot")
        name = gridname + "tree_" + str(i)
        graph.write_png("trees/" + name + ".png")
        os.system("dot -Tpng tree.dot -o tree.png")
        i += 1


# %%
def params(X_train, y_train):
    rf_Grid = GridSearchCV(
        estimator=rf, param_grid=params_arr, cv=4, verbose=3, n_jobs=-1
    )
    rf_Grid.fit(X_train.values, y_train.values)

    params = {
        "n_estimators": [rf_Grid.best_params_["n_estimators"]],
        "max_features": [rf_Grid.best_params_["max_features"]],
        "max_depth": [rf_Grid.best_params_["max_depth"]],
        "min_samples_leaf": [rf_Grid.best_params_["min_samples_leaf"]],
        "min_samples_split": [rf_Grid.best_params_["min_samples_split"]],
        "max_leaf_nodes": [rf_Grid.best_params_["max_leaf_nodes"]],
        "bootstrap": [rf_Grid.best_params_["bootstrap"]],
    }

    print(params)

    return params


# %%
# current is test
current_is_test_params = params(X_train, y_train)

current_test_Grid = GridSearchCV(
    estimator=rf, param_grid=current_is_test_params, cv=4, verbose=3, n_jobs=-1
)
current_test_Grid.fit(X_train.values, y_train.values)

evaluate(
    current_test_Grid, X_train.values, X_test.values, y_train.values, y_test.values
)

visualize_best("current_test_", current_test_Grid, X_train, X_test, y_train, y_test)

CrP.drop(["Compared Performance"], axis=1, inplace=True)

current_test_Grid

# %%
# all_df
y = all_df["Compared Performance"]
X = all_df.drop(
    columns=["Compared Performance", "Market Cap_perf", "Name", "GICS Sector"]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=222
)

all_params = params(X_train, y_train)

all_Grid = GridSearchCV(estimator=rf, param_grid=all_params, cv=4, verbose=3, n_jobs=-1)
all_Grid.fit(X_train.values, y_train.values)

evaluate(all_Grid, X_train.values, X_test.values, y_train.values, y_test.values)

visualize_best("all_", all_Grid, X_train, X_test, y_train, y_test)

all_Grid

# %%
# concluded_df
y = concluded_df["Compared Performance"]
X = concluded_df.drop(
    columns=["Compared Performance", "Market Cap_perf", "Name", "GICS Sector"]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=222
)

concluded_params = params(X_train, y_train)

concluded_Grid = GridSearchCV(
    estimator=rf, param_grid=concluded_params, cv=4, verbose=3, n_jobs=-1
)
concluded_Grid.fit(X_train.values, y_train.values)

evaluate(concluded_Grid, X_train.values, X_test.values, y_train.values, y_test.values)

visualize_best("concluded_", concluded_Grid, X_train, X_test, y_train, y_test)

concluded_Grid

# %% [markdown]
# ## 3.3 Predictor implementation
# COMPARED TO INVESTING INTO THE SP1500 ONLY
#
# Using the findings from above we can determine the companies predicted to outperform

# %%
import statistics

# we add all companies that have been identified as "Outperformed" and assign equal portfolio weights
all_classifier_performance = []
concluded_classifier_performance = []
current_test_classifier_performance = []

# calculating the portfolio performances
for df in all_dfs:
    current_test_outperformed = []
    concluded_outperformed = []
    all_outperformed = []

    compare_df = df.drop(["Market Cap_perf", "Name", "GICS Sector"], axis=1)
    for i in range(df.shape[0]):
        if current_test_Grid.predict([compare_df.iloc[i].values]) == "Outperformed":
            current_test_outperformed.append(df.iloc[i, 0])
        if all_Grid.predict([compare_df.iloc[i].values]) == "Outperformed":
            all_outperformed.append(df.iloc[i, 0])
        if concluded_Grid.predict([compare_df.iloc[i].values]) == "Outperformed":
            concluded_outperformed.append(df.iloc[i, 0])

    print(current_test_outperformed)
    print(all_outperformed)
    print(concluded_outperformed)

    current_test_classifier_performance.append(1 + np.mean(current_test_outperformed))
    all_classifier_performance.append(1 + np.mean(all_outperformed))
    concluded_classifier_performance.append(1 + np.mean(concluded_outperformed))

# %%
# importing comparative indices/assets, gold is still saved
sp1500 = yf_import("^SP1500", periods.iloc[0]["Start"], periods.iloc[-1]["Last"])
nasdaq = yf_import("^IXIC", periods.iloc[0]["Start"], periods.iloc[-1]["Last"])
WTI = web.DataReader(
    "WTISPLC", "fred", periods.iloc[0]["Start"], periods.iloc[-1]["Last"]
)

# %%
performance = periods
gold_performance = []
sp1500_performance = []
nasdaq_performance = []
WTI_performance = []

# creating a data-frame to compare performances of the portfolios with common indices and "crisis" resources
for index, row in periods.iterrows():
    gold_performance.append(
        (
            1
            + ((gold.loc[row["Last"]].values) - (gold.loc[row["Start"]].values))
            / (gold.loc[row["Start"]].values)
        ).item()
    )

    sp1500_performance.append(
        (
            1
            + ((sp1500.loc[row["Last"]].values) - (sp1500.loc[row["Start"]].values))
            / (sp1500.loc[row["Start"]].values)
        ).item()
    )

    nasdaq_performance.append(
        (
            1
            + ((nasdaq.loc[row["Last"]].values) - (nasdaq.loc[row["Start"]].values))
            / (nasdaq.loc[row["Start"]].values)
        ).item()
    )

    WTI_performance.append(
        (
            1
            + ((WTI.loc[row["Last"]].values) - (WTI.loc[row["Start"]].values))
            / (WTI.loc[row["Start"]].values)
        ).item()
    )

# %%
performance["Gold"] = gold_performance
performance["Nasdaq"] = nasdaq_performance
performance["Spot Crude Oil Price WTI"] = WTI_performance
performance["S&P 1500"] = sp1500_performance
performance["all_Grid"] = all_classifier_performance
performance["concluded_Grid"] = concluded_classifier_performance
performance["current_test_Grid"] = current_test_classifier_performance
dfi.export(
    performance.style.set_properties(
        **{"background-color": "white", "color": "black", "border-color": "#948b8b"}
    ),
    "performance.png",
)
performance
