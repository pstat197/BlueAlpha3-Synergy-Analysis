import pandas as pd
import tensorflow_probability as tfp
import IPython.display

from meridian.data import data_frame_input_data_builder
from meridian import model, spec, constants
from meridian.analysis import reviewer

df = pd.read_csv("data/monthly_mocha.csv")
df = df.drop(columns=["date"], errors="ignore")
df = df.loc[:, (df != 0).any(axis=0)]

kpi_col = "subscriptions"

channels = ["Channel0", "Channel1", "Channel2", "Channel3", "Channel4"]

media_cols = [f"{c}_impression" for c in channels]
media_spend_cols = [f"{c}_spend" for c in channels]

control_cols = [
    "sentiment_score_control",
    "competitor_sales_control"
]

non_media_cols = ["Promo"]
organic_cols = ["Organic_channel0_impression"]
organic_channels = ["Organic_channel0"]

builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
    kpi_type="non_revenue",
    default_kpi_column=kpi_col,
    default_revenue_per_kpi_column="revenue_per_conversion",
)

builder = (
    builder.with_kpi(df)
    .with_revenue_per_kpi(df)
    .with_population(df)
    .with_controls(df, control_cols=control_cols)
    .with_media(
        df,
        media_cols=media_cols,
        media_spend_cols=media_spend_cols,
        media_channels=channels,
    )
    .with_non_media_treatments(
        df,
        non_media_treatment_cols=non_media_cols
    )
    .with_organic_media(
        df,
        organic_media_cols=organic_cols,
        organic_media_channels=organic_channels,
    )
)

data = builder.build()

roi_mu = 0.2
roi_sigma = 0.9

prior = spec.prior_distribution.PriorDistribution(
    roi_m=tfp.distributions.LogNormal(
        loc=roi_mu,
        scale=roi_sigma,
        name=constants.ROI_M
    )
)

model_spec = spec.ModelSpec(
    prior=prior,
    enable_aks=True
)

mmm = model.Meridian(
    input_data=data,
    model_spec=model_spec
)

health = reviewer.ModelReviewer(mmm).run()

filename = "health_card.html"
health.output_model_health_card(
    filename=filename,
    filepath="."
)

IPython.display.HTML(filename=filename)