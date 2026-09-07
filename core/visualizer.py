"""
NexInsight - Automatic Plotly Visualization Engine
Generates modern, interactive, responsive charts derived dynamically
from column types without hardcoding, matching the premium fintech light SaaS aesthetic.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class Visualizer:
    """Generates cohesive, premium SaaS-styled Plotly charts."""

    THEME = {
        "bg_color": "rgba(255, 255, 255, 0)",
        "paper_color": "rgba(255, 255, 255, 0)",
        "font_family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "font_color": "#111111",
        "grid_color": "#F3F4F6",
        "border_color": "#E5E7EB",
        "text_secondary": "#6B7280",
        "primary_blue": "#2563EB",
        "bright_blue": "#3B82F6",
        "purple": "#8B5CF6",
        "pink": "#EC4899",
        "green": "#22C55E",
        "orange": "#F59E0B",
        "red": "#EF4444",
        "colors": ["#2563EB", "#EC4899", "#22C55E", "#8B5CF6", "#3B82F6", "#F59E0B", "#06B6D4", "#64748B"]
    }

    @classmethod
    def apply_saas_theme(cls, fig: go.Figure, title: str = "", height: int = 340) -> go.Figure:
        """Applies a clean, minimal, premium fintech light theme with refined typography."""
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=13, color="#111111", family=cls.THEME["font_family"], weight=600),
                x=0.01,
                y=0.96
            ),
            paper_bgcolor=cls.THEME["paper_color"],
            plot_bgcolor=cls.THEME["bg_color"],
            font=dict(family=cls.THEME["font_family"], color=cls.THEME["font_color"], size=11),
            height=height,
            margin=dict(l=35, r=25, t=45, b=35),
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="#111111",
                font_size=11,
                font_color="#FFFFFF",
                font_family=cls.THEME["font_family"],
                bordercolor="#111111"
            ),
            colorway=cls.THEME["colors"]
        )
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=cls.THEME["grid_color"],
            zeroline=False,
            linecolor=cls.THEME["border_color"],
            tickfont=dict(size=10, color=cls.THEME["text_secondary"])
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=cls.THEME["grid_color"],
            zeroline=False,
            linecolor=cls.THEME["border_color"],
            tickfont=dict(size=10, color=cls.THEME["text_secondary"])
        )
        return fig

    @classmethod
    def create_time_series_chart(cls, df: pd.DataFrame, date_col: str, num_col: str) -> go.Figure:
        """Generates an interactive time series line chart with subtle gradient fill."""
        temp_df = df[[date_col, num_col]].dropna().copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
        temp_df[num_col] = pd.to_numeric(temp_df[num_col], errors="coerce")
        temp_df = temp_df.dropna().sort_values(by=date_col)

        # Aggregate if too many points (> 200) to keep render smooth
        if len(temp_df) > 200:
            temp_df = temp_df.resample("D", on=date_col)[num_col].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=temp_df[date_col],
            y=temp_df[num_col],
            mode="lines",
            name=num_col,
            line=dict(color=cls.THEME["primary_blue"], width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.08)"
        ))

        title = f"{num_col} Trend over Time"
        return cls.apply_saas_theme(fig, title)

    @classmethod
    def create_category_bar_chart(cls, df: pd.DataFrame, cat_col: str, num_col: str, agg: str = "mean") -> go.Figure:
        """Generates a sorted clean bar chart aggregated by category."""
        temp_df = df[[cat_col, num_col]].dropna().copy()
        temp_df[num_col] = pd.to_numeric(temp_df[num_col], errors="coerce")
        temp_df = temp_df.dropna()

        grouped = temp_df.groupby(cat_col)[num_col].agg(agg).reset_index()
        grouped = grouped.sort_values(by=num_col, ascending=False).head(10)

        fig = px.bar(
            grouped,
            x=cat_col,
            y=num_col,
            color_discrete_sequence=[cls.THEME["primary_blue"]],
            text_auto=".2s"
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0, opacity=0.9, marker=dict(cornerradius=4))
        title = f"{agg.capitalize()} {num_col} by {cat_col}"
        return cls.apply_saas_theme(fig, title)

    @classmethod
    def create_scatter_plot(cls, df: pd.DataFrame, num_col1: str, num_col2: str, cat_col: Optional[str] = None) -> go.Figure:
        """Generates an interactive scatter plot with trendline."""
        cols = [num_col1, num_col2]
        if cat_col and cat_col in df.columns:
            cols.append(cat_col)
        
        temp_df = df[cols].dropna().copy()
        temp_df[num_col1] = pd.to_numeric(temp_df[num_col1], errors="coerce")
        temp_df[num_col2] = pd.to_numeric(temp_df[num_col2], errors="coerce")
        temp_df = temp_df.dropna()

        # Limit points if gigantic
        if len(temp_df) > 1500:
            temp_df = temp_df.sample(1500, random_state=42)

        # Check if statsmodels is available for trendline
        use_trendline = None
        if len(temp_df) > 10:
            try:
                import statsmodels
                use_trendline = "ols"
            except ImportError:
                use_trendline = None

        if cat_col and cat_col in temp_df.columns:
            fig = px.scatter(
                temp_df,
                x=num_col1,
                y=num_col2,
                color=cat_col,
                opacity=0.8,
                color_discrete_sequence=cls.THEME["colors"],
                trendline=use_trendline
            )
        else:
            fig = px.scatter(
                temp_df,
                x=num_col1,
                y=num_col2,
                opacity=0.75,
                color_discrete_sequence=[cls.THEME["primary_blue"]],
                trendline=use_trendline
            )

        fig.update_traces(marker=dict(size=7))
        title = f"{num_col1} vs. {num_col2}"
        return cls.apply_saas_theme(fig, title)

    @classmethod
    def create_histogram(cls, df: pd.DataFrame, num_col: str) -> go.Figure:
        """Generates clean distribution histogram."""
        series = pd.to_numeric(df[num_col], errors="coerce").dropna()
        fig = px.histogram(
            x=series,
            nbins=30,
            opacity=0.85,
            color_discrete_sequence=[cls.THEME["bright_blue"]]
        )
        title = f"Distribution: {num_col}"
        fig.update_xaxes(title_text=num_col)
        fig.update_yaxes(title_text="Count")
        fig.update_traces(marker_line_width=1, marker_line_color="#FFFFFF")
        return cls.apply_saas_theme(fig, title)

    @classmethod
    def create_donut_chart(cls, df: pd.DataFrame, cat_col: str) -> go.Figure:
        """Generates a modern donut chart for categorical distribution."""
        val_counts = df[cat_col].dropna().value_counts().head(6).reset_index()
        val_counts.columns = [cat_col, "Count"]

        fig = px.pie(
            val_counts,
            names=cat_col,
            values="Count",
            hole=0.62,
            color_discrete_sequence=[cls.THEME["primary_blue"], cls.THEME["pink"], cls.THEME["green"], cls.THEME["purple"], cls.THEME["orange"], "#64748B"]
        )
        fig.update_traces(
            textinfo="percent+label",
            textposition="outside",
            marker=dict(line=dict(color="#FFFFFF", width=2))
        )
        title = f"Share by {cat_col}"
        return cls.apply_saas_theme(fig, title)

    @classmethod
    def create_correlation_heatmap(cls, corr_df: pd.DataFrame) -> go.Figure:
        """Generates an annotated correlation heatmap with clean light colors."""
        fig = px.imshow(
            corr_df,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Blues",
            zmin=-1.0,
            zmax=1.0
        )
        fig.update_layout(coloraxis_colorbar=dict(title="r", tickfont=dict(size=10, color=cls.THEME["text_secondary"])))
        title = "Correlation Matrix"
        return cls.apply_saas_theme(fig, title, height=360)

    @classmethod
    def create_cluster_scatter(cls, pca_x: List[float], pca_y: List[float], labels: List[str]) -> go.Figure:
        """Generates 2D PCA cluster visualization."""
        cluster_df = pd.DataFrame({"PCA 1": pca_x, "PCA 2": pca_y, "Cluster": labels})
        fig = px.scatter(
            cluster_df,
            x="PCA 1",
            y="PCA 2",
            color="Cluster",
            opacity=0.85,
            color_discrete_sequence=[cls.THEME["primary_blue"], cls.THEME["pink"], cls.THEME["green"], cls.THEME["purple"]]
        )
        fig.update_traces(marker=dict(size=7))
        title = "Behavioral Clusters (PCA Projection)"
        return cls.apply_saas_theme(fig, title)

    @classmethod
    def create_anomaly_scatter(cls, df: pd.DataFrame, num_col: str, anomalies: List[Dict[str, Any]]) -> go.Figure:
        """Visualizes data points with anomalies highlighted in vibrant red."""
        series = pd.to_numeric(df[num_col], errors="coerce").dropna().reset_index()
        series.columns = ["Index", "Value"]

        anomaly_indices = [a["row_index"] for a in anomalies if a["column"] == num_col]
        series["Status"] = series["Index"].isin(anomaly_indices).map({True: "Anomaly", False: "Normal"})

        fig = px.scatter(
            series,
            x="Index",
            y="Value",
            color="Status",
            color_discrete_map={"Normal": "#94A3B8", "Anomaly": cls.THEME["red"]},
            opacity=0.85
        )
        fig.update_traces(marker=dict(size=7))
        title = f"Outlier Inspection: {num_col}"
        return cls.apply_saas_theme(fig, title)

    @classmethod
    def auto_generate_dashboard_charts(cls, df: pd.DataFrame, column_types: Dict[str, str]) -> List[go.Figure]:
        """
        Dynamically assembles primary and supporting charts based purely on available column combinations.
        Never assumes specific column names!
        """
        charts = []
        date_cols = [c for c, t in column_types.items() if t == "Date" and c in df.columns]
        num_cols = [c for c, t in column_types.items() if t == "Numeric" and c in df.columns]
        cat_cols = [c for c, t in column_types.items() if t in ["Categorical", "Boolean"] and c in df.columns]

        # 1. Date + Numeric (Primary Trend Line)
        if date_cols and num_cols:
            try:
                charts.append(cls.create_time_series_chart(df, date_cols[0], num_cols[0]))
            except Exception:
                pass

        # 2. Categorical Distribution (Donut or Horizontal Bar)
        if cat_cols:
            try:
                charts.append(cls.create_donut_chart(df, cat_cols[0]))
            except Exception:
                pass

        # 3. Categorical + Numeric (Bar Chart)
        if cat_cols and num_cols:
            try:
                best_cat = None
                for c in cat_cols:
                    n_unq = df[c].nunique()
                    if 2 <= n_unq <= 15:
                        best_cat = c
                        break
                best_cat = best_cat or cat_cols[0]
                target_num = num_cols[1] if len(num_cols) > 1 else num_cols[0]
                charts.append(cls.create_category_bar_chart(df, best_cat, target_num, agg="mean"))
            except Exception:
                pass

        # 4. Numeric + Numeric (Scatter Plot)
        if len(num_cols) >= 2:
            try:
                cat_for_color = cat_cols[0] if cat_cols else None
                charts.append(cls.create_scatter_plot(df, num_cols[0], num_cols[1], cat_for_color))
            except Exception:
                pass

        # 5. Single Numeric Distribution (Histogram)
        if num_cols:
            try:
                target_num = num_cols[-1]
                charts.append(cls.create_histogram(df, target_num))
            except Exception:
                pass

        return charts
