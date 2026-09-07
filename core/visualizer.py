"""
NexInsight - Automatic Plotly Visualization Engine
Generates modern, interactive, responsive charts derived dynamically
from column types without hardcoding.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class Visualizer:
    """Generates cohesive, SaaS-styled Plotly charts."""

    THEME = {
        "bg_color": "rgba(15, 23, 42, 0.4)",  # slate-900 transparent
        "paper_color": "rgba(0, 0, 0, 0)",
        "font_family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "font_color": "#F1F5F9",
        "grid_color": "rgba(255, 255, 255, 0.08)",
        "colors": ["#6366F1", "#06B6D4", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6", "#3B82F6", "#14B8A6"]
    }

    @classmethod
    def apply_saas_theme(cls, fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
        """Applies a consistent sleek dark-neutral theme with refined typography and margins."""
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=14, color="#F8FAFC", family=cls.THEME["font_family"], weight=600),
                x=0.02,
                y=0.96
            ),
            paper_bgcolor=cls.THEME["paper_color"],
            plot_bgcolor=cls.THEME["bg_color"],
            font=dict(family=cls.THEME["font_family"], color=cls.THEME["font_color"], size=12),
            height=height,
            margin=dict(l=40, r=30, t=50, b=40),
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="#1E293B",
                font_size=12,
                font_family=cls.THEME["font_family"],
                bordercolor="#334155"
            ),
            colorway=cls.THEME["colors"]
        )
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=cls.THEME["grid_color"],
            zeroline=False,
            linecolor="rgba(255, 255, 255, 0.15)",
            tickfont=dict(size=11, color="#94A3B8")
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=cls.THEME["grid_color"],
            zeroline=False,
            linecolor="rgba(255, 255, 255, 0.15)",
            tickfont=dict(size=11, color="#94A3B8")
        )
        return fig

    @classmethod
    def create_time_series_chart(cls, df: pd.DataFrame, date_col: str, num_col: str) -> go.Figure:
        """Generates an interactive time series line chart with gradient fill."""
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
            line=dict(color="#6366F1", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.12)"
        ))

        title = f"Temporal Trend: {num_col} over {date_col}"
        return cls.apply_saas_theme(fig, title)

    @classmethod
    def create_category_bar_chart(cls, df: pd.DataFrame, cat_col: str, num_col: str, agg: str = "mean") -> go.Figure:
        """Generates a sorted horizontal or vertical bar chart aggregated by category."""
        temp_df = df[[cat_col, num_col]].dropna().copy()
        temp_df[num_col] = pd.to_numeric(temp_df[num_col], errors="coerce")
        temp_df = temp_df.dropna()

        grouped = temp_df.groupby(cat_col)[num_col].agg(agg).reset_index()
        grouped = grouped.sort_values(by=num_col, ascending=False).head(12)

        fig = px.bar(
            grouped,
            x=cat_col,
            y=num_col,
            color=num_col,
            color_continuous_scale=["#38BDF8", "#6366F1", "#A855F7"],
            text_auto=".2s"
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0, opacity=0.9)
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
                trendline=use_trendline
            )
        else:
            fig = px.scatter(
                temp_df,
                x=num_col1,
                y=num_col2,
                opacity=0.8,
                color_discrete_sequence=["#06B6D4"],
                trendline=use_trendline
            )

        title = f"Correlation: {num_col1} vs. {num_col2}"
        return cls.apply_saas_theme(fig, title)

    @classmethod
    def create_histogram(cls, df: pd.DataFrame, num_col: str) -> go.Figure:
        """Generates distribution histogram with KDE or density indicators."""
        series = pd.to_numeric(df[num_col], errors="coerce").dropna()
        fig = px.histogram(
            x=series,
            nbins=35,
            marginal="box",
            opacity=0.85,
            color_discrete_sequence=["#10B981"]
        )
        title = f"Distribution: {num_col}"
        fig.update_xaxes(title_text=num_col)
        fig.update_yaxes(title_text="Count")
        return cls.apply_saas_theme(fig, title)

    @classmethod
    def create_donut_chart(cls, df: pd.DataFrame, cat_col: str) -> go.Figure:
        """Generates a modern donut chart for categorical distribution."""
        val_counts = df[cat_col].dropna().value_counts().head(8).reset_index()
        val_counts.columns = [cat_col, "Count"]

        fig = px.pie(
            val_counts,
            names=cat_col,
            values="Count",
            hole=0.55,
            color_discrete_sequence=cls.THEME["colors"]
        )
        fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color="#0F172A", width=2)))
        title = f"Category Share: {cat_col}"
        return cls.apply_saas_theme(fig, title)

    @classmethod
    def create_correlation_heatmap(cls, corr_df: pd.DataFrame) -> go.Figure:
        """Generates an annotated correlation heatmap."""
        fig = px.imshow(
            corr_df,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0
        )
        fig.update_layout(coloraxis_colorbar=dict(title="Correlation"))
        title = "Pearson Correlation Matrix"
        return cls.apply_saas_theme(fig, title, height=420)

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
            color_discrete_sequence=cls.THEME["colors"]
        )
        title = "K-Means Cluster Distribution (PCA Projection)"
        return cls.apply_saas_theme(fig, title)

    @classmethod
    def create_anomaly_scatter(cls, df: pd.DataFrame, num_col: str, anomalies: List[Dict[str, Any]]) -> go.Figure:
        """Visualizes data points with anomalies highlighted in vibrant red/amber."""
        series = pd.to_numeric(df[num_col], errors="coerce").dropna().reset_index()
        series.columns = ["Index", "Value"]

        anomaly_indices = [a["row_index"] for a in anomalies if a["column"] == num_col]
        series["Is_Anomaly"] = series["Index"].isin(anomaly_indices)

        fig = px.scatter(
            series,
            x="Index",
            y="Value",
            color="Is_Anomaly",
            color_discrete_map={False: "#475569", True: "#EF4444"},
            opacity=0.8
        )
        title = f"Anomaly Inspection: {num_col}"
        return cls.apply_saas_theme(fig, title)

    @classmethod
    def auto_generate_dashboard_charts(cls, df: pd.DataFrame, column_types: Dict[str, str]) -> List[go.Figure]:
        """
        Dynamically assembles 4 to 6 the most informative charts
        based purely on available column combinations.
        Never assumes specific column names!
        """
        charts = []
        date_cols = [c for c, t in column_types.items() if t == "Date" and c in df.columns]
        num_cols = [c for c, t in column_types.items() if t == "Numeric" and c in df.columns]
        cat_cols = [c for c, t in column_types.items() if t in ["Categorical", "Boolean"] and c in df.columns]

        # 1. Date + Numeric (Time Series)
        if date_cols and num_cols:
            try:
                charts.append(cls.create_time_series_chart(df, date_cols[0], num_cols[0]))
            except Exception:
                pass

        # 2. Categorical + Numeric (Bar Chart)
        if cat_cols and num_cols:
            try:
                # Pick categorical column with good variance (3 to 15 unique values)
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

        # 3. Numeric + Numeric (Scatter Plot)
        if len(num_cols) >= 2:
            try:
                cat_for_color = cat_cols[0] if cat_cols else None
                charts.append(cls.create_scatter_plot(df, num_cols[0], num_cols[1], cat_for_color))
            except Exception:
                pass

        # 4. Categorical Distribution (Donut Chart)
        if cat_cols:
            try:
                charts.append(cls.create_donut_chart(df, cat_cols[0]))
            except Exception:
                pass

        # 5. Single Numeric Distribution (Histogram)
        if num_cols:
            try:
                # Pick numeric column not already primary
                target_num = num_cols[-1]
                charts.append(cls.create_histogram(df, target_num))
            except Exception:
                pass

        # 6. Second numeric distribution or another cat/num combo if needed
        if len(num_cols) >= 3 and len(charts) < 6:
            try:
                charts.append(cls.create_histogram(df, num_cols[1]))
            except Exception:
                pass

        return charts
