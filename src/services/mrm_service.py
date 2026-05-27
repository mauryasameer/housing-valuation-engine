import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the Variance Inflation Factor (VIF) for numerical features

    to check for multicollinearity (VIF > 5-10 indicates high collinearity).

    Args:
        df: Feature DataFrame (scaled or unscaled).

    Returns:
        DataFrame sorted by VIF descending.
    """
    # Clean inputs: select only numeric columns and drop constants if they already exist
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_clean = df[numeric_cols].dropna()

    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_clean.columns

    # Add constant for VIF computation
    X = sm.add_constant(df_clean)
    vifs = []
    for i in range(len(df_clean.columns)):
        # Column index is i+1 because index 0 is the constant column we added
        try:
            val = variance_inflation_factor(X.values, i + 1)
            vifs.append(val)
        except Exception:
            vifs.append(np.nan)

    vif_data["VIF"] = vifs
    return vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)


def run_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> dict:
    """Perform residual tests to audit basic assumptions of OLS/Regularized Regression.

    Args:
        y_true: Actual targets.
        y_pred: Predicted targets.
        X: Feature matrix.

    Returns:
        Dictionary of test statistics and pass/fail statuses.
    """
    residuals = y_true - y_pred

    # 1. Durbin-Watson Autocorrelation Test (ideal range: 1.5 to 2.5)
    dw_stat = durbin_watson(residuals)
    dw_status = (
        "Pass (No Autocorrelation)" if 1.5 <= dw_stat <= 2.5 else "Warning (Potential Autocorrelation)"
    )

    # 2. Jarque-Bera Normality Test (null hypothesis: residuals are normally distributed)
    jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)
    jb_status = "Pass (Normally Distributed)" if jb_p > 0.05 else "Fail (Non-Normal Residuals)"

    # 3. Breusch-Pagan Heteroscedasticity Test (null hypothesis: constant variance / homoscedastic)
    # Check if X is 1D or 2D and add a constant
    if len(X.shape) == 1:
        X_const = sm.add_constant(X.reshape(-1, 1))
    else:
        X_const = sm.add_constant(X)

    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_const)
        bp_status = "Pass (Homoscedastic)" if bp_p > 0.05 else "Fail (Heteroscedasticity Detected)"
    except Exception:
        bp_stat, bp_p = np.nan, np.nan
        bp_status = "Error executing Breusch-Pagan test"

    return {
        "durbin_watson": {"statistic": float(dw_stat), "status": dw_status},
        "jarque_bera": {
            "statistic": float(jb_stat),
            "p_value": float(jb_p),
            "skewness": float(skew),
            "kurtosis": float(kurtosis),
            "status": jb_status,
        },
        "breusch_pagan": {
            "statistic": float(bp_stat) if not np.isnan(bp_stat) else None,
            "p_value": float(bp_p) if not np.isnan(bp_p) else None,
            "status": bp_status,
        },
    }


def audit_model_stability(train_r2: float, test_r2: float, train_mse: float, test_mse: float) -> dict:
    """Analyze the difference in metrics between training and test sets

    to audit overfitting risks.

    Args:
        train_r2: Training set R-squared.
        test_r2: Testing set R-squared.
        train_mse: Training set Mean Squared Error.
        test_mse: Testing set Mean Squared Error.

    Returns:
        Dictionary of stability assessment.
    """
    r2_diff = train_r2 - test_r2
    mse_pct_increase = ((test_mse - train_mse) / train_mse) * 100 if train_mse > 0 else 0.0

    status = "Stable"
    risk_level = "Low"
    recommendations = []

    if r2_diff > 0.08:
        status = "Overfitting"
        risk_level = "High"
        recommendations.append("Increase L1/L2 regularization strength (raise alpha/lambda).")
        recommendations.append("Perform further feature pruning to remove weak predictors.")
    elif r2_diff < -0.05:
        status = "Instability"
        risk_level = "Medium"
        recommendations.append("Check for distribution mismatch or outliers in train-test splits.")

    if mse_pct_increase > 15.0:
        status = "Overfitting"
        risk_level = "High" if risk_level == "High" else "Medium"
        recommendations.append("Test set error is significantly higher than training error.")

    return {
        "status": status,
        "risk_level": risk_level,
        "r2_difference": float(r2_diff),
        "mse_pct_increase": float(mse_pct_increase),
        "recommendations": recommendations,
    }
