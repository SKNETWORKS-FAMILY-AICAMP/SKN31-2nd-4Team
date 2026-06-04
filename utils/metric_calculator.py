def calculate_metrics(df):

    total_expiration_users = len(df)

    total_risk_revenue_nt = int(
        df['실제_결제금액'].sum()
    )

    avg_subscription_fee_nt = int(
        df['실제_결제금액'].mean()
    )

    actual_churn_users = len(
        df[df['이탈여부'] == 1]
    )

    expected_saved_users = int(
        actual_churn_users * 0.20
    )

    expected_recovery_value_nt = int(
        expected_saved_users *
        avg_subscription_fee_nt
    )

    total_risk_revenue_krw = int(
        total_risk_revenue_nt * 42
    )

    avg_subscription_fee_krw = int(
        avg_subscription_fee_nt * 42
    )

    expected_recovery_value_krw = int(
        expected_recovery_value_nt * 42
    )

    return {
        "total_users": total_expiration_users,
        "risk_nt": total_risk_revenue_nt,
        "risk_krw": total_risk_revenue_krw,
        "avg_fee_nt": avg_subscription_fee_nt,
        "avg_fee_krw": avg_subscription_fee_krw,
        "recovery_nt": expected_recovery_value_nt,
        "recovery_krw": expected_recovery_value_krw
    }