"""A/B-тест новой формы оплаты ЖКХ: очистка данных, воронка, статтесты."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

INPUT_FILE = "data.xlsx"
SHEET_USERS = "Пользователи"
SHEET_PAYMENTS = "Платежи"
STEP_COLS = ["step1_opened", "step2_entered", "step3_confirmed", "step4_success"]
OUTPUT_DIR = Path("output")
ALPHA = 0.05


def load_and_clean(path: str):
    users = pd.read_excel(path, sheet_name=SHEET_USERS)
    payments = pd.read_excel(path, sheet_name=SHEET_PAYMENTS)

    log = {"users_raw": len(users), "payments_raw": len(payments)}

    no_group = users["group"].isna()
    log["users_no_group"] = int(no_group.sum())
    users = users[~no_group]

    bad_age = (users["age"] < 0) | (users["age"] > 120)
    log["users_bad_age"] = int(bad_age.sum())
    users = users[~bad_age].copy()

    log["users_missing_city_kept"] = int(users["city"].isna().sum())

    no_uid = payments["user_id"].isna()
    log["payments_no_user_id"] = int(no_uid.sum())
    payments = payments[~no_uid].copy()
    payments["user_id"] = payments["user_id"].astype(int)

    group_map = users.set_index("user_id")["group"]
    true_group = payments["user_id"].map(group_map)
    log["payments_group_fixed"] = int((payments["group"] != true_group).sum())
    payments["group"] = true_group

    orphaned = payments["group"].isna()
    log["payments_orphaned"] = int(orphaned.sum())
    payments = payments[~orphaned].copy()

    log["users_clean"] = len(users)
    log["payments_clean"] = len(payments)
    return users, payments, log


def print_cleaning_log(log: dict) -> None:
    print(f"Пользователи: {log['users_raw']} -> {log['users_clean']}")
    print(f"  без группы: {log['users_no_group']}")
    print(f"  некорректный возраст: {log['users_bad_age']}")
    print(f"  пустой город (оставлен): {log['users_missing_city_kept']}")
    print(f"Платежи: {log['payments_raw']} -> {log['payments_clean']}")
    print(f"  без user_id: {log['payments_no_user_id']}")
    print(f"  расхождение группы (исправлено по листу «Пользователи»): "
          f"{log['payments_group_fixed']}")
    print()


def build_user_funnel(users: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    """Агрегация до уровня пользователя: успех = достиг шага хотя бы
    в одной из своих попыток. Это единица рандомизации, поэтому и
    единица анализа должна быть той же — иначе наблюдения не независимы.
    """
    reached = payments[STEP_COLS].notna()
    reached["user_id"] = payments["user_id"].values
    per_user = reached.groupby("user_id")[STEP_COLS].max()
    attempts = payments.groupby("user_id").size().rename("n_attempts")

    first_attempt = payments.sort_values("payment_id").drop_duplicates("user_id", keep="first")
    first_success = first_attempt.set_index("user_id")["step4_success"].notna().rename("first_attempt_success")

    funnel = (
        users.set_index("user_id")
        .join(per_user)
        .join(attempts)
        .join(first_success)
    )
    funnel[STEP_COLS] = funnel[STEP_COLS].fillna(False)
    funnel["n_attempts"] = funnel["n_attempts"].fillna(0).astype(int)
    funnel["first_attempt_success"] = funnel["first_attempt_success"].fillna(False)
    return funnel.reset_index()


def funnel_summary(funnel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, sub in funnel.groupby("group"):
        n = len(sub)
        row = {"group": group, "n_users": n}
        for i, col in enumerate(STEP_COLS, start=1):
            row[f"step{i}_n"] = int(sub[col].sum())
            row[f"step{i}_pct"] = 100 * sub[col].sum() / n
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def wilson_ci(successes: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def two_proportion_ztest(s1: int, n1: int, s2: int, n2: int) -> tuple[float, float]:
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value


def diff_ci(s1: int, n1: int, s2: int, n2: int, alpha: float = ALPHA):
    p1, p2 = s1 / n1, s2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = stats.norm.ppf(1 - alpha / 2)
    diff = p2 - p1
    return diff, diff - z * se, diff + z * se


def chi_square_test(s1: int, n1: int, s2: int, n2: int) -> tuple[float, float]:
    table = np.array([[s1, n1 - s1], [s2, n2 - s2]])
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    return chi2, p


def srm_check(n1: int, n2: int):
    total = n1 + n2
    return stats.chisquare([n1, n2], f_exp=[total / 2, total / 2])


def plot_funnel(summary: pd.DataFrame, out_path: Path) -> None:
    labels = ["Открыл форму", "Ввёл реквизиты", "Подтвердил платёж", "Успешная оплата"]
    x = np.arange(len(labels))
    width = 0.35
    colors = {"A": "#9aa5b1", "B": "#2f6fed"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, group in enumerate(summary.index):
        values = [summary.loc[group, f"step{j}_pct"] for j in range(1, 5)]
        offset = (i - (len(summary.index) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=group, color=colors.get(group))
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%",
                    ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Доля пользователей, %")
    ax.set_title("Воронка оплаты ЖКХ по группам (на уровне пользователей)")
    ax.set_ylim(0, 110)
    ax.legend(title="Группа")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_final_conversion(summary: pd.DataFrame, out_path: Path) -> None:
    colors = {"A": "#9aa5b1", "B": "#2f6fed"}
    labels, values, err_low, err_high = [], [], [], []
    for group in summary.index:
        s, n = summary.loc[group, "step4_n"], summary.loc[group, "n_users"]
        lo, hi = wilson_ci(s, n)
        p = s / n
        labels.append(group)
        values.append(100 * p)
        err_low.append(100 * (p - lo))
        err_high.append(100 * (hi - p))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    bars = ax.bar(labels, values, yerr=[err_low, err_high], capsize=8,
                   color=[colors.get(g) for g in labels])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value:.1f}%",
                ha="center", fontweight="bold")
    ax.set_ylabel("Конверсия шаг1→шаг4, %")
    ax.set_title("Итоговая конверсия по группам (95% ДИ, Wilson)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_guardrails(funnel: pd.DataFrame) -> None:
    multi_attempt = funnel.groupby("group")["n_attempts"].apply(lambda s: 100 * (s > 1).mean())
    print("Доля пользователей с >1 попыткой оплаты, %:")
    print(multi_attempt.round(1).to_string())
    print()

    by_device = (
        funnel.groupby(["device_type", "group"])["step4_success"]
        .mean()
        .mul(100)
        .unstack()
    )
    print("Конверсия в шаг 4 по устройствам, %:")
    print(by_device.round(1))
    print()

    first_attempt_conv = funnel.groupby("group")["first_attempt_success"].mean().mul(100)
    print("Конверсия по первой попытке (без учёта повторов), %:")
    print(first_attempt_conv.round(1).to_string())
    print()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    users, payments, log = load_and_clean(INPUT_FILE)
    print_cleaning_log(log)

    n_a = int((users["group"] == "A").sum())
    n_b = int((users["group"] == "B").sum())
    srm_chi2, srm_p = srm_check(n_a, n_b)
    print(f"SRM-проверка: A={n_a}, B={n_b}, chi2={srm_chi2:.4f}, p={srm_p:.4f}\n")

    funnel = build_user_funnel(users, payments)
    summary = funnel_summary(funnel)
    print(summary[[f"step{i}_pct" for i in range(1, 5)]].round(2))
    print()

    s_a, n_a = summary.loc["A", "step4_n"], summary.loc["A", "n_users"]
    s_b, n_b = summary.loc["B", "step4_n"], summary.loc["B", "n_users"]

    z, p_value = two_proportion_ztest(s_a, n_a, s_b, n_b)
    chi2, p_chi2 = chi_square_test(s_a, n_a, s_b, n_b)
    diff, diff_lo, diff_hi = diff_ci(s_a, n_a, s_b, n_b)
    ci_a, ci_b = wilson_ci(s_a, n_a), wilson_ci(s_b, n_b)

    print(f"Конверсия A: {100*s_a/n_a:.2f}% ({s_a}/{n_a}), "
          f"95% ДИ [{100*ci_a[0]:.1f}; {100*ci_a[1]:.1f}]")
    print(f"Конверсия B: {100*s_b/n_b:.2f}% ({s_b}/{n_b}), "
          f"95% ДИ [{100*ci_b[0]:.1f}; {100*ci_b[1]:.1f}]")
    print(f"Прирост: {100*diff:.2f} п.п. ({100*diff/(s_a/n_a):.1f}% относительно A)")
    print(f"95% ДИ разности: [{100*diff_lo:.2f}; {100*diff_hi:.2f}] п.п.")
    print(f"Z-тест: z = {z:.3f}, p = {p_value:.3e}")
    print(f"Хи-квадрат: chi2 = {chi2:.3f}, p = {p_chi2:.3e}")
    print(f"Проверка: z^2 = {z**2:.3f} (должно совпадать с chi2)\n")

    print_guardrails(funnel)

    plot_funnel(summary, OUTPUT_DIR / "funnel_by_group.png")
    plot_final_conversion(summary, OUTPUT_DIR / "final_conversion_ci.png")


if __name__ == "__main__":
    main()
