import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_ab_test(data_path="ab_test_data.csv"):
    # Load data
    df = pd.read_csv(data_path)

    # Create 'plots' folder if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Separate groups
    control = df[df['group'] == 0]
    treatment = df[df['group'] == 1]

    print("=== Summary Statistics by Group ===")
    print(df.groupby('group')[['AOV', 'session_length', 'click_through', 'converted']].agg(['mean', 'std', 'median']))

    # 1. AOV difference - independent t-test
    t_stat, p_val = stats.ttest_ind(treatment['AOV'], control['AOV'])
    print(f"\nAOV t-test: t={t_stat:.3f}, p={p_val:.3f}")

    # 2. Conversion rate difference - chi-square test
    contingency_table = pd.crosstab(df['group'], df['converted'])
    chi2, p_chi, _, _ = stats.chi2_contingency(contingency_table)
    print(f"Conversion rate chi-square test: chi2={chi2:.3f}, p={p_chi:.3f}")

    # 3. Session length difference - independent t-test
    t_stat_sess, p_val_sess = stats.ttest_ind(treatment['session_length'], control['session_length'])
    print(f"Session Length t-test: t={t_stat_sess:.3f}, p={p_val_sess:.3f}")

    # 4. Click-through rate difference - chi-square test
    contingency_click = pd.crosstab(df['group'], df['click_through'])
    chi2_click, p_click, _, _ = stats.chi2_contingency(contingency_click)
    print(f"Click-through rate chi-square test: chi2={chi2_click:.3f}, p={p_click:.3f}")

    sns.set(style="whitegrid")

    # Plot 1: AOV distribution by group
    plt.figure(figsize=(8,6))
    sns.histplot(data=df, x='AOV', hue='group', kde=True, stat="density", common_norm=False, palette="muted")
    plt.title("AOV Distribution by Group")
    plt.xlabel("Average Order Value")
    plt.legend(labels=['Control', 'Treatment'])
    plt.tight_layout()
    plt.savefig("plots/aov_distribution.png")
    plt.show()

    # Plot 2: Conversion Rate by Group (bar plot)
    plt.figure(figsize=(6,6))
    conv_rates = df.groupby('group')['converted'].mean()
    sns.barplot(x=conv_rates.index, y=conv_rates.values, palette="muted")
    plt.xticks([0,1], ['Control', 'Treatment'])
    plt.ylim(0, 1)
    plt.title("Conversion Rate by Group")
    plt.ylabel("Conversion Rate")
    plt.tight_layout()
    plt.savefig("plots/conversion_rate.png")
    plt.show()

    # Plot 3: Session Length distribution by group
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df, x='group', y='session_length', palette="muted")
    plt.xticks([0,1], ['Control', 'Treatment'])
    plt.title("Session Length Distribution by Group")
    plt.ylabel("Session Length (minutes)")
    plt.tight_layout()
    plt.savefig("plots/session_length_distribution.png")
    plt.show()

    # Plot 4: Click-through Rate by Group (bar plot)
    plt.figure(figsize=(6,6))
    click_rates = df.groupby('group')['click_through'].mean()
    sns.barplot(x=click_rates.index, y=click_rates.values, palette="muted")
    plt.xticks([0,1], ['Control', 'Treatment'])
    plt.ylim(0, 1)
    plt.title("Click-through Rate by Group")
    plt.ylabel("Click-through Rate")
    plt.tight_layout()
    plt.savefig("plots/click_through_rate.png")
    plt.show()

    # Correlation heatmap for numeric columns
    plt.figure(figsize=(8, 6))
    corr = df[['AOV', 'session_length', 'click_through', 'converted']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap of Key Metrics")
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png")
    plt.show()

if __name__ == "__main__":
    analyze_ab_test()
