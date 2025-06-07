import pandas as pd
import numpy as np

def simulate_ab_data(num_users=10000, effect_size=0.1, seed=42):
    np.random.seed(seed)

    # Create user IDs
    users = pd.DataFrame({'user_id': range(1, num_users + 1)})

    # Assign user segment randomly: new (0) or returning (1)
    users['customer_type'] = np.random.choice([0, 1], size=num_users, p=[0.6, 0.4])

    # Assign groups randomly: control (0), treatment (1)
    users['group'] = np.random.choice([0, 1], size=num_users)

    # Base average order value for control group
    base_aov = 100

    # Effect: increase in AOV for treatment group
    users['AOV'] = base_aov + np.random.normal(0, 20, num_users) + users['group'] * effect_size * base_aov

    # Conversion rate probabilities (e.g., 0.2 for control, slightly higher for treatment)
    users['conversion_prob'] = 0.2 + users['group'] * 0.05

    # Conversion outcome (1 or 0)
    users['converted'] = np.random.binomial(1, users['conversion_prob'])

    # Session length (minutes)
    users['session_length'] = np.random.normal(5, 1.5, num_users) + users['group'] * 0.5
    users['session_length'] = users['session_length'].clip(lower=0.5)  # No negative values

    # Click-through rate (CTR)
    users['click_through'] = np.random.binomial(1, 0.3 + users['group'] * 0.1, num_users)

    return users

if __name__ == "__main__":
    df = simulate_ab_data()
    print(df.head())
    df.to_csv("ab_test_data.csv", index=False)
    print("✅ Dummy data saved to ab_test_data.csv")
