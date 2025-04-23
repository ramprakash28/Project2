import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def generate_data(N, means, bias, scale, range_min, range_max, seed, multicollinear):
    np.random.seed(seed)
    n_features = len(means)

    # Generate base data
    X, y = make_classification(
        n_samples=N,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=seed,
        shuffle=False
    )

    # Apply scaling and shifting
    for i in range(X.shape[1]):
        X[:, i] = X[:, i] * scale + means[i % len(means)]

    # Clip values
    X = np.clip(X, range_min, range_max)

    if multicollinear:
        # Adding multicollinearity by duplicating and modifying a few columns
        col1 = X[:, 0]
        col2 = X[:, 1]
        noise = np.random.normal(0, 0.01, size=N)
        new_col1 = 0.95 * col1 + 0.05 * col2 + noise
        new_col2 = 0.5 * col1 - 0.4 * col2 + noise

        X = np.column_stack((X, new_col1, new_col2))

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, required=True, help="Number of samples")
    parser.add_argument("-m", nargs='+', type=float, required=True, help="List of means")
    parser.add_argument("-b", type=float, required=True, help="Bias")
    parser.add_argument("-scale", type=float, required=True, help="Scale")
    parser.add_argument("-rnge", nargs=2, type=float, required=True, help="Min and max range")
    parser.add_argument("-seed", type=int, required=True, help="Random seed")
    parser.add_argument("-output_file", type=str, required=True, help="Output CSV file")
    parser.add_argument("--multicollinear", action="store_true", help="Add multicollinearity to the dataset")

    args = parser.parse_args()

    df = generate_data(
        N=args.N,
        means=args.m,
        bias=args.b,
        scale=args.scale,
        range_min=args.rnge[0],
        range_max=args.rnge[1],
        seed=args.seed,
        multicollinear=args.multicollinear
    )

    df.to_csv(args.output_file, index=False)
    print(f"✅ Data saved to {args.output_file}")

if __name__ == "__main__":
    main()
