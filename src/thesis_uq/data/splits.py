from sklearn.model_selection import train_test_split

def train_valid_test_split(X, y, seed=42, temporal=False, presplit=None):
    if presplit is not None:
        # Pre-defined split boundaries (e.g., CDR dataset)
        n_train, n_valid = presplit
        X_train, y_train = X[:n_train], y[:n_train]
        X_valid, y_valid = X[n_train:n_train+n_valid], y[n_train:n_train+n_valid]
        X_test, y_test = X[n_train+n_valid:], y[n_train+n_valid:]
        return X_train, y_train, X_valid, y_valid, X_test, y_test

    if temporal:
        n = len(y)
        n_train = int(n * 0.8)
        n_valid = int(n * 0.1)
        X_train, y_train = X[:n_train], y[:n_train]
        X_valid, y_valid = X[n_train:n_train+n_valid], y[n_train:n_train+n_valid]
        X_test, y_test = X[n_train+n_valid:], y[n_train+n_valid:]
        return X_train, y_train, X_valid, y_valid, X_test, y_test

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return X_train, y_train, X_valid, y_valid, X_test, y_test
