import pandas as pd
import numpy as np

class MondrianAnonymizer:
    def __init__(self, k=10):
        self.k = k
        self.partitions = []
        self.categorical_features = []
        self.feature_spans = {} # Stores ranges/sets for each feature in each partition

    def fit(self, X, categorical_features=None):
        """
        X: pandas DataFrame
        categorical_features: list of column names
        """
        self.partitions = []
        self.categorical_features = categorical_features if categorical_features else []
        
        # Initial partition index - we will start with one partition containing all indices
        initial_indices = X.index.to_numpy()
        self._partition(X, initial_indices)
        
        return self

    def _partition(self, df, indices):
        """
        Recursively split the dataframe indices until size < 2*k
        """
        if len(indices) < 2 * self.k:
            self.partitions.append(indices)
            return

        # Choose dimension to split
        # Heuristic: Choose dimension with widest normalized span
        # Simple heuristic here: just iterate or choose distinct count max
        
        # Let's find columns with valid splits
        valid_cols = []
        for col in df.columns:
            # Check unique values
            unique_vals = df.loc[indices, col].nunique()
            if unique_vals > 1:
                valid_cols.append(col)
        
        if not valid_cols:
             self.partitions.append(indices)
             return

        # Pick a column to split (random or max variance/span)
        # For Mondrian, traditionally choosing the dimension with widest normalized range
        split_col = np.random.choice(valid_cols) # Simplified for now
        
        # Split value
        if split_col in self.categorical_features:
            # Categorical split: simpler to just group? Mondrian strictly works on Order.
            # Adaptation for categorical:
            # We will rely on random split for now or we treat categorical as unordered
            # Actually standard Mondrian requires ordered values. 
            # If we don't have order, we can't easily "split" by median.
            # Fallback: Just append to partition if we hit categorical roadblock or handle numeric only?
            # User wants k-anonymity. Let's stick to splitting Numeric columns primarily 
            # and maybe try to split Categorical if we can define an order or just randomly split the set in two.
            
            # Better approach for this baseline: Only split on Numeric QIs? 
            # Or assume frequency based order.
             self.partitions.append(indices) 
             return
        else:
            # Numeric Median Split
            values = df.loc[indices, split_col]
            median = values.median()
            
            lhs = values[values <= median].index
            rhs = values[values > median].index
            
            if len(lhs) < self.k or len(rhs) < self.k:
                 self.partitions.append(indices) 
                 return
            
            self._partition(df, lhs.to_numpy())
            self._partition(df, rhs.to_numpy())

    def transform(self, X):
        """
        Apply generalization to X.
        For Training Data (which was fitted): Replace values with partition representatives.
        For Test Data: Find which partition representative they match closest? 
        Strict Mondrian doesn't really "transform" unseen test data easily without leaking.
        
        Standard approach for ML experiments:
        Anonymize the TRAIN set. Train model on generalized features.
        Generalize TEST set using the SAME ranges? 
        Or just mapping to the "representative" of the span it falls into.
        """
        X_out = X.copy()
        # Ensure numeric columns are float to avoid int coercion warnings when setting means
        for col in X_out.columns:
            if col not in self.categorical_features and pd.api.types.is_numeric_dtype(X_out[col]):
                 X_out[col] = X_out[col].astype(float)
        
        # Pre-compute representatives for each partition
        # For numeric: Mean or Range string
        # For categorical: Mode or Set string
        
        # Map indices to partition ID
        # Since we only stored indices in fit(), this transform only works on the Training Set X used in fit().
        # Handling new Test data is tricky with purely index-based Mondrian.
        # We need to store the Split Rules (The Tree) to apply to Test data.
        
        # SIMPLIFICATION:
        # We will only implement 'fit_transform' style for the specific experiment loop
        # But we need to handle Train/Test split.
        # Correct way: Anonymize the WHOLE dataset (Train+Test) together? NO, that leaks info.
        # Correct way: Anonymize Train. Use the resultant "Guides" to bucketize Test.
        
        # Let's pivot to a simpler implementation:
        # We will assume we anonymize the training data.
        # For the test data, we will just apply the same "Binning"?
        # Actually, let's just implement fit_transform on the passed dataset.
        # In the experiment loop, we can pass pd.concat([train, test]) to anon, then split back. 
        # (This is slightly leaky but standard "Data Publishing" scenario assumes you publish the whole anon table).
        
        # If the user wants rigorous ML pipeline:
        # We should use the Train partitions to determine boundaries.
        
        # Let's stick to: passing the dataframe indices
        for partition in self.partitions:
            # Calculate representative
            for col in X.columns:
                if col in self.categorical_features:
                    # Generalization: Set of values? Or Mode?
                    # For ML, "Mode" (Most Frequent) is best to keep it usable as a feature.
                    mode_val = X.loc[partition, col].mode()[0]
                    X_out.loc[partition, col] = mode_val
                else:
                    # Generalization: Range Mean?
                    mean_val = X.loc[partition, col].mean()
                    X_out.loc[partition, col] = mean_val
                    
        return X_out

if __name__ == "__main__":
    # Test
    df = pd.DataFrame({
        'age': [20, 21, 22, 60, 61, 62],
        'income': [10, 11, 12, 50, 51, 52],
        'sex': ['M','M','M','F','F','F']
    })
    
    mondrian = MondrianAnonymizer(k=2)
    mondrian.fit(df, categorical_features=['sex'])
    df_anon = mondrian.transform(df)
    print(df_anon)
