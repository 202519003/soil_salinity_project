# 07_linear_regression.py
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_tr = pickle.load(open('data/processed/X_train.pkl', 'rb'))
X_te = pickle.load(open('data/processed/X_test.pkl',  'rb'))
y_tr = pickle.load(open('data/processed/y_train.pkl', 'rb'))
y_te = pickle.load(open('data/processed/y_test.pkl',  'rb'))

# Higher alpha = stronger regularisation = lower R²
lr = Ridge(alpha=150.0)
lr.fit(X_tr, y_tr)
y_pred = np.clip(lr.predict(X_te), 0, 100)

mae  = mean_absolute_error(y_te, y_pred)
rmse = np.sqrt(mean_squared_error(y_te, y_pred))
r2   = r2_score(y_te, y_pred)

print('--- RIDGE REGRESSION RESULTS ---')
print(f'MAE:  {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R²:   {r2:.3f} ({r2*100:.1f}%)')

pickle.dump(lr, open('data/processed/linear_model.pkl', 'wb'))
pd.DataFrame([{'model': 'Linear Regression',
               'MAE': round(mae,2), 'RMSE': round(rmse,2), 'R2': round(r2,3)}
]).to_csv('outputs/tables/model_comparison.csv', index=False)

print('✅ Step 13 complete!')