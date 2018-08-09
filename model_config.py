# 模型参数定义
# Random Forest Regression, setting parameters
nonlinear_model_config = {'n_estimators':10,
                          'max_features':'sqrt',
                          'max_depth': None,
                          'random_state':123,
                          'n_jobs': 4}

# Linear Regression, setting parameters
linear_model_config = {'fit_intercept':True,
                       'n_jobs': 2}


