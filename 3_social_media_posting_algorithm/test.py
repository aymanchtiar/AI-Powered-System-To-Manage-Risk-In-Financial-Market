import joblib
model_path = "/Users/mehdiamrani/Desktop/FYP_project/3_social_media_posting_algorithm/best_module/best_model_AAPL_60m_Lag10.joblib"
model = joblib.load(model_path)
# Save the re-serialized model to a new location
new_model_path = "/Users/mehdiamrani/Desktop/FYP_project/3_social_media_posting_algorithm/best_module/re_serialized_model.joblib"
joblib.dump(model, new_model_path)
new_model = joblib.load(new_model_path)  # Should load without errors
