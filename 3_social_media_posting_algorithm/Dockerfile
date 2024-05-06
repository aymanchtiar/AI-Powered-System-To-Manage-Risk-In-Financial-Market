
FROM python:3.8-slim

WORKDIR /usr/src/app

COPY . .


COPY best_module/best_model_AAPL_60m_Lag6.joblib /usr/src/app/best_model_AAPL_60m_Lag10.joblib
COPY posts_data.csv /usr/src/app/posts_data.csv

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENV NAME World

CMD ["python", "product.py"]
