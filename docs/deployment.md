# Deployment and model updates

Author: Mahdi Dadgar

The project has two front ends on the same models: a Streamlit app for people and a REST API for other systems. This page describes how I host them and how the models are updated when new flock data arrive.

## 1. The app on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Click **Create app**, choose this repository, branch `main` and main file `app.py`.
3. Under **Advanced settings**, choose Python 3.12 or 3.13.
4. Click **Deploy**. Streamlit installs `requirements.txt` and starts the app.

The app loads the saved models from `models/` and reads the example flocks from `data/raw/`. Every push to `main` redeploys it.

## 2. The API in Docker

### Run it locally

```bash
docker build -t layer-flock-api .
docker run -p 8000:8000 layer-flock-api
```

Open http://localhost:8000/docs for the interactive documentation, or call it from the command line:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/predict/flock -H "Content-Type: application/json" -d @docs/api_examples/plan_request.json
curl -X POST localhost:8000/forecast -H "Content-Type: application/json" -d @docs/api_examples/forecast_request.json
```

The image holds only what the API needs (code, trained models, standards tables), runs as a non-root user and has a health check. It listens on the port in the `PORT` environment variable, or 8000 if that is not set.

### Host it on Render

1. Sign in at [render.com](https://render.com) with GitHub.
2. Click **New** → **Web Service** and connect this repository.
3. Set **Language** to **Docker**. Render builds the `Dockerfile` in the repository root.
4. Choose an instance type. The free type is enough for a demonstration.
5. Under **Advanced**, set the health check path to `/health`.
6. Click **Deploy**. Render sets `PORT` (10000 by default), and the container picks it up.

The API is then at `https://<service-name>.onrender.com/docs`. A free service sleeps after 15 minutes without requests and needs about a minute to wake up, so the first call after a quiet period is slow.

## 3. Updating the models with new data

Each flock that finishes laying adds 50 to 70 weekly records. This is how they would get into the models:

1. **Add the records.** New flocks go into `data/raw/flocks.csv`, weekly records into `data/raw/weekly_records.csv` and egg quality samples into `data/raw/egg_quality.csv`, in the same columns.
2. **Retrain and validate.** `python -m src.train` holds out whole farms, scores every model on them, computes the intervals, refits on all farms and writes the models and `models/model_card.json`.
3. **Compare with the model in use.** Before replacing anything, compare the new model card with the current one: held-out RMSE, forecast error by horizon and interval coverage. A new model should be at least as good on farms it has not seen. If the diet or farm mix has changed a lot, run `python -m src.compare_models` again to check that the chosen model types still hold.
4. **Version.** Raise `MODEL_VERSION` in `src/train.py`. Every API response carries the version, so each prediction can be traced to the model that made it.
5. **Release.** Push to GitHub. The tests run, the Docker image is built and called, and the app and API redeploy. If something goes wrong, the previous models are one `git revert` away.

Two checks matter between retrains. The first is input drift: the API already warns when a diet falls outside the training range, and a growing share of such warnings means the models need new data. The second is forecast error: when finished flocks are compared with the forecasts made for them, a rising error is the signal to retrain.
