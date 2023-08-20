module.exports = {
  apps: [
    {
      name: 'eus-api',
      script: '~/anaconda3/envs/torch/bin/python -m uvicorn predict:app --host 0.0.0.0 --port 8002 --reload',
      watch: '.',
      env: {
        'PORT': 8002,
        'ENVIRONMENT': 'production',
      },
      env_production: {
        'ENVIRONMENT': 'production',
      },
    },
  ],
};

