module.exports = {
  apps: [{
    name: 'my-eus-app',
    script: 'npm',
    args: 'start',
    cwd: '~/gitee/my-eus-app',
    env: {
      NODE_ENV: 'production',
      PORT: 3001
    }
  }]
};

