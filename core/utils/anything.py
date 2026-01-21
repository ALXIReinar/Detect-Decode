from core.config_dir.config import env, WORKDIR


def create_log_dirs():
    env.LOG_DIR.mkdir(exist_ok=True)

    debug, info_warn, err = env.LOG_DIR / 'debug', env.LOG_DIR / 'info_warning', env.LOG_DIR / 'error'

    debug.mkdir(exist_ok=True, parents=True)
    info_warn.mkdir(exist_ok=True, parents=True)
    err.mkdir(exist_ok=True, parents=True)
