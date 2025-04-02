
CKPT_PATH=""
CONFIG_PATH=""
HOST="127.0.0.1"
PORT="5000"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --ckpt-path)
        CKPT_PATH="$2"
        shift
        shift
        ;;
        --config)
        CONFIG_PATH="$2"
        shift
        shift
        ;;
        --host)
        HOST="$2"
        shift
        shift
        ;;
        --port)
        PORT="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

if [ -z "$CKPT_PATH" ] || [ -z "$CONFIG_PATH" ]; then
    echo "Usage: $0 --ckpt-path PATH --config PATH [--host HOST] [--port PORT]"
    exit 1
fi

pip install -r requirements.txt

python app.py --ckpt-path "$CKPT_PATH" --config "$CONFIG_PATH" --host "$HOST" --port "$PORT"
