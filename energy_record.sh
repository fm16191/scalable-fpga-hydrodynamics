#!/bin/bash

start() {
  [[ -z "$1" ]] && { echo "Error: No output file specified!"; exit 1; }
  OUTPUT_FILE="$1"
  bw_card_monitor -p 1 -c0 -s human -m 'Total Input Power|Total FPGA Power' --output_fmt '{time:.2f} {name:30s} {presentReading: 9.3f} {units: <9s}' > "$OUTPUT_FILE" &
  echo "Monitoring started. Output is being written to '$OUTPUT_FILE'."
}

stop() {
  PIDS=$(pgrep -f 'bw_card_monitor')

  [[ -z "$PIDS" ]] && { echo "Error: No monitoring process is running."; exit 1; }

  echo "Stopping the following PIDs: $PIDS"

  # kill each PID
  for pid in $PIDS; do
    kill "$pid" && echo "Killed PID $pid" || echo "Failed to kill PID $pid"
  done

  echo "Monitoring stopped."
}


case "$1" in
  start)
    start "$2"
    ;;
  stop)
    stop
    ;;
  *)
    echo "Usage: $0 {start <output_file>|stop}"
    exit 1
    ;;
esac
