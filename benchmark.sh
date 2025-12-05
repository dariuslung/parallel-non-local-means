#!/bin/bash

# Default Configurations
# Modes: 0=CPU Serial, 1=CPU Parallel, 2=Global, 3=Global+Intrin, 4=Shared, 5=Shared+Intrin
MODES=(2 3 4 5) 
# Image IDs corresponding to your 4 sizes (e.g., 0=64, 1=128, 2=256, 3=512)
IMAGES=(0 1 2 3) 
# Number of times to repeat each test for averaging
ITERATIONS=5
# Output file
CSV_FILE="./data/benchmark/benchmark_results.csv"
EXECUTABLE="./build/main"
# Timer Selection Mode
# "1"   = First [Timer] occurrence
# "2"   = Second [Timer] occurrence
TIMER_SELECT=1

# Help Function
function show_help {
    echo "Usage: ./benchmark.sh [options]"
    echo "Options:"
    echo "  -m  List of modes to run (quote them, e.g., \"0 1 2\")"
    echo "  -n  List of image IDs to run (quote them, e.g., \"0 1\")"
    echo "  -i  Number of iterations per test (default: 5)"
    echo "  -t  Timer selection: 1 (NLM), or 2 (Total) (default: 1)"
    echo "  -h  Show this help"
    exit 1
}

# Parse Command Line Arguments
while getopts "m:n:i:t:h" opt; do
    case "$opt" in
    m)  read -a MODES <<< "$OPTARG" ;;
    n)  read -a IMAGES <<< "$OPTARG" ;;
    i)  ITERATIONS=$OPTARG ;;
    t)  TIMER_SELECT=$OPTARG ;;
    h)  show_help ;;
    *)  show_help ;;
    esac
done

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE not found. Please run 'make' first."
    exit 1
fi

# Initialize CSV File with Headers
echo "Mode,ImageID,Iteration,Time_ms" > "$CSV_FILE"
echo "Starting Benchmark..."
echo "------------------------------------------------"
echo "Results will be saved to: $CSV_FILE"
echo "Modes: ${MODES[*]}"
echo "Images: ${IMAGES[*]}"
echo "Iterations: $ITERATIONS"
echo "Timer Selection: $TIMER_SELECT"
echo "------------------------------------------------"

# Loop through settings
for img in "${IMAGES[@]}"; do
    for mode in "${MODES[@]}"; do
        
        # Friendly Output to Console
        mode_name="Unknown"
        case $mode in
            0) mode_name="CPU_Serial" ;;
            1) mode_name="CPU_Parallel" ;;
            2) mode_name="GPU_Global" ;;
            3) mode_name="GPU_Global_Intrinsics" ;;
            4) mode_name="GPU_Shared" ;;
            5) mode_name="GPU_Shared_Intrinsics" ;;
        esac

        echo "Running: Image $img | Mode $mode ($mode_name)"

        # Warm-up run (Only for GPU modes to initialize CUDA context)
        if [ "$mode" -ge 2 ]; then
             $EXECUTABLE -m "$mode" -n "$img" > /dev/null 2>&1
        fi

        # Actual Benchmark Loop
        for ((i=1; i<=ITERATIONS; i++)); do
            
            # Run the program and capture output
            # Use grep to find the Timer line and awk to extract the number
            output=$($EXECUTABLE -m "$mode" -n "$img")
            
            # Extract time using the format: "[Timer] ... : 123.45 ms"
            # Logic: Split by ': ', take the 2nd part, then take the 1st word
            timer_lines=$(echo "$output" | grep "\[Timer\]")
            if [ "$TIMER_SELECT" == 1 ]; then
                # Pick the 1st line
                selected_line=$(echo "$timer_lines" | sed -n '1p')
            elif [ "$TIMER_SELECT" == 2 ]; then
                # Pick the 2nd line
                selected_line=$(echo "$timer_lines" | sed -n '2p')
            else
                echo "Error: Invalid -t option. Use 1, or 2."
                exit 1
            fi
            
            time_ms=$(echo "$selected_line" | awk -F': ' '{print $2}' | awk '{print $1}')

            if [ -z "$time_ms" ]; then
                echo "  Error: Could not extract time. Run failed."
                time_ms="NaN"
            else
                echo "  -> Run $i: $time_ms ms"
                # Append to CSV
                echo "$mode,$img,$i,$time_ms" >> "$CSV_FILE"
            fi
        done
        echo "------------------------------------------------"
    done
done

echo "Benchmark Complete!"