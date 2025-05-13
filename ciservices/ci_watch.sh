#!/bin/bash
echo "Watching for code changes to run CI..."

# Find relevant files and watch them
find . -type f \( -name "*.py" -o -name "Makefile" \) | \
entr -r make check
