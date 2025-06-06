#!/bin/bash
# Robust dependency install with fallbacks
set -e
LOG=/tmp/install.log
{
  pip install --upgrade pip
  pip install \
      numpy pandas scipy yfinance \
      matplotlib plotly seaborn \
      "pyportfolioopt>=2.4.4,<3.0.0" \
      streamlit "streamlit-extras>=0.4.0" \
      pandas_datareader async-lru
  if ! pip install "cvxpy>=1.4,<2"; then
      pip install cvxopt && echo "⚠︎ cvxpy failed, using cvxopt shim"
      echo "HAS_CVXPY=0" >> ~/.feature-flags
  else
      echo "HAS_CVXPY=1" >> ~/.feature-flags
  fi
  pip install pillow && python - <<'PY'
import matplotlib
matplotlib.use("agg")
PY
} >> $LOG 2>&1 || true
exit 0
