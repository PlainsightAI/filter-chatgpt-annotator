# syntax=docker/dockerfile:1.4
# openfilter-base = python:3.14-slim + all outstanding Debian security patches
# (rebuilt weekly): provides the PYTHONDONTWRITEBYTECODE/PYTHONUNBUFFERED env, the
# appuser account, and /app (WORKDIR) + /app/logs — so none of that is repeated here.
FROM plainsightai/openfilter-base:py3.14

# system libs needed by OpenCV/matplotlib/etc.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxrender1 \
      libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Install pip + filter-chatgpt-annotator (PyPI name) at version from VERSION file.
# The PyPI distribution name kept as filter-chatgpt-annotator;
# the IMPORT path is filter_chattag.
RUN --mount=type=bind,source=VERSION,target=/tmp/VERSION,ro \
    set -eux; \
    RAW="$(head -n1 /tmp/VERSION)"; \
    # strip optional leading v/V and whitespace
    PKG_VERSION="$(printf '%s' "$RAW" | tr -d ' \t\r\n' | sed 's/^[vV]//')"; \
    [ -n "$PKG_VERSION" ] || { echo "VERSION file is empty"; exit 1; }; \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      --index-url https://pypi.org/simple \
      --extra-index-url https://python.openfilter.io/simple \
      "filter-chatgpt-annotator==${PKG_VERSION}"

USER appuser
CMD ["python", "-m", "filter_chattag.filter"]
