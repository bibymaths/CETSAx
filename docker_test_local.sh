#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024 Abhinav Mishra
# SPDX-License-Identifier: BSD-3-Clause
# Runs pytest inside the development container.
set -euo pipefail

docker compose -f docker-compose.dev.yml run --rm cetsax pytest tests/
