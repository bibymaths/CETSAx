#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024 Abhinav Mishra
# SPDX-License-Identifier: BSD-3-Clause
# Builds the Docker image and pushes it to ghcr.io.
# Requires GITHUB_TOKEN to be set in the environment.
set -euo pipefail

IMAGE="ghcr.io/bibymaths/cetsax"
TAG="${1:-latest}"

echo "Logging in to ghcr.io..."
echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${GITHUB_ACTOR}" --password-stdin

echo "Building ${IMAGE}:${TAG}..."
docker build -t "${IMAGE}:${TAG}" .

echo "Pushing ${IMAGE}:${TAG}..."
docker push "${IMAGE}:${TAG}"

echo "Done."
