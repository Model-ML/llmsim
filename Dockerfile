# Build with the same Rust version as CI.
FROM rust:1.99.0-slim-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# The workspace manifests and embedded scenario examples are build inputs.
COPY Cargo.toml README.md rust-toolchain.toml ./
COPY crates ./crates
COPY src ./src
COPY examples/scenarios ./examples/scenarios
RUN cargo build --release --bin llmsim

# Runtime stage
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary
COPY --from=builder /app/target/release/llmsim /usr/local/bin/

# Create non-root user
RUN useradd -m -s /bin/bash llmsim
USER llmsim

# Default configuration
ENV LLMSIM_HOST=0.0.0.0
ENV LLMSIM_PORT=8080

EXPOSE 8080

ENTRYPOINT ["llmsim", "serve"]
