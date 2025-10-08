from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Union

import torch
from humanfriendly import parse_size
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils import limits

from agentgrid.constants import PUBLIC_INITIAL_PEERS
from agentgrid.server.server import Server
from agentgrid.utils.convert_block import QuantType

logger = logging.getLogger(__name__)


ConfigSource = Union[str, Path, Mapping[str, Any]]


@dataclass
class ServerLaunchArtifacts:
    server: Server
    config: dict[str, Any]


def load_server_config(source: ConfigSource) -> dict[str, Any]:
    """Load a server configuration from a mapping or JSON file path."""

    if isinstance(source, Mapping):
        return dict(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, Mapping):
        raise ValueError(f"Config at {path} must contain a JSON object")
    return dict(data)


def build_server_from_config(source: ConfigSource) -> ServerLaunchArtifacts:
    """Construct a :class:`Server` instance from a configuration mapping or JSON file."""

    config = load_server_config(source)
    config = _normalize_model_alias(config)

    increase_file_limit = config.pop("increase_file_limit", None)
    if increase_file_limit:
        limits.increase_file_limit(increase_file_limit, increase_file_limit)

    compression = _prepare_compression(config)
    max_disk_space = _prepare_max_disk_space(config)
    host_maddrs, announce_maddrs = _prepare_multiaddrs(config)
    _prepare_swarm_settings(config)
    _prepare_quant_type(config)
    _prepare_startup_timeout(config)

    if not torch.backends.openmp.is_available():
        torch.set_num_threads(1)

    server = Server(
        **config,
        host_maddrs=host_maddrs,
        announce_maddrs=announce_maddrs,
        compression=compression,
        max_disk_space=max_disk_space,
    )
    return ServerLaunchArtifacts(server=server, config=config)


def run_server_from_config(source: ConfigSource) -> None:
    """Launch the server using the provided configuration."""

    artifacts = build_server_from_config(source)
    server = artifacts.server
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, shutting down")
    finally:
        server.shutdown()


def _normalize_model_alias(config: MutableMapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    model_alias = cfg.pop("model", None)
    if model_alias is not None and "converted_model_name_or_path" not in cfg:
        cfg["converted_model_name_or_path"] = model_alias
    return cfg


def _prepare_compression(config: MutableMapping[str, Any]) -> CompressionType:
    compression = config.pop("compression", "NONE")
    try:
        compression_enum = CompressionType.Value
    except AttributeError:
        compression_enum = None

    if compression_enum and isinstance(compression, int):
        return CompressionType(compression)
    if isinstance(compression, str):
        return getattr(CompressionType, compression.upper())
    if isinstance(compression, CompressionType.__class__):
        return CompressionType(compression)
    if isinstance(compression, CompressionType):
        return compression
    raise TypeError("compression must be a string or CompressionType value")


def _prepare_max_disk_space(config: MutableMapping[str, Any]) -> int | None:
    max_disk_space = config.pop("max_disk_space", None)
    if max_disk_space is None:
        return None
    if isinstance(max_disk_space, int):
        return max_disk_space
    if isinstance(max_disk_space, str):
        return int(parse_size(max_disk_space))
    raise TypeError("max_disk_space must be int, string, or null")


def _prepare_multiaddrs(config: MutableMapping[str, Any]) -> tuple[list[str], list[str] | None]:
    host_maddrs = config.pop("host_maddrs", None)
    announce_maddrs = config.pop("announce_maddrs", None)
    port = config.pop("port", None)
    public_ip = config.pop("public_ip", None)

    if port is not None and host_maddrs is not None:
        raise ValueError("Cannot specify both 'port' and 'host_maddrs'")
    if public_ip is not None and announce_maddrs is not None:
        raise ValueError("Cannot specify both 'public_ip' and 'announce_maddrs'")

    if port is None:
        port = 0

    if host_maddrs is None:
        host_maddrs = [f"/ip4/0.0.0.0/tcp/{port}", f"/ip6/::/tcp/{port}"]

    if public_ip is not None:
        if port == 0:
            raise ValueError("When 'public_ip' is provided, a non-zero 'port' must also be set")
        announce_maddrs = [f"/ip4/{public_ip}/tcp/{port}"]

    return list(host_maddrs), list(announce_maddrs) if announce_maddrs is not None else None


def _prepare_swarm_settings(config: MutableMapping[str, Any]) -> None:
    if config.pop("new_swarm", False):
        config["initial_peers"] = []
    if "initial_peers" not in config:
        config["initial_peers"] = PUBLIC_INITIAL_PEERS


def _prepare_quant_type(config: MutableMapping[str, Any]) -> None:
    quant_type = config.pop("quant_type", None)
    if quant_type is None or isinstance(quant_type, QuantType):
        if quant_type is not None:
            config["quant_type"] = quant_type
        return
    if isinstance(quant_type, str):
        config["quant_type"] = QuantType[quant_type.upper()]
        return
    raise TypeError("quant_type must be a string, QuantType, or null")


def _prepare_startup_timeout(config: MutableMapping[str, Any]) -> None:
    value = config.pop("daemon_startup_timeout", None)
    if value is not None:
        config["startup_timeout"] = value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch Agent Grid server from JSON config")
    parser.add_argument("config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without running the server")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_server_config(config_path)

    if args.dry_run:
        build_server_from_config(config)
        print(json.dumps(config, indent=2))
        return

    run_server_from_config(config)


if __name__ == "__main__":
    main()
