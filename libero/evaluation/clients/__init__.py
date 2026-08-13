"""Built-in clients and their registration side effects."""

from .registry import available_clients, create_client, register_client
from .openpi_client import OpenPIClient

__all__ = ["OpenPIClient", "available_clients", "create_client", "register_client"]
