"""Minimal websocket client for the Motus deploy server."""

import functools
import logging
import time
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client


def pack_array(obj: Any) -> Any:
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ('V', 'O', 'c'):
        raise ValueError(f'Unsupported dtype: {obj.dtype}')

    if isinstance(obj, np.ndarray):
        return {
            b'__ndarray__': True,
            b'data': obj.tobytes(),
            b'dtype': obj.dtype.str,
            b'shape': obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b'__npgeneric__': True,
            b'data': obj.item(),
            b'dtype': obj.dtype.str,
        }

    return obj


def unpack_array(obj: dict[bytes, Any]) -> Any:
    if b'__ndarray__' in obj:
        return np.ndarray(buffer=obj[b'data'], dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])

    if b'__npgeneric__' in obj:
        return np.dtype(obj[b'dtype']).type(obj[b'data'])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


class WebsocketClientPolicy:
    """Implements the same client interface as openpi's websocket policy client."""

    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int | None = None,
        api_key: str | None = None,
    ) -> None:
        if host.startswith('ws'):
            self._uri = host
        else:
            self._uri = f'ws://{host}'
        if port is not None:
            self._uri += f':{port}'

        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict[str, Any]]:
        logging.info('Waiting for server at %s...', self._uri)
        while True:
            try:
                headers = {'Authorization': f'Api-Key {self._api_key}'} if self._api_key else None
                connection = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata = unpackb(connection.recv())
                return connection, metadata
            except ConnectionRefusedError:
                logging.info('Still waiting for server...')
                time.sleep(5)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f'Error in inference server:\n{response}')
        return unpackb(response)

    def reset(self) -> None:
        pass
