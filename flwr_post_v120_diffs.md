## v1.21

- Strategies updated to use Message objects instead. See https://flower.ai/docs/framework/how-to-upgrade-to-message-api.html
- ServerApp and ClientApp replaced with SuperExec, see https://flower.ai/docs/framework/explanation-flower-architecture.html.
    - Need to replace docker images with SuperExec version. Might only need one image instead of two.
- Imports from ``flwr.server, flwr.client, flwr.common`` is changed to ``flwer.serverapp, flwr.clientapp, flwr.app`` respectively.

## v1.22

- Nothing important.

## v1.23

- Nothing important.

## v1.24

- Python 3.13 support.

## v1.25

- Nothing important.

## v1.26

- flwr CLI changes. ``federation`` configuration in pyproject.toml files now obsolete. See https://flower.ai/docs/framework/1.26/en/ref-api-cli.html and https://flower.ai/docs/framework/1.26/en/ref-flower-configuration.html.

## v1.26.1

- Nothing important.

## v1.27

- Nothing important.

## v1.28

- Nothing important.

## v1.29

- Nothing important.

## v1.30

- Maybe something with the task system is important?

## v1.31

- Something with inter-task communication?

## v1.32

- Nothing important.

## v1.32.1 

- Nothing important.

## 1.33

- Nothing important.

## 1.34

- Nothing important.

## 1.35

- ``--serverapp-api-address`` on SuperLink and ``--clientappio-api-address`` on SuperNode replaced with seperate ``--host`` and ``-port``.
- Default adresses changed from ``0.0.0.0:9091`` to ``127.0.0.1:8000`` on SuperLink and ``0.0.0.0:9094`` to ``127.0.0.1:9094`` on SuperNode. Set ``--host 0.0.0.0`` when the Runtime API must remain reachable externally.