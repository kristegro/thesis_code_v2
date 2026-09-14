First:

> pip freeze > requirements.in

Then:

> pip-compile --generate-hashes requirements.in

Which generates ``requirements.txt``.

``pip-compile`` can be installed by

> pip install pip-tools