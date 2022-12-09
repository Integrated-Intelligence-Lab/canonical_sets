.. highlight:: none

.. _poetry: https://python-poetry.org/docs/
.. _PEP8: https://peps.python.org/pep-0008/
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html#style-guide


Contributing
============

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change. Please make also sure to update
tests as appropriate.

To easily install all the (dev) dependencies use `poetry`_:

.. code-block::

     poetry install


Some useful ``Make`` commands which can be used during development:

.. code-block::

     clean         Remove all pycache, coverage, ... files in project folder.
     format        Run isort and black to autoformat python files.
     check         Run isort, black, flake8 and mypy to perform checks.
     test          Run pytest for unit and integration testing located in tests folder.
     install       Install the package and (dev) dependencies.
     docs          Create docs.
Note that the docs command can only be used in the poetry environment.

Example to clean project folder:

.. code-block::

     make clean


Note that we aim to adhere to the `PEP8`_ and the `numpydoc`_ style guides.
