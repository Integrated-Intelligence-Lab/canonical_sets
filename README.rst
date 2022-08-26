.. |nbsp| unicode:: U+00A0 .. NO-BREAK SPACE

.. |pic1| image:: https://img.shields.io/badge/python-3.8%20%7C%203.9-blue
.. |pic2| image:: https://img.shields.io/github/license/mashape/apistatus.svg
.. |pic3| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. |pic4| image:: https://img.shields.io/badge/%20type_checker-mypy-%231674b1?style=flat
.. |pic5| image:: https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey
.. |pic6| image:: https://github.com/Integrated-Intelligence-Lab/canonical_sets/actions/workflows/testing.yml/badge.svg
.. |pic7| image:: https://img.shields.io/readthedocs/canonical_sets
.. |pic8| image:: https://img.shields.io/pypi/v/canonical_sets

.. _canonical_sets: https://github.com/Integrated-Intelligence-Lab/canonical_sets/tree/main/canonical_sets
.. _examples: https://github.com/Integrated-Intelligence-Lab/canonical_sets/tree/main/examples
.. _contribute: https://github.com/Integrated-Intelligence-Lab/canonical_sets/blob/main/CONTRIBUTING.rst

.. _Twitter: https://twitter.com/DataLabBE
.. _website: https://data.research.vub.be/
.. _papers: https://researchportal.vub.be/en/organisations/data-analytics-laboratory/publications/


Canonical sets 
==============

|pic2| |nbsp| |pic5| |nbsp| |pic1| |nbsp| |pic8|

|pic6| |nbsp| |pic7| |nbsp| |pic3| |nbsp| |pic4|

AI systems can create, propagate, support, and automate bias in decision-making processes. To mitigate biased decisions,
we both need to understand the origin of the bias and define what it means for an algorithm to make fair decisions.
By Locating Unfairness through Canonical Inverse Design (LUCID), we generate a canonical set that shows the desired inputs
for a model given a preferred output. The canonical set reveals the model's internal logic and exposes potential unethical
biases by repeatedly interrogating the decision-making process. By shifting the focus towards equality of treatment and
looking into the algorithm's internal workings, LUCID is a valuable addition to the toolbox of algorithmic fairness evaluation.
Read our paper on LUCID for more details.

We encourage everyone to `contribute`_ to this project by submitting an issue or a pull request!


Installation
------------

Install ``canonical_sets`` from PyPi.

.. code-block:: bash

    pip install canonical_sets

For development install, see `contribute`_.


Usage
-----
``LUCID`` can be used for the gradient-based inverse design to generate canonical sets, and is available for both
``PyTorch`` and ``Tensorflow`` models. It's fully customizable, but can also be used out-of-the-box for a wide range of
models by using its default settings:

.. code-block:: python

    from canonical_sets import LUCID

    lucid = LUCID(model, outputs, example_data)
    lucid.results.head()

It only requires a model, a preferred output, and an example input (which is often a part of the training data).
The results are stored in a ``pd.DataFrame``, and can be accessed by calling ``results``.

For detailed examples see `examples`_ and for the source code see `canonical_sets`_. We advice to start with either the
``tensorflow`` or ``pytorch`` example, and then the advanced example. If you have any remaining questions, feel free to
submit an issue or PR!


Data
----
``canonical_sets`` contains some functionality to easily access commonly used data sets in the fairness literature:

.. code-block:: python

    from canonical_sets import Adult, Compas

    adult = Adult()
    adult.train_data.head()

    compas = Compas()
    compas.train_data.head()

The default settings can be customized to change the pre-processing, splitting, etc. See `examples`_  for details.


Community
---------

If you are interested in cross-disciplinary research related to machine learning, feel free to:

* Follow DataLab on `Twitter`_.
* Check the `website`_.
* Read our `papers`_.


Disclaimer
----------

The package and the code is provided "as-is" and there is NO WARRANTY of any kind. 
Use it only if the content and output files make sense to you.


Acknowledgements
----------------

This project benefited from financial support from Innoviris.


Citation
--------

.. code-block::

    @inproceedings{mazijn_canonicalsets_2022,
      title={{Exposing Algorithmic Bias through Inverse Design}},
      author={Mazijn, Carmen and Prunkl, Carina and Algaba, Andres and Danckaert, Jan and Ginis, Vincent},
      booktitle={Workshop at International Conference on Machine Learning},
      year={2022},
    }
