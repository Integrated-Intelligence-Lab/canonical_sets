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
.. _documentation: https://canonical-sets.readthedocs.io/en/latest/
.. _LUCID: https://responsibledecisionmaking.github.io/assets/pdf/papers/21.pdf
.. _LUCID-GAN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4289597

.. _Twitter: https://twitter.com/DataLabBE
.. _website: https://data.research.vub.be/
.. _papers: https://researchportal.vub.be/en/organisations/data-analytics-laboratory/publications/

.. _ctgan: https://github.com/sdv-dev/CTGAN
.. _ctganbugs: https://github.com/sdv-dev/CTGAN/pulls/AndresAlgaba


Canonical sets 
==============

|pic2| |nbsp| |pic5| |nbsp| |pic1| |nbsp| |pic8|

|pic6| |nbsp| |pic7| |nbsp| |pic3| |nbsp| |pic4|

AI systems can create, propagate, support, and automate bias in decision-making processes. To mitigate biased decisions,
we both need to understand the origin of the bias and define what it means for an algorithm to make fair decisions.
By Locating Unfairness through Canonical Inverse Design (LUCID), we generate a canonical set that shows the desired inputs
for a model given a preferred output. The canonical set reveals the model's internal logic and exposes potential unethical
biases by repeatedly interrogating the decision-making process.

LUCID-GAN extends on LUCID by generating canonical inputs via a conditional generative model instead of
gradient-based inverse design. LUCID-GAN generates canonical inputs conditional on the predictions of the model under
fairness evaluation. LUCID-GAN has several benefits, including that it applies to non-differentiable models, ensures
that a canonical set consists of realistic inputs, and allows us to assess indirect discrimination and explicitly
check for intersectional unfairness.

Read our paper on `LUCID`_ and `LUCID-GAN`_ for more details, or check out the `documentation`_.

We encourage everyone to `contribute`_ to this project by submitting an issue or a pull request!


Installation
------------

Install ``canonical_sets`` from PyPi.

.. code-block:: bash

    pip install canonical_sets

For development install, see `contribute`_. You can also check the `documentation`_.


Usage
-----


LUCID
~~~~~

``LUCID`` can be used for the gradient-based inverse design to generate canonical sets, and is available for both
``PyTorch`` and ``Tensorflow`` models. It only requires a model, a preferred output, and an example input
(which is often a part of the training data). The results are stored in a ``pd.DataFrame``, and can be accessed by
calling ``results``. It's fully customizable, but can also be used out-of-the-box for a wide range of
applications by using its default settings:

.. code-block:: python

    from canonical_sets import LUCID

    lucid = LUCID(model, outputs, example_data)
    lucid.results.head()

LUCID-GAN
~~~~~~~~~

``LUCIDGAN`` generates canonical sets by using conditional generative models (GANs). This approach has several benefits,
including that it applies to non--differentiable models, ensures that a canonical set consists of realistic inputs,
and allows us to assess indirect discrimination and explicitly check for intersectional unfairness. LUCID-GAN only
requires the input and predictions of a black-box model. It's fully customizable, but can also be used out-of-the-box
for a wide range of applications by using its default settings:

.. code-block:: python

    from canonical_sets import LUCIDGAN

    lucidgan = LUCIDGAN()
    lucidgan.fit(data, predictions)
    samples = lucidgan.sample(100)
    samples.head()

For detailed examples see `examples`_ and for the source code see `canonical_sets`_. For ``LUCID``, we advice to start with either the
``tensorflow`` or ``pytorch`` example, and then the advanced example. For ``LUCIDGAN``, you can replicate the experiments from the paper
with the ``GAN_adult`` and ``GAN_compas`` examples. Note that the results might slightly differ due to the randomness in generating the
samples. You can also check the `documentation`_ for more details. If you have any remaining questions, feel free to submit an issue or PR!


Output-based group metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Most group fairness notions focus on the equality of outcome by computing statistical parity metrics on a model's output.
The two most prominent examples of these statistical output-based metrics are Demographic Parity (DP) and Equality Of Opportunity (EOP).
In DP, we compare the Positivity Rate (PR) of the subpopulations under fairness evaluation, and in EOP, we compare the True Positive Rate (TPR).
The choice between DP and EOP depends on the underlying assumptions and worldview of the evaluator.
The ``Metrics`` class allows you to compute these metrics for binary classification tasks given the predictions and ground truth:

.. code-block:: python

    from canonical_sets.group import Metrics

    metrics = Metrics(preds, targets)
    metrics.metrics


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
You can also check the `documentation`_.


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

Currently some dependencies of the package do not support the Apple M1 and M2 chips.
We will offer support asap.


Acknowledgements
----------------

This project benefited from financial support from Innoviris.

``LUCIDGAN`` is based on the ``CTGAN`` class from the `ctgan`_ package. It has been extended to fix
several bugs (see my PRs on the `ctganbugs`_ GitHub page) and to allow for the extension of the conditional
vector. Note that a part of the code and comments is identical to the original ``CTGAN`` class.


Citation
--------

.. code-block:: none

    @inproceedings{mazijn_lucid_2023,
      title={{LUCID: Exposing Algorithmic Bias through Inverse Design}},
      author={Mazijn, Carmen and Prunkl, Carina and Algaba, Andres and Danckaert, Jan and Ginis, Vincent},
      booktitle={Thirty-Seventh AAAI Conference on Artificial Intelligence (accepted)},
      year={2023},
    }

    @article{algaba_lucidgan_2022,
      title={{LUCID-GAN: Conditional Generative Models to Locate Unfairness}},
      author={Algaba, Andres and Mazijn, Carmen and Prunkl, Carina and Danckaert, Jan and Ginis, Vincent},
      year={2022},
      journal={Working paper}
    }

