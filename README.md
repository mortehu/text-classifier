Text Classifier
===============

Given a set of positive and negative training example documents, this program
builds a model that can then be used to predict the class of other documents.

# License

Copyright © e.ventures and Morten Hustveit, 2015

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

# Example Usage

Let's teach a model to distinguish between `.h` and `.cc` files:

    $ find base/ -name \*.cc |
      xargs text-classifier --strategy=plain learn training-data 0
    $ find base/ -name \*.h |
      xargs text-classifier --strategy=plain learn training-data 1
    $ text-classifier --cost-function=f1 --weight=bns --no-normalize \
      analyze training-data model

Now let's use the model:

    $ find tools/text-classifier -name \*.cc -or -name \*.h |
      xargs text-classifier --no-normalize classify model |
      sort -k2 -g | column -t
    tools/text-classifier/23andme.cc          -0.0794910192
    tools/text-classifier/html-tokenizer.cc   -0.0824450776
    tools/text-classifier/text-classifier.cc  -0.0853220075
    tools/text-classifier/23andme.h           0.308668882
    tools/text-classifier/html-tokenizer.h    0.272325069
    tools/text-classifier/utf8.h              0.298772722

# External Dependencies

  * `libkj` from [Cap'n Proto](https://github.com/sandstorm-io/capnproto),
  * [libsnappy](https://google.github.io/snappy/) (optional), and
  * [libsparsehash](https://github.com/sparsehash/sparsehash).

# Reading Material

* [BNS Feature Scaling: An Improved Representation over TF-IDF for SVM Text Classification](http://www.hpl.hp.com/techreports/2007/HPL-2007-32R1.pdf)

* [A Dual Coordinate Descent Method for Large-scale Linear SVM](https://www.csie.ntu.edu.tw/~cjlin/papers/cddual.pdf)
