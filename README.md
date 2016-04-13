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

Let's train a model to distinguish between `.h` and `.cc` files:

    $ find base/ -name \*.cc |
      xargs text-classifier --strategy=plain learn training-data 0
    $ find base/ -name \*.h |
      xargs text-classifier --strategy=plain learn training-data 1
    $ text-classifier --cost-function=f1 --weight=bns --no-normalize \
      analyze training-data model
    7531 of 35030 features pass threshold.
    C_pos=1.12e-05 C_neg=1.93-05 mean_F1=0.89 min_F1=0.80 max_F1=1.0
    C_pos=2.24e-05 C_neg=3.87-05 mean_F1=0.98 min_F1=0.89 max_F1=1.0
    C_pos=4.48e-05 C_neg=7.73-05 mean_F1=1.00 min_F1=1.00 max_F1=1.0

Now let's use the model on a different set of files:

    $ find tools/text-classifier -name \*.cc -or -name \*.h |
      xargs text-classifier --no-normalize classify model |
      sort -k2 -g | column -t
    tools/text-classifier/svm.cc                  -0.131145343
    tools/text-classifier/reuters_test.cc         -0.0867494419
    tools/text-classifier/text-classifier.cc      -0.0808973536
    tools/text-classifier/html-tokenizer.cc       -0.0731460601
    tools/text-classifier/23andme.cc              -0.0576973334
    tools/text-classifier/model.cc                -0.025508523
    tools/text-classifier/model.h                 0.254370809
    tools/text-classifier/svm.h                   0.273033738
    tools/text-classifier/html-tokenizer.h        0.280142158
    tools/text-classifier/common.h                0.283355683
    tools/text-classifier/utf8.h                  0.288347989
    tools/text-classifier/23andme.h               0.30374068

# External Dependencies

  * `libkj` from [Cap'n Proto](https://github.com/sandstorm-io/capnproto),
  * [libsnappy](https://google.github.io/snappy/) (optional), and
  * [libsparsehash](https://github.com/sparsehash/sparsehash).

# Reading Material

* [BNS Feature Scaling: An Improved Representation over TF-IDF for SVM Text Classification](http://www.hpl.hp.com/techreports/2007/HPL-2007-32R1.pdf)

* [A Dual Coordinate Descent Method for Large-scale Linear SVM](https://www.csie.ntu.edu.tw/~cjlin/papers/cddual.pdf)
