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

Let's teach a model to distinguish between `.h` and `.c` files:

    $ find base/ -name \*.cc |
      xargs text-classifier --strategy=plain learn training-data 0
    $ find base/ -name \*.h |
      xargs text-classifier --strategy=plain learn training-data 1
    $ text-classifier --weight=bns analyze training-data model

Now let's use the model:

    $ find tools/text-classifier -name \*.cc -or -name \*.h |
      xargs text-classifier classify model |
      sort -k2 -g
    tools/text-classifier/html-tokenizer.cc  0.0106704822
    tools/text-classifier/23andme.cc         0.0110902358
    tools/text-classifier/text-classifier.cc 0.0284913629
    tools/text-classifier/html-tokenizer.h   0.0651432425
    tools/text-classifier/23andme.h          0.0669780597

