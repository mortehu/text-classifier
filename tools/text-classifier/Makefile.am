bin_PROGRAMS += \
  tools/text-classifier/text-classifier
check_PROGRAMS += \
  tools/text-classifier/reuters_test

tools_text_classifier_text_classifier_SOURCES = \
  tools/text-classifier/23andme.cc \
  tools/text-classifier/23andme.h \
  tools/text-classifier/common.h \
  tools/text-classifier/html-tokenizer.cc \
  tools/text-classifier/html-tokenizer.h \
  tools/text-classifier/model.cc \
  tools/text-classifier/model.h \
  tools/text-classifier/svm.cc \
  tools/text-classifier/svm.h \
  tools/text-classifier/text-classifier.cc \
  tools/text-classifier/utf8.h
tools_text_classifier_text_classifier_LDADD = \
  index/libtagsoup.la \
  base/libbase.la

tools_text_classifier_reuters_test_SOURCES = \
  tools/text-classifier/reuters_test.cc
tools_text_classifier_reuters_test_LDADD = \
  index/libtagsoup.la \
  base/libbase.la \
  third_party/gtest/libgtest.a

EXTRA_DIST += \
  tools/text-classifier/reuters/README.txt \
  tools/text-classifier/reuters/reut2-000.sgm \
  tools/text-classifier/reuters/reut2-001.sgm \
  tools/text-classifier/reuters/reut2-002.sgm \
  tools/text-classifier/reuters/reut2-003.sgm \
  tools/text-classifier/reuters/reut2-004.sgm \
  tools/text-classifier/reuters/reut2-005.sgm \
  tools/text-classifier/reuters/reut2-006.sgm \
  tools/text-classifier/reuters/reut2-007.sgm \
  tools/text-classifier/reuters/reut2-008.sgm \
  tools/text-classifier/reuters/reut2-009.sgm \
  tools/text-classifier/reuters/reut2-010.sgm \
  tools/text-classifier/reuters/reut2-011.sgm \
  tools/text-classifier/reuters/reut2-012.sgm \
  tools/text-classifier/reuters/reut2-013.sgm \
  tools/text-classifier/reuters/reut2-014.sgm \
  tools/text-classifier/reuters/reut2-015.sgm \
  tools/text-classifier/reuters/reut2-016.sgm \
  tools/text-classifier/reuters/reut2-017.sgm \
  tools/text-classifier/reuters/reut2-018.sgm \
  tools/text-classifier/reuters/reut2-019.sgm \
  tools/text-classifier/reuters/reut2-020.sgm \
  tools/text-classifier/reuters/reut2-021.sgm
