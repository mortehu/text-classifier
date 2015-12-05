noinst_LTLIBRARIES += \
  base/libbase.la

base_libbase_la_SOURCES = \
  base/columnfile-reader.cc \
  base/columnfile-writer.cc \
  base/error.cc \
  base/file.cc \
  base/hash.cc \
  base/random.cc \
  base/string.cc
base_libbase_la_LIBADD = $(CAPNP_LIBS)

if HAVE_LIBSNAPPY
base_libbase_la_LIBADD += -lsnappy
endif
