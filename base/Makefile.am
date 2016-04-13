noinst_LTLIBRARIES += \
  base/libbase.la

base_libbase_la_CPPFLAGS = \
  $(AM_CPPFLAGS) \
  $(LIBLZ4_CFLAGS) \
  $(LIBLZMA_CFLAGS) \
  $(ZLIB_CFLAGS)
base_libbase_la_SOURCES = \
  base/columnfile-reader.cc \
  base/columnfile-writer.cc \
  base/compression.cc \
  base/error.cc \
  base/file.cc \
  base/hash.cc \
  base/random.cc \
  base/string.cc
base_libbase_la_LIBADD = \
  $(CAPNP_LIBS) \
  $(LIBLZ4_LIBS) \
  $(LIBLZMA_LIBS) \
  $(ZLIB_LIBS)

if HAVE_LIBSNAPPY
base_libbase_la_LIBADD += -lsnappy
endif
