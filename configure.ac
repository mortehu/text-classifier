AC_INIT([text-classifier],[1.0])
AM_INIT_AUTOMAKE([foreign nostdinc subdir-objects])

AC_CONFIG_MACRO_DIR([m4])

AM_MAINTAINER_MODE([disable])

m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

AC_CANONICAL_HOST

AC_GNU_SOURCE

PKG_CHECK_MODULES([CAPNP], [capnp])

AM_PROG_AR
AM_PROG_CC_C_O
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_MAKE_SET

LT_INIT

AC_LANG_PUSH([C++])
AC_CHECK_HEADER([snappy.h], [], [
    AC_MSG_ERROR([required header file not found])
])
AC_CHECK_LIB([snappy], [snappy_compress], [], [
    AC_MSG_ERROR([required library not found])
])

AX_OPENMP()
AX_CXX_COMPILE_STDCXX_14([ext])
AC_LANG_POP([C++])

AC_LANG_PUSH([C])
AX_OPENMP()
AC_LANG_POP([C])

CFLAGS="$CFLAGS -fuse-ld=gold $OPENMP_CFLAGS"
CXXFLAGS="$CXXFLAGS -fuse-ld=gold $OPENMP_CXXFLAGS"

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
