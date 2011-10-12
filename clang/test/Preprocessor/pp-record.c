// RUN: %clang_cc1 -fsyntax-only -detailed-preprocessing-record %s

// http://llvm.org/PR11120

#define FILE_HEADER_NAME "pp-record.h"

#if defined(FILE_HEADER_NAME)
#include FILE_HEADER_NAME
#endif
