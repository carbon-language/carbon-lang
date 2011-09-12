// RUN: %clang_cc1 -emit-module -o %t/macros.pcm -DMODULE %s
// RUN: %clang_cc1 -verify -fmodule-cache-path %t %s

#if defined(MODULE)
#define INTEGER(X) int
#define FLOAT float
#define DOUBLE double

#__export_macro__ INTEGER
#__export_macro__ DOUBLE

#else

__import_module__ macros;

#ifndef INTEGER
#  error INTEGER macro should be visible
#endif

#ifdef FLOAT
#  error FLOAT macro should not be visible
#endif

#ifdef MODULE
#  error MODULE macro should not be visible
#endif

double d;
DOUBLE *dp = &d;

#__export_macro__ WIBBLE // expected-error{{no macro named 'WIBBLE' to export}}

#endif
