// Preamble detection test: see below for comments and test commands.

#include <blah>
#ifndef FOO
#else
#ifdef BAR
#elif WIBBLE
#endif
#pragma unknown
#endif

#ifdef WIBBLE
#include "honk"
#else
int foo();
#endif

// This test checks for detection of the preamble of a file, which
// includes all of the starting comments and #includes. Note that any
// changes to the preamble part of this file must be mirrored in
// Inputs/preamble.txt, since we diff against it.

// RUN: %clang_cc1 -print-preamble %s > %t
// RUN: diff %t %S/Inputs/preamble.txt

