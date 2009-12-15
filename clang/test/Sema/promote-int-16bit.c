// RUN: %clang_cc1 -fsyntax-only -verify %s -triple pic16-unknown-unknown

// Check that unsigned short promotes to unsigned int on targets where
// sizeof(unsigned short) == sizeof(unsigned int)
__typeof(1+(unsigned short)1) x;
unsigned x;
