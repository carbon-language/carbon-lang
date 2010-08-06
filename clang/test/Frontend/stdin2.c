// RUN: cd %S && %clang_cc1 -E - < stdin2.c

/*
 * Bug 4897; current working directory should be searched
 *           for #includes when input is stdin.
 */

#ifndef BUG_4897
#define BUG_4897
#include "stdin2.c"
#endif
