
#ifndef TEST
#define TEST
#define INSIDE_RECURSION
#include "2009-06-16-test.c"
#undef INSIDE_RECURSION
#endif

#ifndef INSIDE_RECURSION
int i;
#endif

#ifdef INSIDE_RECURSION
int j;
#endif
