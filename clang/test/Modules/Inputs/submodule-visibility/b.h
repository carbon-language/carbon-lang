int m = n;

#include "other.h"
#include "c.h"

#if defined(A) && !defined(ALLOW_NAME_LEAKAGE)
#error A is defined
#endif

#define B
