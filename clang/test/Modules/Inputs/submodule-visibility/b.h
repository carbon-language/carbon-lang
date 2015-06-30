int m = n;

#if defined(A) && !defined(ALLOW_NAME_LEAKAGE)
#error A is defined
#endif

#define B
