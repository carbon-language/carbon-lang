// RUN: clang-cc -DA -DB -E %s | grep 'int a = 37 == 37' &&
// RUN: clang-cc -DA -E %s | grep 'int a = 37 == 37' &&
// RUN: clang-cc -DB -E %s | grep 'int a = 37 == 37' &&
// RUN: clang-cc -E %s | grep 'int a = 927 == 927'
#if defined(A) or defined(B)
#define X 37
#else
#define X 927
#endif

#if defined(A) || defined(B)
#define Y 37
#else
#define Y 927
#endif

int a = X == Y;
