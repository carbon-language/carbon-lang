// RUN: not clang -E %s 2>&1 | grep 'invalid token at start of a preprocessor expression'
// RUN: not clang -E %s 2>&1 | grep 'token is not a valid binary operator in a preprocessor subexpression'
// PR2220

#if 1 * * 2
#endif

#if 4 [ 2
#endif

