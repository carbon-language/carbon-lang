// RUN: clang-cc -E %s | grep '#pragma foo bar'

// GCC doesn't expand macro args for unrecognized pragmas.
#define bar xX
#pragma foo bar

