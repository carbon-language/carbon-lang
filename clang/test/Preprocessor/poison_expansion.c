// RUN: clang %s -E 2>&1 | not grep error

#define strrchr rindex
#pragma GCC poison rindex

// Can poison multiple times.
#pragma GCC poison rindex

strrchr(some_string, 'h');
