// RUN: clang %s -E 2>&1 | grep error

#pragma GCC poison rindex
rindex(some_string, 'h');
