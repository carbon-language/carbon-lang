// RUN: touch %t.c
// RUN: not %clang -E %t.c -o %t.c 2> %t.stderr
// RUN: grep "modified" %t.stderr
