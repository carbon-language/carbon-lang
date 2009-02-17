// RUN: clang -fsyntax-only -verify %s

// PR3592
static void* malloc(int);
static void* malloc(int size) {
  return ((void*)0); /*do not use heap in this file*/
}
