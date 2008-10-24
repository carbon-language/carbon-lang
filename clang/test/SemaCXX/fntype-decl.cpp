// RUN: clang -fsyntax-only -verify %s

// PR2942
typedef void fn(int);
fn f;
