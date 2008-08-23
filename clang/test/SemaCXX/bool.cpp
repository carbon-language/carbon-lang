// RUN: clang -fsyntax-only -verify %s 

// Bool literals can be enum values.
enum {
  ReadWrite = false,
  ReadOnly = true
};
