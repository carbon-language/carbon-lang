// RUN: clang %s -fsyntax-only -verify

// rdar://6597252
Class foo(Class X) {
  return 1 ? X : X;
}

