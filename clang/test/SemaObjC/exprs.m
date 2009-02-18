// RUN: clang %s -fsyntax-only

// rdar://6597252
Class foo(Class X) {
  return 1 ? X : X;
}

