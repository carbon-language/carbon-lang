// RUN: clang-cc -fsyntax-only -verify %s

int *use_new(int N) {
  return new int [N];
}

int std = 17;
