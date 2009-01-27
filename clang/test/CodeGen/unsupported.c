// RUN: clang -verify -emit-llvm -o - %s 

int f0(int x) {
  int vla[x];
  return vla[x-1]; // expected-error {{cannot codegen this return inside scope with VLA yet}}
}
