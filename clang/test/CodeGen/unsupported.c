// RUN: clang -verify -emit-llvm -o %t %s 

int f0(int x) {
  int vla[x]; // expected-error {{cannot codegen this variable-length array yet}}
  return vla[x-1];
}
