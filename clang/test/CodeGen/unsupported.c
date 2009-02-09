// RUN: clang -verify -emit-llvm -o - %s 

void *x = L"foo"; // expected-error {{cannot compile this wide string yet}}
