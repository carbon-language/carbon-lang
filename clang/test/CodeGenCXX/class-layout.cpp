// RUN: clang-cc %s -emit-llvm -o %t &&

// An extra byte shoudl be allocated for an empty class.
// RUN: grep '%.truct.A = type { i8 }' %t
struct A { } a;
