// RUN: clang -g -emit-llvm -o %t %s
// PR2784

struct OPAQUE;
typedef struct OPAQUE *PTR;
PTR p;
