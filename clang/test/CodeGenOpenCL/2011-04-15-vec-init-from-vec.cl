// RUN: %clang_cc1 %s -emit-llvm -o %t

typedef __attribute__((ext_vector_type(8)))  unsigned char uchar8;
typedef __attribute__((ext_vector_type(4)))  unsigned long ulong4;
typedef __attribute__((ext_vector_type(16))) unsigned char uchar16;

// OpenCL allows vectors to be initialized by vectors Handle bug in
// VisitInitListExpr for this case below.
void foo( ulong4 v )
{
  uchar8 val[4] = {{(uchar8){((uchar16)(v.lo)).lo}}};
}