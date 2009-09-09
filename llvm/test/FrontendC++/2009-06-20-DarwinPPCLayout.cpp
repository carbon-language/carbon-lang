// RUN: %llvmgxx -S -m32 -emit-llvm %s -o - | grep baz | grep global | grep {struct.bar}
// RUN: %llvmgxx -S -m32 -emit-llvm %s -o - | grep ccc | grep global | grep {struct.CC}
// RUN: %llvmgxx -S -m32 -emit-llvm %s -o - | grep quux | grep global | grep {struct.bar}
// RUN: %llvmgxx -S -m32 -emit-llvm %s -o - | grep foo | grep global | grep {struct.SRCFilter::FilterEntry}
// RUN: %llvmgxx -S -m32 -emit-llvm %s -o - | grep {struct.bar} | grep {1 x i32}
// RUN: %llvmgxx -S -m32 -emit-llvm %s -o - | grep {struct.CC} | grep {struct.payre<KBFP,float*} | grep {.base.32} | grep {1 x i32}
// RUN: %llvmgxx -S -m32 -emit-llvm %s -o - | grep {struct.SRCFilter::FilterEntry} | not grep {1 x i32}
// XFAIL: *
// XTARGET: powerpc-apple-darwin

template<class _T1, class _T2>     struct payre     {
  _T1 first;
  _T2 second;
  payre()       : first(), second() {    }
};
struct KBFP {
  double mCutoffFrequency;
};
class SRCFilter {
  struct FilterEntry: public payre<KBFP, float*>{};
  static FilterEntry foo;
};
SRCFilter::FilterEntry SRCFilter::foo;    // 12 bytes
payre<KBFP, float*> baz;                  // 16 bytes
class CC {                                // 16 bytes
  public: payre<KBFP, float*> x;          
};
class CC ccc;

struct bar { KBFP x; float* y;};          // 16 bytes
struct bar quux;

