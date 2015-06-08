// RUN: %clang -S -emit-llvm -O1 -mllvm -disable-llvm-optzns -S %s -o - | FileCheck %s

// This test should not to generate llvm.lifetime.start/llvm.lifetime.end for
// f function because all temporary objects in this function are used for the
// final result

class S {
  char *ptr;
  unsigned int len;
};

class T {
  S left;
  S right;

public:
  T(const char s[]);
  T(S);

  T concat(const T &Suffix) const;
  const char * str() const;
};

const char * f(S s)
{
// CHECK: %1 = alloca %class.T, align 4
// CHECK: %2 = alloca %class.T, align 4
// CHECK: %3 = alloca %class.S, align 4
// CHECK: %4 = alloca %class.T, align 4
// CHECK: %5 = call x86_thiscallcc %class.T* @"\01??0T@@QAE@QBD@Z"
// CHECK: %6 = bitcast %class.S* %3 to i8*
// CHECK: %7 = bitcast %class.S* %s to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i32
// CHECK: %8 = call x86_thiscallcc %class.T* @"\01??0T@@QAE@VS@@@Z"
// CHECK: call x86_thiscallcc void @"\01?concat@T@@QBE?AV1@ABV1@@Z"
// CHECK: %9 = call x86_thiscallcc i8* @"\01?str@T@@QBEPBDXZ"(%class.T* %4)

  return T("[").concat(T(s)).str();
}
