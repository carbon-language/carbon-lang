// RUN: not %clang_cc1 %s -fno-rtti -triple=i686-pc-win32 -emit-llvm -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK32
// RUN: %clang_cc1 %s -fno-rtti -triple=x86_64-pc-win32 -emit-llvm -o - | FileCheck --check-prefix=CHECK64

namespace byval_thunk {
struct Agg {
  Agg();
  Agg(const Agg &);
  ~Agg();
  int x;
};

struct A { virtual void foo(Agg x); };
struct B { virtual void foo(Agg x); };
struct C : A, B { virtual void foo(Agg x); };
C c;

// CHECK32: cannot compile this non-trivial argument copy for thunk yet

// CHECK64-LABEL: define linkonce_odr void @"\01?foo@C@byval_thunk@@W7EAAXUAgg@2@@Z"
// CHECK64:             (%"struct.byval_thunk::C"* %this, %"struct.byval_thunk::Agg"* %x)
// CHECK64:   getelementptr i8* %{{.*}}, i32 -8
// CHECK64:   call void @"\01?foo@C@byval_thunk@@UEAAXUAgg@2@@Z"(%"struct.byval_thunk::C"* %2, %"struct.byval_thunk::Agg"* %x)
// CHECK64-NOT: call
// CHECK64:   ret void
}
