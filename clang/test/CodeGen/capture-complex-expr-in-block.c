// RUN: %clang_cc1 %s -emit-llvm -o - -fblocks -triple x86_64-apple-darwin10 | FileCheck %s
// rdar://10033986

typedef void (^BLOCK)(void);
int main ()
{
    _Complex double c;
    BLOCK b =  ^() {
      _Complex double z;
      z = z + c;
    };
    b();
}

// CHECK: define internal void @__main_block_invoke_0
// CHECK:  [[C1:%.*]] = alloca { double, double }, align 8
// CHECK:  [[C1]].realp = getelementptr inbounds { double, double }* [[C1]], i32 0, i32 0
// CHECK-NEXT:  [[C1]].real = load double* [[C1]].realp
// CHECK-NEXT:  [[C1]].imagp = getelementptr inbounds { double, double }* [[C1]], i32 0, i32 1
// CHECK-NEXT:  [[C1]].imag = load double* [[C1]].imagp
