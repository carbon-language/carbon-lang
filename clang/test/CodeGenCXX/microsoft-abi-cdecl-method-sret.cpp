// RUN: %clang_cc1 -triple i386-pc-win32 -emit-llvm %s -o - | FileCheck %s

// PR15768

// A trivial 12 byte struct is returned indirectly.
struct S {
  S();
  int a, b, c;
};

struct C {
  S variadic_sret(const char *f, ...);
  S __cdecl cdecl_sret();
  S __cdecl byval_and_sret(S a);
  int c;
};

S C::variadic_sret(const char *f, ...) { return S(); }
S C::cdecl_sret() { return S(); }
S C::byval_and_sret(S a) { return S(); }

// CHECK: define x86_cdeclmethodcc void @"\01?variadic_sret@C@@QAA?AUS@@PBDZZ"(%struct.S* noalias sret %agg.result, %struct.C* %this, i8* %f, ...)
// CHECK: define x86_cdeclmethodcc void @"\01?cdecl_sret@C@@QAA?AUS@@XZ"(%struct.S* noalias sret %agg.result, %struct.C* %this)
// CHECK: define x86_cdeclmethodcc void @"\01?byval_and_sret@C@@QAA?AUS@@U2@@Z"(%struct.S* noalias sret %agg.result, %struct.C* %this, %struct.S* byval align 4 %a)

int main() {
  C c;
  c.variadic_sret("asdf");
  c.cdecl_sret();
  c.byval_and_sret(S());
}
// CHECK-LABEL: define i32 @main()
// CHECK: call x86_cdeclmethodcc void {{.*}} @"\01?variadic_sret@C@@QAA?AUS@@PBDZZ"
// CHECK: call x86_cdeclmethodcc void @"\01?cdecl_sret@C@@QAA?AUS@@XZ"
// CHECK: call x86_cdeclmethodcc void @"\01?byval_and_sret@C@@QAA?AUS@@U2@@Z"
