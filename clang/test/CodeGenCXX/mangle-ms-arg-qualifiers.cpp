// RUN: %clang_cc1 -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -o - -cxx-abi microsoft -triple=x86_64-pc-win32 | FileCheck -check-prefix=X64 %s

void foo(const unsigned int) {}
// CHECK: "\01?foo@@YAXI@Z"
// X64: "\01?foo@@YAXI@Z"

void foo(const double) {}
// CHECK: "\01?foo@@YAXN@Z"
// X64: "\01?foo@@YAXN@Z"

void bar(const volatile double) {}
// CHECK: "\01?bar@@YAXN@Z"
// X64: "\01?bar@@YAXN@Z"

void foo_pad(char * x) {}
// CHECK: "\01?foo_pad@@YAXPAD@Z"
// X64: "\01?foo_pad@@YAXPEAD@Z"

void foo_pbd(const char * x) {}
// CHECK: "\01?foo_pbd@@YAXPBD@Z"
// X64: "\01?foo_pbd@@YAXPEBD@Z"

void foo_pcd(volatile char * x) {}
// CHECK: "\01?foo_pcd@@YAXPCD@Z"
// X64: "\01?foo_pcd@@YAXPECD@Z"

void foo_qad(char * const x) {}
// CHECK: "\01?foo_qad@@YAXQAD@Z"
// X64: "\01?foo_qad@@YAXQEAD@Z"

void foo_rad(char * volatile x) {}
// CHECK: "\01?foo_rad@@YAXRAD@Z"
// X64: "\01?foo_rad@@YAXREAD@Z"

void foo_sad(char * const volatile x) {}
// CHECK: "\01?foo_sad@@YAXSAD@Z"
// X64: "\01?foo_sad@@YAXSEAD@Z"

void foo_papad(char ** x) {}
// CHECK: "\01?foo_papad@@YAXPAPAD@Z"
// X64: "\01?foo_papad@@YAXPEAPEAD@Z"

void foo_papbd(char const ** x) {}
// CHECK: "\01?foo_papbd@@YAXPAPBD@Z"
// X64: "\01?foo_papbd@@YAXPEAPEBD@Z"

void foo_papcd(char volatile ** x) {}
// CHECK: "\01?foo_papcd@@YAXPAPCD@Z"
// X64: "\01?foo_papcd@@YAXPEAPECD@Z"

void foo_pbqad(char * const* x) {}
// CHECK: "\01?foo_pbqad@@YAXPBQAD@Z"
// X64: "\01?foo_pbqad@@YAXPEBQEAD@Z"

void foo_pcrad(char * volatile* x) {}
// CHECK: "\01?foo_pcrad@@YAXPCRAD@Z"
// X64: "\01?foo_pcrad@@YAXPECREAD@Z"

void foo_qapad(char ** const x) {}
// CHECK: "\01?foo_qapad@@YAXQAPAD@Z"
// X64: "\01?foo_qapad@@YAXQEAPEAD@Z"

void foo_rapad(char ** volatile x) {}
// CHECK: "\01?foo_rapad@@YAXRAPAD@Z"
// X64: "\01?foo_rapad@@YAXREAPEAD@Z"

void foo_pbqbd(const char * const* x) {}
// CHECK: "\01?foo_pbqbd@@YAXPBQBD@Z"
// X64: "\01?foo_pbqbd@@YAXPEBQEBD@Z"

void foo_pbqcd(volatile char * const* x) {}
// CHECK: "\01?foo_pbqcd@@YAXPBQCD@Z"
// X64: "\01?foo_pbqcd@@YAXPEBQECD@Z"

void foo_pcrbd(const char * volatile* x) {}
// CHECK: "\01?foo_pcrbd@@YAXPCRBD@Z"
// X64: "\01?foo_pcrbd@@YAXPECREBD@Z"

void foo_pcrcd(volatile char * volatile* x) {}
// CHECK: "\01?foo_pcrcd@@YAXPCRCD@Z"
// X64: "\01?foo_pcrcd@@YAXPECRECD@Z"

void foo_aad(char &x) {}
// CHECK: "\01?foo_aad@@YAXAAD@Z"
// X64: "\01?foo_aad@@YAXAEAD@Z"

void foo_abd(const char &x) {}
// CHECK: "\01?foo_abd@@YAXABD@Z"
// X64: "\01?foo_abd@@YAXAEBD@Z"

void foo_aapad(char *&x) {}
// CHECK: "\01?foo_aapad@@YAXAAPAD@Z"
// X64: "\01?foo_aapad@@YAXAEAPEAD@Z"

void foo_aapbd(const char *&x) {}
// CHECK: "\01?foo_aapbd@@YAXAAPBD@Z"
// X64: "\01?foo_aapbd@@YAXAEAPEBD@Z"

void foo_abqad(char * const &x) {}
// CHECK: "\01?foo_abqad@@YAXABQAD@Z"
// X64: "\01?foo_abqad@@YAXAEBQEAD@Z"

void foo_abqbd(const char * const &x) {}
// CHECK: "\01?foo_abqbd@@YAXABQBD@Z"
// X64: "\01?foo_abqbd@@YAXAEBQEBD@Z"

void foo_aay144h(int (&x)[5][5]) {}
// CHECK: "\01?foo_aay144h@@YAXAAY144H@Z"
// X64: "\01?foo_aay144h@@YAXAEAY144H@Z"

void foo_aay144cbh(const int (&x)[5][5]) {}
// CHECK: "\01?foo_aay144cbh@@YAXAAY144$$CBH@Z"
// X64: "\01?foo_aay144cbh@@YAXAEAY144$$CBH@Z"

void foo_qay144h(int (&&x)[5][5]) {}
// CHECK: "\01?foo_qay144h@@YAX$$QAY144H@Z"
// X64: "\01?foo_qay144h@@YAX$$QEAY144H@Z"

void foo_qay144cbh(const int (&&x)[5][5]) {}
// CHECK: "\01?foo_qay144cbh@@YAX$$QAY144$$CBH@Z"
// X64: "\01?foo_qay144cbh@@YAX$$QEAY144$$CBH@Z"

void foo_p6ahxz(int x()) {}
// CHECK: "\01?foo_p6ahxz@@YAXP6AHXZ@Z"
// X64: "\01?foo_p6ahxz@@YAXP6AHXZ@Z"

void foo_a6ahxz(int (&x)()) {}
// CHECK: "\01?foo_a6ahxz@@YAXA6AHXZ@Z"
// X64: "\01?foo_a6ahxz@@YAXA6AHXZ@Z"

void foo_q6ahxz(int (&&x)()) {}
// CHECK: "\01?foo_q6ahxz@@YAX$$Q6AHXZ@Z"
// X64: "\01?foo_q6ahxz@@YAX$$Q6AHXZ@Z"

void foo_qay04h(int x[5][5]) {}
// CHECK: "\01?foo_qay04h@@YAXQAY04H@Z"
// X64: "\01?foo_qay04h@@YAXQEAY04H@Z"

void foo_qay04cbh(const int x[5][5]) {}
// CHECK: "\01?foo_qay04cbh@@YAXQAY04$$CBH@Z"
// X64: "\01?foo_qay04cbh@@YAXQEAY04$$CBH@Z"

typedef double Vector[3];

void foo(Vector*) {}
// CHECK: "\01?foo@@YAXPAY02N@Z"
// X64: "\01?foo@@YAXPEAY02N@Z"

void foo(Vector) {}
// CHECK: "\01?foo@@YAXQAN@Z"
// X64: "\01?foo@@YAXQEAN@Z"

void foo_const(const Vector) {}
// CHECK: "\01?foo_const@@YAXQBN@Z"
// X64: "\01?foo_const@@YAXQEBN@Z"

void foo_volatile(volatile Vector) {}
// CHECK: "\01?foo_volatile@@YAXQCN@Z"
// X64: "\01?foo_volatile@@YAXQECN@Z"

void foo(Vector*, const Vector, const double) {}
// CHECK: "\01?foo@@YAXPAY02NQBNN@Z"
// X64: "\01?foo@@YAXPEAY02NQEBNN@Z"
