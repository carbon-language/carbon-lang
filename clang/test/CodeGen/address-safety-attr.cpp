// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - %s -fsanitize=address | FileCheck -check-prefix ASAN %s
// RUN: echo "src:%s" > %t
// RUN: %clang_cc1 -emit-llvm -o - %s -fsanitize=address -fsanitize-blacklist=%t | FileCheck %s

// FIXME: %t is like "src:x:\path\to\clang\test\CodeGen\address-safety-attr.cpp"
// REQUIRES: shell

// The address_safety attribute should be attached to functions
// when AddressSanitizer is enabled, unless no_address_safety_analysis attribute
// is present.

// CHECK:  NoAddressSafety1{{.*}}#0
// ASAN:  NoAddressSafety1{{.*}}#0
__attribute__((no_address_safety_analysis))
int NoAddressSafety1(int *a) { return *a; }

// CHECK:  NoAddressSafety2{{.*}}#0
// ASAN:  NoAddressSafety2{{.*}}#0
__attribute__((no_address_safety_analysis))
int NoAddressSafety2(int *a);
int NoAddressSafety2(int *a) { return *a; }

// CHECK:  AddressSafetyOk{{.*}}#0
// ASAN: AddressSafetyOk{{.*}}#1
int AddressSafetyOk(int *a) { return *a; }

// CHECK:  TemplateAddressSafetyOk{{.*}}#0
// ASAN: TemplateAddressSafetyOk{{.*}}#1
template<int i>
int TemplateAddressSafetyOk() { return i; }

// CHECK:  TemplateNoAddressSafety{{.*}}#0
// ASAN: TemplateNoAddressSafety{{.*}}#0
template<int i>
__attribute__((no_address_safety_analysis))
int TemplateNoAddressSafety() { return i; }

int force_instance = TemplateAddressSafetyOk<42>()
                   + TemplateNoAddressSafety<42>();

// Check that __cxx_global_var_init* get the address_safety attribute.
int global1 = 0;
int global2 = *(int*)((char*)&global1+1);
// CHECK: @__cxx_global_var_init{{.*}}#1
// ASAN: @__cxx_global_var_init{{.*}}#2

// CHECK: attributes #0 = { nounwind "target-features"={{.*}} }
// CHECK: attributes #1 = { nounwind }

// ASAN: attributes #0 = { nounwind "target-features"={{.*}} }
// ASAN: attributes #1 = { address_safety nounwind "target-features"={{.*}} }
// ASAN: attributes #2 = { address_safety nounwind }
