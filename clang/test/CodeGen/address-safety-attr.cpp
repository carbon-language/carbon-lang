// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck -check-prefix=WITHOUT %s
// RUN: %clang_cc1 -emit-llvm -o - %s -fsanitize=address | FileCheck -check-prefix=ASAN %s
// RUN: echo "src:%s" > %t
// RUN: %clang_cc1 -emit-llvm -o - %s -fsanitize=address -fsanitize-blacklist=%t | FileCheck -check-prefix=BL %s

// FIXME: %t is like "src:x:\path\to\clang\test\CodeGen\address-safety-attr.cpp"
// REQUIRES: shell

// The address_safety attribute should be attached to functions
// when AddressSanitizer is enabled, unless no_address_safety_analysis attribute
// is present.

// WITHOUT:  NoAddressSafety1{{.*}}) #[[NOATTR:[0-9]+]]
// BL:  NoAddressSafety1{{.*}}) #[[NOATTR:[0-9]+]]
// ASAN:  NoAddressSafety1{{.*}}) #[[NOATTR:[0-9]+]]
__attribute__((no_address_safety_analysis))
int NoAddressSafety1(int *a) { return *a; }

// WITHOUT:  NoAddressSafety2{{.*}}) #[[NOATTR]]
// BL:  NoAddressSafety2{{.*}}) #[[NOATTR]]
// ASAN:  NoAddressSafety2{{.*}}) #[[NOATTR]]
__attribute__((no_address_safety_analysis))
int NoAddressSafety2(int *a);
int NoAddressSafety2(int *a) { return *a; }

// WITHOUT:  AddressSafetyOk{{.*}}) #[[NOATTR]]
// BL:  AddressSafetyOk{{.*}}) #[[NOATTR]]
// ASAN: AddressSafetyOk{{.*}}) #[[WITH:[0-9]+]]
int AddressSafetyOk(int *a) { return *a; }

// WITHOUT:  TemplateAddressSafetyOk{{.*}}) #[[NOATTR]]
// BL:  TemplateAddressSafetyOk{{.*}}) #[[NOATTR]]
// ASAN: TemplateAddressSafetyOk{{.*}}) #[[WITH]]
template<int i>
int TemplateAddressSafetyOk() { return i; }

// WITHOUT:  TemplateNoAddressSafety{{.*}}) #[[NOATTR]]
// BL:  TemplateNoAddressSafety{{.*}}) #[[NOATTR]]
// ASAN: TemplateNoAddressSafety{{.*}}) #[[NOATTR]]
template<int i>
__attribute__((no_address_safety_analysis))
int TemplateNoAddressSafety() { return i; }

int force_instance = TemplateAddressSafetyOk<42>()
                   + TemplateNoAddressSafety<42>();

// Check that __cxx_global_var_init* get the address_safety attribute.
int global1 = 0;
int global2 = *(int*)((char*)&global1+1);
// WITHOUT: @__cxx_global_var_init{{.*}}#1
// BL: @__cxx_global_var_init{{.*}}#1
// ASAN: @__cxx_global_var_init{{.*}}#2

// WITHOUT: attributes #[[NOATTR]] = { nounwind "target-features"={{.*}} }
// BL: attributes #[[NOATTR]] = { nounwind "target-features"={{.*}} }

// ASAN: attributes #[[NOATTR]] = { nounwind "target-features"={{.*}} }
// ASAN: attributes #[[WITH]] = { address_safety nounwind "target-features"={{.*}} }
