// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck -check-prefix=WITHOUT %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=address | FileCheck -check-prefix=ASAN %s
// RUN: echo "src:%s" > %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=address -fsanitize-blacklist=%t | FileCheck -check-prefix=BL %s

// FIXME: %t is like "src:x:\path\to\clang\test\CodeGen\address-safety-attr.cpp"
// REQUIRES: shell

// The sanitize_address attribute should be attached to functions
// when AddressSanitizer is enabled, unless no_sanitize_address attribute
// is present.

// WITHOUT:  NoAddressSafety1{{.*}}) [[NOATTR:#[0-9]+]]
// BL:  NoAddressSafety1{{.*}}) [[NOATTR:#[0-9]+]]
// ASAN:  NoAddressSafety1{{.*}}) [[NOATTR:#[0-9]+]]
__attribute__((no_sanitize_address))
int NoAddressSafety1(int *a) { return *a; }

// WITHOUT:  NoAddressSafety2{{.*}}) [[NOATTR]]
// BL:  NoAddressSafety2{{.*}}) [[NOATTR]]
// ASAN:  NoAddressSafety2{{.*}}) [[NOATTR]]
__attribute__((no_sanitize_address))
int NoAddressSafety2(int *a);
int NoAddressSafety2(int *a) { return *a; }

// WITHOUT:  AddressSafetyOk{{.*}}) [[NOATTR]]
// BL:  AddressSafetyOk{{.*}}) [[NOATTR]]
// ASAN: AddressSafetyOk{{.*}}) [[WITH:#[0-9]+]]
int AddressSafetyOk(int *a) { return *a; }

// WITHOUT:  TemplateAddressSafetyOk{{.*}}) [[NOATTR]]
// BL:  TemplateAddressSafetyOk{{.*}}) [[NOATTR]]
// ASAN: TemplateAddressSafetyOk{{.*}}) [[WITH]]
template<int i>
int TemplateAddressSafetyOk() { return i; }

// WITHOUT:  TemplateNoAddressSafety{{.*}}) [[NOATTR]]
// BL:  TemplateNoAddressSafety{{.*}}) [[NOATTR]]
// ASAN: TemplateNoAddressSafety{{.*}}) [[NOATTR]]
template<int i>
__attribute__((no_sanitize_address))
int TemplateNoAddressSafety() { return i; }

int force_instance = TemplateAddressSafetyOk<42>()
                   + TemplateNoAddressSafety<42>();

// Check that __cxx_global_var_init* get the sanitize_address attribute.
int global1 = 0;
int global2 = *(int*)((char*)&global1+1);
// WITHOUT: @__cxx_global_var_init{{.*}}[[NOATTR_NO_TF:#[0-9]+]]
// BL: @__cxx_global_var_init{{.*}}[[NOATTR_NO_TF:#[0-9]+]]
// ASAN: @__cxx_global_var_init{{.*}}[[WITH_NO_TF:#[0-9]+]]

// WITHOUT: attributes [[NOATTR]] = { nounwind{{.*}} }
// WITHOUT: attributes [[NOATTR_NO_TF]] = { nounwind }

// BL: attributes [[NOATTR]] = { nounwind{{.*}} }
// BL: attributes [[NOATTR_NO_TF]] = { nounwind }

// ASAN: attributes [[NOATTR]] = { nounwind{{.*}} }
// ASAN: attributes [[WITH]] = { nounwind sanitize_address{{.*}} }
// ASAN: attributes [[WITH_NO_TF]] = { nounwind sanitize_address }
