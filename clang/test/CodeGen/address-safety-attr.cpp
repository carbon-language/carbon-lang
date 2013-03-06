// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck -check-prefix=WITHOUT %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=address | FileCheck -check-prefix=ASAN %s
// RUN: echo "src:%s" > %t.file.blacklist
// RUN: echo "fun:*BlacklistedFunction*" > %t.func.blacklist
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=address -fsanitize-blacklist=%t.file.blacklist | FileCheck -check-prefix=BLFILE %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=address -fsanitize-blacklist=%t.func.blacklist | FileCheck -check-prefix=BLFUNC %s

// FIXME: %t.file.blacklist is like "src:x:\path\to\clang\test\CodeGen\address-safety-attr.cpp"
// REQUIRES: shell

// The sanitize_address attribute should be attached to functions
// when AddressSanitizer is enabled, unless no_sanitize_address attribute
// is present.

// WITHOUT:  NoAddressSafety1{{.*}}) [[NOATTR:#[0-9]+]]
// BLFILE:  NoAddressSafety1{{.*}}) [[NOATTR:#[0-9]+]]
// BLFUNC:  NoAddressSafety1{{.*}}) [[NOATTR:#[0-9]+]]
// ASAN:  NoAddressSafety1{{.*}}) [[NOATTR:#[0-9]+]]
__attribute__((no_sanitize_address))
int NoAddressSafety1(int *a) { return *a; }

// WITHOUT:  NoAddressSafety2{{.*}}) [[NOATTR]]
// BLFILE:  NoAddressSafety2{{.*}}) [[NOATTR]]
// BLFUNC:  NoAddressSafety2{{.*}}) [[NOATTR]]
// ASAN:  NoAddressSafety2{{.*}}) [[NOATTR]]
__attribute__((no_sanitize_address))
int NoAddressSafety2(int *a);
int NoAddressSafety2(int *a) { return *a; }

// WITHOUT:  AddressSafetyOk{{.*}}) [[NOATTR]]
// BLFILE:  AddressSafetyOk{{.*}}) [[NOATTR]]
// BLFUNC: AddressSafetyOk{{.*}}) [[WITH:#[0-9]+]]
// ASAN: AddressSafetyOk{{.*}}) [[WITH:#[0-9]+]]
int AddressSafetyOk(int *a) { return *a; }

// WITHOUT:  BlacklistedFunction{{.*}}) [[NOATTR]]
// BLFILE:  BlacklistedFunction{{.*}}) [[NOATTR]]
// BLFUNC:  BlacklistedFunction{{.*}}) [[NOATTR]]
// ASAN:  BlacklistedFunction{{.*}}) [[WITH]]
int BlacklistedFunction(int *a) { return *a; }

// WITHOUT:  TemplateAddressSafetyOk{{.*}}) [[NOATTR]]
// BLFILE:  TemplateAddressSafetyOk{{.*}}) [[NOATTR]]
// BLFUNC:  TemplateAddressSafetyOk{{.*}}) [[WITH]]
// ASAN: TemplateAddressSafetyOk{{.*}}) [[WITH]]
template<int i>
int TemplateAddressSafetyOk() { return i; }

// WITHOUT:  TemplateNoAddressSafety{{.*}}) [[NOATTR]]
// BLFILE:  TemplateNoAddressSafety{{.*}}) [[NOATTR]]
// BLFUNC:  TemplateNoAddressSafety{{.*}}) [[NOATTR]]
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
// BLFILE: @__cxx_global_var_init{{.*}}[[NOATTR_NO_TF:#[0-9]+]]
// BLFUNC: @__cxx_global_var_init{{.*}}[[WITH_NO_TF:#[0-9]+]]
// ASAN: @__cxx_global_var_init{{.*}}[[WITH_NO_TF:#[0-9]+]]

// WITHOUT: attributes [[NOATTR]] = { nounwind{{.*}} }
// WITHOUT: attributes [[NOATTR_NO_TF]] = { nounwind }

// BLFILE: attributes [[NOATTR]] = { nounwind{{.*}} }
// BLFILE: attributes [[NOATTR_NO_TF]] = { nounwind }

// BLFUNC: attributes [[NOATTR]] = { nounwind{{.*}} }
// BLFUNC: attributes [[WITH]] = { nounwind sanitize_address{{.*}} }
// BLFUNC: attributes [[WITH_NO_TF]] = { nounwind sanitize_address }

// ASAN: attributes [[NOATTR]] = { nounwind{{.*}} }
// ASAN: attributes [[WITH]] = { nounwind sanitize_address{{.*}} }
// ASAN: attributes [[WITH_NO_TF]] = { nounwind sanitize_address }
