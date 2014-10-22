int DefinedInDifferentFile(int *a);
// RUN: echo "int DefinedInDifferentFile(int *a) { return *a; }" > %t.extra-source.cpp
// RUN: echo "struct S { S(){} ~S(){} };" >> %t.extra-source.cpp
// RUN: echo "S glob_array[5];" >> %t.extra-source.cpp

// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -include %t.extra-source.cpp | FileCheck -check-prefix=WITHOUT %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -include %t.extra-source.cpp -fsanitize=address | FileCheck -check-prefix=ASAN %s

// RUN: echo "fun:*BlacklistedFunction*" > %t.func.blacklist
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -include %t.extra-source.cpp -fsanitize=address -fsanitize-blacklist=%t.func.blacklist | FileCheck -check-prefix=BLFUNC %s

// RUN: echo "src:%s" > %t.file.blacklist
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -include %t.extra-source.cpp -fsanitize=address -fsanitize-blacklist=%t.file.blacklist | FileCheck -check-prefix=BLFILE %s

// FIXME: %t.file.blacklist is like "src:x:\path\to\clang\test\CodeGen\address-safety-attr.cpp"
// REQUIRES: shell

// The sanitize_address attribute should be attached to functions
// when AddressSanitizer is enabled, unless no_sanitize_address attribute
// is present.

// Attributes for function defined in different source file:
// WITHOUT: DefinedInDifferentFile{{.*}} [[NOATTR:#[0-9]+]]
// BLFILE:  DefinedInDifferentFile{{.*}} [[WITH:#[0-9]+]]
// BLFUNC:  DefinedInDifferentFile{{.*}} [[WITH:#[0-9]+]]
// ASAN:    DefinedInDifferentFile{{.*}} [[WITH:#[0-9]+]]

// Check that functions generated for global in different source file are
// not blacklisted.
// WITHOUT: @__cxx_global_var_init{{.*}}[[NOATTR_NO_TF:#[0-9]+]]
// WITHOUT: @__cxx_global_array_dtor{{.*}}[[NOATTR_NO_TF]]
// BLFILE: @__cxx_global_var_init{{.*}}[[WITH_NO_TF:#[0-9]+]]
// BLFILE: @__cxx_global_array_dtor{{.*}}[[WITH_NO_TF]]
// BLFUNC: @__cxx_global_var_init{{.*}}[[WITH_NO_TF:#[0-9]+]]
// BLFUNC: @__cxx_global_array_dtor{{.*}}[[WITH_NO_TF]]
// ASAN: @__cxx_global_var_init{{.*}}[[WITH_NO_TF:#[0-9]+]]
// ASAN: @__cxx_global_array_dtor{{.*}}[[WITH_NO_TF]]


// WITHOUT:  NoAddressSafety1{{.*}}) [[NOATTR]]
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
// BLFUNC: AddressSafetyOk{{.*}}) [[WITH]]
// ASAN: AddressSafetyOk{{.*}}) [[WITH]]
int AddressSafetyOk(int *a) { return *a; }

// WITHOUT:  BlacklistedFunction{{.*}}) [[NOATTR]]
// BLFILE:  BlacklistedFunction{{.*}}) [[NOATTR]]
// BLFUNC:  BlacklistedFunction{{.*}}) [[NOATTR]]
// ASAN:  BlacklistedFunction{{.*}}) [[WITH]]
int BlacklistedFunction(int *a) { return *a; }

#define GENERATE_FUNC(name) \
    int name(int *a) { return *a; }
// WITHOUT: GeneratedFunction{{.*}}) [[NOATTR]]
// BLFILE:  GeneratedFunction{{.*}}) [[NOATTR]]
// BLFUNC:  GeneratedFunction{{.*}}) [[WITH]]
// ASAN:    GeneratedFunction{{.*}}) [[WITH]]
GENERATE_FUNC(GeneratedFunction)

#define GENERATE_NAME(name) name##_generated
// WITHOUT: Function_generated{{.*}}) [[NOATTR]]
// BLFILE:  Function_generated{{.*}}) [[NOATTR]]
// BLFUNC:  Function_generated{{.*}}) [[WITH]]
// ASAN:    Function_generated{{.*}}) [[WITH]]
int GENERATE_NAME(Function)(int *a) { return *a; }

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
// WITHOUT: @__cxx_global_var_init{{.*}}[[NOATTR_NO_TF]]
// BLFILE: @__cxx_global_var_init{{.*}}[[NOATTR_NO_TF:#[0-9]+]]
// BLFUNC: @__cxx_global_var_init{{.*}}[[WITH_NO_TF]]
// ASAN: @__cxx_global_var_init{{.*}}[[WITH_NO_TF]]

// WITHOUT: attributes [[NOATTR]] = { nounwind{{.*}} }
// WITHOUT: attributes [[NOATTR_NO_TF]] = { nounwind }

// BLFILE: attributes [[WITH]] = { nounwind sanitize_address{{.*}} }
// BLFILE: attributes [[WITH_NO_TF]] = { nounwind sanitize_address }
// BLFILE: attributes [[NOATTR_NO_TF]] = { nounwind }
// BLFILE: attributes [[NOATTR]] = { nounwind{{.*}} }

// BLFUNC: attributes [[WITH]] = { nounwind sanitize_address{{.*}} }
// BLFUNC: attributes [[WITH_NO_TF]] = { nounwind sanitize_address }
// BLFUNC: attributes [[NOATTR]] = { nounwind{{.*}} }

// ASAN: attributes [[WITH]] = { nounwind sanitize_address{{.*}} }
// ASAN: attributes [[WITH_NO_TF]] = { nounwind sanitize_address }
// ASAN: attributes [[NOATTR]] = { nounwind{{.*}} }
