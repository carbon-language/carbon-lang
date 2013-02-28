// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck -check-prefix=WITHOUT %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=thread | FileCheck -check-prefix=TSAN %s
// RUN: echo "src:%s" > %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=thread -fsanitize-blacklist=%t | FileCheck -check-prefix=BL %s

// REQUIRES: shell

// The sanitize_thread attribute should be attached to functions
// when ThreadSanitizer is enabled, unless no_sanitize_thread attribute
// is present.

// WITHOUT:  NoTSAN1{{.*}}) [[NOATTR:#[0-9]+]]
// BL:  NoTSAN1{{.*}}) [[NOATTR:#[0-9]+]]
// TSAN:  NoTSAN1{{.*}}) [[NOATTR:#[0-9]+]]
__attribute__((no_sanitize_thread))
int NoTSAN1(int *a) { return *a; }

// WITHOUT:  NoTSAN2{{.*}}) [[NOATTR]]
// BL:  NoTSAN2{{.*}}) [[NOATTR]]
// TSAN:  NoTSAN2{{.*}}) [[NOATTR]]
__attribute__((no_sanitize_thread))
int NoTSAN2(int *a);
int NoTSAN2(int *a) { return *a; }

// WITHOUT:  TSANOk{{.*}}) [[NOATTR]]
// BL:  TSANOk{{.*}}) [[NOATTR]]
// TSAN: TSANOk{{.*}}) [[WITH:#[0-9]+]]
int TSANOk(int *a) { return *a; }

// WITHOUT:  TemplateTSANOk{{.*}}) [[NOATTR]]
// BL:  TemplateTSANOk{{.*}}) [[NOATTR]]
// TSAN: TemplateTSANOk{{.*}}) [[WITH]]
template<int i>
int TemplateTSANOk() { return i; }

// WITHOUT:  TemplateNoTSAN{{.*}}) [[NOATTR]]
// BL:  TemplateNoTSAN{{.*}}) [[NOATTR]]
// TSAN: TemplateNoTSAN{{.*}}) [[NOATTR]]
template<int i>
__attribute__((no_sanitize_thread))
int TemplateNoTSAN() { return i; }

int force_instance = TemplateTSANOk<42>()
                   + TemplateNoTSAN<42>();

// Check that __cxx_global_var_init* get the sanitize_thread attribute.
int global1 = 0;
int global2 = *(int*)((char*)&global1+1);
// WITHOUT: @__cxx_global_var_init{{.*}}[[NOATTR_NO_TF:#[0-9]+]]
// BL: @__cxx_global_var_init{{.*}}[[NOATTR_NO_TF:#[0-9]+]]
// TSAN: @__cxx_global_var_init{{.*}}[[WITH_NO_TF:#[0-9]+]]

// WITHOUT: attributes [[NOATTR]] = { nounwind{{.*}} }
// WITHOUT: attributes [[NOATTR_NO_TF]] = { nounwind }

// BL: attributes [[NOATTR]] = { nounwind{{.*}} }
// BL: attributes [[NOATTR_NO_TF]] = { nounwind{{.*}} }

// TSAN: attributes [[NOATTR]] = { nounwind{{.*}} }
// TSAN: attributes [[WITH]] = { nounwind sanitize_thread{{.*}} }
// TSAN: attributes [[WITH_NO_TF]] = { nounwind sanitize_thread }
