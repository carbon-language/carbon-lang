// RUN: %clang_cc1 -E -std=c++0x %s -o - | FileCheck --check-prefix=CHECK-0X %s
// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-NO-0X %s

#if __has_feature(cxx_lambdas)
int lambdas();
#else
int no_lambdas();
#endif

// CHECK-0X: no_lambdas
// CHECK-NO-0X: no_lambdas


#if __has_feature(cxx_nullptr)
int has_nullptr();
#else
int no_nullptr();
#endif

// CHECK-0X: no_nullptr
// CHECK-NO-0X: no_nullptr


#if __has_feature(cxx_concepts)
int concepts();
#else
int no_concepts();
#endif

// CHECK-0X: no_concepts
// CHECK-NO-0X: no_concepts


#if __has_feature(cxx_decltype)
int has_decltype();
#else
int no_decltype();
#endif

// CHECK-0X: has_decltype
// CHECK-NO-0X: no_decltype


#if __has_feature(cxx_auto_type)
int auto_type();
#else
int no_auto_type();
#endif

// CHECK-0X: auto_type
// CHECK-NO-0X: no_auto_type


#if __has_feature(cxx_attributes)
int attributes();
#else
int no_attributes();
#endif

// CHECK-0X: attributes
// CHECK-NO-0X: no_attributes


#if __has_feature(cxx_static_assert)
int has_static_assert();
#else
int no_static_assert();
#endif

// CHECK-0X: has_static_assert
// CHECK-NO-0X: no_static_assert

// We accept this as an extension.
#if __has_feature(cxx_deleted_functions)
int deleted_functions();
#else
int no_deleted_functions();
#endif

// CHECK-0X: deleted_functions
// CHECK-NO-0X: deleted_functions


#if __has_feature(cxx_rvalue_references)
int rvalue_references();
#else
int no_rvalue_references();
#endif

// CHECK-0X: no_rvalue_references
// CHECK-NO-0X: no_rvalue_references


#if __has_feature(cxx_variadic_templates)
int variadic_templates();
#else
int no_variadic_templates();
#endif

// CHECK-0X: variadic_templates
// Note: We allow variadic templates in C++98/03 with a warning.
// CHECK-NO-0X: variadic_templates


#if __has_feature(cxx_inline_namespaces)
int inline_namespaces();
#else
int no_inline_namespaces();
#endif

// CHECK-0X: inline_namespaces
// CHECK-NO-0X: inline_namespaces
