#define FOO(x) x
#define BAR(x) FOO(x)
#define WIBBLE(x) BAR(x)

WIBBLE(int x);

// RUN: env CINDEXTEST_NESTED_MACROS=1 c-index-test -test-load-source all %s | FileCheck -check-prefix CHECK-WITH-NESTED %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_NESTED_MACROS=1 c-index-test -test-load-source all %s | FileCheck -check-prefix CHECK-WITH-NESTED %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_NESTED_MACROS=1 c-index-test -test-load-source-reparse 5 all %s | FileCheck -check-prefix CHECK-WITH-NESTED %s
// CHECK-WITH-NESTED: nested-macro-instantiations.cpp:5:1: macro instantiation=WIBBLE:3:9 Extent=[5:1 - 5:7]
// CHECK-WITH-NESTED: nested-macro-instantiations.cpp:3:19: macro instantiation=BAR:2:9 Extent=[3:19 - 5:14]
// CHECK-WITH-NESTED: nested-macro-instantiations.cpp:2:16: macro instantiation=FOO:1:9 Extent=[2:16 - 5:14]
// CHECK-WITH-NESTED: nested-macro-instantiations.cpp:5:1: VarDecl=x:5:1 (Definition) Extent=[5:1 - 5:14]

// RUN: c-index-test -test-load-source all %s | FileCheck -check-prefix CHECK-WITHOUT-NESTED %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source all %s | FileCheck -check-prefix CHECK-WITHOUT-NESTED %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 5 all %s | FileCheck -check-prefix CHECK-WITHOUT-NESTED %s
// CHECK-WITHOUT-NESTED: nested-macro-instantiations.cpp:5:1: macro instantiation=WIBBLE:3:9 Extent=[5:1 - 5:7]
// CHECK-WITHOUT-NESTED-NOT: nested-macro-instantiations.cpp:3:19: macro instantiation=BAR:2:9 Extent=[3:19 - 5:14]
// CHECK-WITHOUT-NESTED: nested-macro-instantiations.cpp:5:1: VarDecl=x:5:1 (Definition) Extent=[5:1 - 5:14]
