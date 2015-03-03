// RUN: sed -e "s:INPUT_DIR:%S/Inputs:g" -e "s:OUT_DIR:%t:g" -e "s:EXTERNAL_NAMES:true:" %S/Inputs/use-external-names.yaml > %t.external.yaml
// RUN: sed -e "s:INPUT_DIR:%S/Inputs:g" -e "s:OUT_DIR:%t:g" -e "s:EXTERNAL_NAMES:false:" %S/Inputs/use-external-names.yaml > %t.yaml
// REQUIRES: shell

#include "external-names.h"

////
// Preprocessor (__FILE__ macro and # directives):

// RUN: %clang_cc1 -I %t -ivfsoverlay %t.external.yaml -E %s | FileCheck -check-prefix=CHECK-PP-EXTERNAL %s
// CHECK-PP-EXTERNAL: # {{[0-9]*}} "[[NAME:.*Inputs.external-names.h]]"
// CHECK-PP-EXTERNAL-NEXT: void foo(char **c) {
// CHECK-PP-EXTERNAL-NEXT: *c = "[[NAME]]";

// RUN: %clang_cc1 -I %t -ivfsoverlay %t.yaml -E %s | FileCheck -check-prefix=CHECK-PP %s
// CHECK-PP-NOT: Inputs

////
// Diagnostics:

// RUN: %clang_cc1 -I %t -ivfsoverlay %t.external.yaml -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-DIAG-EXTERNAL %s
// CHECK-DIAG-EXTERNAL: {{.*}}Inputs{{.}}external-names.h:{{[0-9]*:[0-9]*}}: warning: incompatible pointer

// RUN: %clang_cc1 -I %t -ivfsoverlay %t.yaml -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-DIAG %s
// CHECK-DIAG-NOT: Inputs

////
// Debug info

// RUN: %clang_cc1 -I %t -ivfsoverlay %t.external.yaml -triple %itanium_abi_triple -g -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-DEBUG-EXTERNAL %s
// CHECK-DEBUG-EXTERNAL: !MDSubprogram({{.*}}file: ![[Num:[0-9]+]]
// CHECK-DEBUG-EXTERNAL: ![[Num]] = !MDFile(filename: "{{[^"]*}}Inputs{{.}}external-names.h"

// RUN: %clang_cc1 -I %t -ivfsoverlay %t.yaml -triple %itanium_abi_triple -g -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-DEBUG %s
// CHECK-DEBUG-NOT: Inputs
