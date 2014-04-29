// RUN: rm -rf %t

// Build PCH using A from path 1
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I %S/Inputs/modules-with-same-name/DependsOnA -I %S/Inputs/modules-with-same-name/path1/A -emit-pch -o %t-A.pch %s

// Use the PCH with the same header search options; should be fine
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I %S/Inputs/modules-with-same-name/DependsOnA -I %S/Inputs/modules-with-same-name/path1/A -include-pch %t-A.pch %s -fsyntax-only -Werror

// Different -W options are ok
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I %S/Inputs/modules-with-same-name/DependsOnA -I %S/Inputs/modules-with-same-name/path1/A -include-pch %t-A.pch %s -fsyntax-only -Werror -Wauto-import

// Use the PCH with no way to resolve DependsOnA
// RUN: not %clang_cc1 -fmodules -fmodules-cache-path=%t -include-pch %t-A.pch %s -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-NODOA %s
// CHECK-NODOA: module 'DependsOnA' imported by AST file '{{.*A.pch}}' not found

// Use the PCH with no way to resolve A
// RUN: not %clang_cc1 -fmodules -fmodules-cache-path=%t -I %S/Inputs/modules-with-same-name/DependsOnA -include-pch %t-A.pch %s -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-NOA %s
// CHECK-NOA: module 'A' imported by AST file '{{.*DependsOnA.*pcm}}' not found

// Use the PCH and have it resolve the the other A
// RUN: not %clang_cc1 -fmodules -fmodules-cache-path=%t -I %S/Inputs/modules-with-same-name/DependsOnA -I %S/Inputs/modules-with-same-name/path2/A -include-pch %t-A.pch %s -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-WRONGA %s
// CHECK-WRONGA: module 'A' imported by AST file '{{.*DependsOnA.*pcm}}' found in a different module map file ({{.*path2.*}}) than when the importing AST file was built ({{.*path1.*}})

#ifndef HEADER
#define HEADER
@import DependsOnA;
#endif
