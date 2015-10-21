// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -emit-module -fmodule-name=std -fmodules-cache-path=%t %S/Inputs/libstdcxx-ambiguous-internal/module.modulemap -Werror
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -emit-module -fmodule-name=std -fmodules-cache-path=%t %S/Inputs/libstdcxx-ambiguous-internal/module.modulemap -fmodules-local-submodule-visibility -DAMBIGUOUS 2>&1| FileCheck %s

// CHECK-NOT: error
// CHECK: warning: ambiguous use of internal linkage function 'f' defined in multiple modules
// CHECK-NOT: error
