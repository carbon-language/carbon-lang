// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fmodule-name=TBranchProxy -emit-module -x c++ %S/Inputs/PR20786/module.modulemap
