// RUN: rm -rf %T
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%T -fmodule-name=TBranchProxy -emit-module -x c++ %S/Inputs/PR20786/module.modulemap
