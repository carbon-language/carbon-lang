// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/non-header-dependency_input.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/sanitize-blacklist.txt %t.dir/Inputs/sanitize-blacklist.txt
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/non-header-dependency.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 | FileCheck %s

#define FOO "foo"

// CHECK: Inputs{{/|\\}}sanitize-blacklist.txt
// CHECK-NEXT: non-header-dependency_input.cpp
