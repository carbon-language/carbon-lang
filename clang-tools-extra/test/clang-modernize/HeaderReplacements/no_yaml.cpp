// Ensure that if -serialize-replacements is not provided, no serialized
// replacement files should be generated and the changes are made directly.
//
// RUN: rm -rf %T/Inputs
// RUN: mkdir -p %T/Inputs
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/no_yaml.h > %T/Inputs/no_yaml.h
// RUN: clang-modernize -loop-convert %t.cpp -include=%T/Inputs -- -I %T/Inputs/no_yaml.h
// RUN: FileCheck --input-file=%t.cpp %s
// RUN: FileCheck --input-file=%T/Inputs/no_yaml.h %S/Inputs/no_yaml.h
// RUN: ls %T | FileCheck %s --check-prefix=NO_YAML
//
// NO_YAML-NOT: {{no_yaml.cpp_.*.yaml}}
#include "Inputs/no_yaml.h"

void func() {
  int arr[10];
  for (unsigned i = 0; i < sizeof(arr)/sizeof(int); ++i) {
    arr[i] = 0;
    // CHECK: for (auto & elem : arr) {
    // CHECK-NEXT: elem = 0;
  }

  update(arr);
}
