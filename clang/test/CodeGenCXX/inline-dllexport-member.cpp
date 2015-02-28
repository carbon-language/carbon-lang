// RUN: %clang_cc1 -triple i686-windows-gnu -fms-compatibility -g -emit-llvm %s -o - \
// RUN:    | FileCheck %s

struct __declspec(dllexport) s {
  static const unsigned int ui = 0;
};

// CHECK: , [[SCOPE:![^,]*]], {{.*}}, i32* @_ZN1s2uiE, {{.*}}} ; [ DW_TAG_variable ] [ui] [line 5] [def]
// CHECK: [[SCOPE]] = {{.*}} ; [ DW_TAG_file_type ]

