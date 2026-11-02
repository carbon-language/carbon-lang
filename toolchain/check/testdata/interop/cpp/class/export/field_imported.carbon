// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/export/field_imported.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/export/field_imported.carbon

// --- other.carbon
package Other;

class Point {
  var x: i32;
  var y: i32;
}

// --- imported_field.carbon
library "[[@TEST_NAME]]";

import Other;
import Cpp;

inline Cpp '''
void AccessImportedFields() {
  Carbon::Other::Point p;
  p.x = 12;
  p.y = 34;
}
''';

// --- imported_field_same_package.carbon
library "[[@TEST_NAME]]";

import library "point";
import Cpp;

inline Cpp '''
void AccessImportedFields() {
  Carbon::Point p;
  p.x = 12;
}
''';

// --- point.carbon
library "point";

class Point {
  var x: i32;
}
