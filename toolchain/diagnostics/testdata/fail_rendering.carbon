// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// ARGS: --include-diagnostic-kind --diagnostic-snippets compile --phase=check %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/diagnostics/testdata/fail_rendering.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/diagnostics/testdata/fail_rendering.carbon

// Covers how diagnostics are drawn, which the rest of the test suite turns off.
// See /toolchain/docs/diagnostics_rendering.md.

// --- fail_one_message.carbon

library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: error: name `missing` not found [NameNotFound]
  // CHECK:STDERR:        .-| fail_one_message.carbon:[[@LINE+13]]:23
  // CHECK:STDERR:        :
  // CHECK:STDERR: ->  19 |   var unused_x: i32 = missing;
  // CHECK:STDERR:        :                       -------
  // CHECK:STDERR:        '------------
  // CHECK:STDERR:
  // CHECK:STDERR: warning: binding `unused_x` unused [UnusedBinding]
  // CHECK:STDERR:        .-| fail_one_message.carbon:[[@LINE+6]]:7
  // CHECK:STDERR:        :
  // CHECK:STDERR: ->  19 |   var unused_x: i32 = missing;
  // CHECK:STDERR:        :       -------------
  // CHECK:STDERR:        '------------
  // CHECK:STDERR:
  var unused_x: i32 = missing;
}

// --- fail_note_in_same_file.carbon

library "[[@TEST_NAME]]";

fn G() {}

fn H() {
  // CHECK:STDERR: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:        .-| fail_note_in_same_file.carbon:[[@LINE+10]]:3
  // CHECK:STDERR:        :
  // CHECK:STDERR:      4 | fn G() {}
  // CHECK:STDERR:        : --------
  // CHECK:STDERR:        :     '--| calling function declared here [InCallToEntity]
  // CHECK:STDERR:        :
  // CHECK:STDERR: ->  18 |   G(1);
  // CHECK:STDERR:        :   ----
  // CHECK:STDERR:        '------------
  // CHECK:STDERR:
  G(1);
}

// --- fail_wide_line_numbers.carbon

library "[[@TEST_NAME]]";

fn Wide() {
  // Padding, so that the diagnostic below lands past line 9 and the frame has
  // to move right to fit its line number.
  //
  //
  //
  // CHECK:STDERR: error: name `also_missing` not found [NameNotFound]
  // CHECK:STDERR:        .-| fail_wide_line_numbers.carbon:[[@LINE+13]]:23
  // CHECK:STDERR:        :
  // CHECK:STDERR: ->  24 |   var unused_y: i32 = also_missing;
  // CHECK:STDERR:        :                       ------------
  // CHECK:STDERR:        '------------
  // CHECK:STDERR:
  // CHECK:STDERR: warning: binding `unused_y` unused [UnusedBinding]
  // CHECK:STDERR:        .-| fail_wide_line_numbers.carbon:[[@LINE+6]]:7
  // CHECK:STDERR:        :
  // CHECK:STDERR: ->  24 |   var unused_y: i32 = also_missing;
  // CHECK:STDERR:        :       -------------
  // CHECK:STDERR:        '------------
  // CHECK:STDERR:
  var unused_y: i32 = also_missing;
}

// --- declared.carbon

library "[[@TEST_NAME]]";

fn Declared();

// --- fail_across_files.carbon

library "[[@TEST_NAME]]";

import library "declared";

fn Use() {
  // CHECK:STDERR: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:        .-| fail_across_files.carbon:[[@LINE+13]]:3
  // CHECK:STDERR:        :
  // CHECK:STDERR: ->  21 |   Declared(1);
  // CHECK:STDERR:        :   -----------
  // CHECK:STDERR:        |------------
  // CHECK:STDERR:        | .-- imported from: fail_across_files.carbon:[[@LINE-9]]:1 [InImport]
  // CHECK:STDERR:        |-| declared.carbon:4:1
  // CHECK:STDERR:        :
  // CHECK:STDERR:      4 | fn Declared();
  // CHECK:STDERR:        : --------------
  // CHECK:STDERR:        :        '--| calling function declared here [InCallToEntity]
  // CHECK:STDERR:        '------------
  // CHECK:STDERR:
  Declared(1);
}

// --- fail_two_spans_on_one_line.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
// CHECK:STDERR:        .-| fail_two_spans_on_one_line.carbon:[[@LINE+7]]:27
// CHECK:STDERR:        :
// CHECK:STDERR: ->  12 | fn Same() {} fn Twice() { Same(1); }
// CHECK:STDERR:        : -----------               -------
// CHECK:STDERR:        :      '--| calling function declared here [InCallToEntity]
// CHECK:STDERR:        '------------
// CHECK:STDERR:
fn Same() {} fn Twice() { Same(1); }

// --- fail_non_ascii_source.carbon

library "[[@TEST_NAME]]";

fn Wide() {
  // CHECK:STDERR: error: name `Core.String` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:        .-| fail_non_ascii_source.carbon:[[@LINE+13]]:19
  // CHECK:STDERR:        :
  // CHECK:STDERR: ->  19 |   var text: i32 = "<E4><B8><96><E7><95><8C>";
  // CHECK:STDERR:        :                   --------------------------
  // CHECK:STDERR:        '------------
  // CHECK:STDERR:
  // CHECK:STDERR: warning: binding `text` unused [UnusedBinding]
  // CHECK:STDERR:        .-| fail_non_ascii_source.carbon:[[@LINE+6]]:7
  // CHECK:STDERR:        :
  // CHECK:STDERR: ->  19 |   var text: i32 = "<E4><B8><96><E7><95><8C>";
  // CHECK:STDERR:        :       ---------
  // CHECK:STDERR:        '------------
  // CHECK:STDERR:
  var text: i32 = "世界";
}

// --- fail_tab_indented.carbon

library "[[@TEST_NAME]]";

fn Tabbed() {
// CHECK:STDERR: error: name `nowhere` not found [NameNotFound]
// CHECK:STDERR:        .-| fail_tab_indented.carbon:[[@LINE+13]]:26
// CHECK:STDERR:        :
// CHECK:STDERR: ->  19 |         var also_missing: i32 = nowhere;
// CHECK:STDERR:        :                                 -------
// CHECK:STDERR:        '------------
// CHECK:STDERR:
// CHECK:STDERR: warning: binding `also_missing` unused [UnusedBinding]
// CHECK:STDERR:        .-| fail_tab_indented.carbon:[[@LINE+6]]:6
// CHECK:STDERR:        :
// CHECK:STDERR: ->  19 |         var also_missing: i32 = nowhere;
// CHECK:STDERR:        :             -----------------
// CHECK:STDERR:        '------------
// CHECK:STDERR:
	var also_missing: i32 = nowhere;
}

// --- cpp_decl.h

struct Widget {};
void Use(Widget w, int extra);

// --- fail_cpp_interop.carbon

library "[[@TEST_NAME]]";

import Cpp library "cpp_decl.h";

fn CallIt() {
  // CHECK:STDERR: error: no matching function for call to 'Use' [CppInteropParseError]
  // CHECK:STDERR:        .-| fail_cpp_interop.carbon:[[@LINE+15]]:3
  // CHECK:STDERR:        :
  // CHECK:STDERR: ->  23 |   Cpp.Use(1);
  // CHECK:STDERR:        :   -------
  // CHECK:STDERR:        |------------
  // CHECK:STDERR:        | .-- included from: fail_cpp_interop.carbon:[[@LINE-9]]:10 [InCppInclude]
  // CHECK:STDERR:        |-| ./cpp_decl.h:3:6
  // CHECK:STDERR:        :
  // CHECK:STDERR:      3 | void Use(Widget w, int extra);
  // CHECK:STDERR:        :      --- -------------------
  // CHECK:STDERR:        :       |           '--| requires 2 arguments, but 1 was provided
  // CHECK:STDERR:        :       |              | [CppInteropParseNoteReason]
  // CHECK:STDERR:        :       '--| candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR:        '------------
  // CHECK:STDERR:
  Cpp.Use(1);
}
