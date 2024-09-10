// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/cross_package_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/cross_package_import.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- other_define.carbon

package Other library "[[@TEST_NAME]]";

class C {}

// --- other_extern.carbon

package Other library "[[@TEST_NAME]]";

extern class C;

// --- other_conflict.carbon

package Other library "[[@TEST_NAME]]";

fn C() {}

// ============================================================================
// Test files
// ============================================================================

// --- define.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: define.carbon:[[@LINE+4]]:1: ERROR: Imported API 'Other//define' not found.
// CHECK:STDERR: import Other library "define";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Other library "define";

var c: Other.C = {};

// --- fail_extern.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_extern.carbon:[[@LINE+4]]:1: ERROR: Imported API 'Other//extern' not found.
// CHECK:STDERR: import Other library "extern";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Other library "extern";

var c: Other.C = {};

// --- fail_todo_merge_define_extern.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_merge_define_extern.carbon:[[@LINE+4]]:1: ERROR: Imported API 'Other//define' not found.
// CHECK:STDERR: import Other library "define";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Other library "define";
// CHECK:STDERR: fail_todo_merge_define_extern.carbon:[[@LINE+4]]:1: ERROR: Imported API 'Other//extern' not found.
// CHECK:STDERR: import Other library "extern";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Other library "extern";

var c: Other.C = {};

// --- fail_conflict.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+4]]:1: ERROR: Imported API 'Other//define' not found.
// CHECK:STDERR: import Other library "define";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Other library "define";
// CHECK:STDERR: fail_conflict.carbon:[[@LINE+3]]:1: ERROR: Imported API 'Other//conflict' not found.
// CHECK:STDERR: import Other library "conflict";
// CHECK:STDERR: ^~~~~~
import Other library "conflict";

var c: Other.C = {};

// CHECK:STDOUT: --- other_define.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- other_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- other_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- define.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10: %.1 = struct_literal ()
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10: %.1 = struct_literal ()
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_merge_define_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15: %.1 = struct_literal ()
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14: %.1 = struct_literal ()
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
