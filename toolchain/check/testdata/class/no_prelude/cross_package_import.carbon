// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/cross_package_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/cross_package_import.carbon

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

import Other library "other_define";

//@dump-sem-ir-begin
var c: Other.C = {};
//@dump-sem-ir-end

// --- fail_extern.carbon

library "[[@TEST_NAME]]";

import Other library "other_extern";

// CHECK:STDERR: fail_extern.carbon:[[@LINE+8]]:8: error: binding pattern has incomplete type `C` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var c: Other.C = {};
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR: fail_extern.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: other_extern.carbon:4:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
var c: Other.C = {};

// --- fail_todo_merge_define_extern.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_merge_define_extern.carbon:[[@LINE+8]]:1: in import [InImport]
// CHECK:STDERR: other_extern.carbon:4:1: error: duplicate name `C` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_merge_define_extern.carbon:[[@LINE+4]]:1: in import [InImport]
// CHECK:STDERR: other_define.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
import Other library "other_define";
import Other library "other_extern";

// CHECK:STDERR: fail_todo_merge_define_extern.carbon:[[@LINE+4]]:8: note: in name lookup for `C` [InNameLookup]
// CHECK:STDERR: var c: Other.C = {};
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR:
var c: Other.C = {};

// --- fail_conflict.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+8]]:1: in import [InImport]
// CHECK:STDERR: other_conflict.carbon:4:1: error: duplicate name `C` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn C() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE+4]]:1: in import [InImport]
// CHECK:STDERR: other_define.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
import Other library "other_define";
import Other library "other_conflict";

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+4]]:8: note: in name lookup for `C` [InNameLookup]
// CHECK:STDERR: var c: Other.C = {};
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR:
var c: Other.C = {};

// CHECK:STDOUT: --- define.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .C = %Other.C
// CHECK:STDOUT:     import Other//other_define
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.C: type = import_ref Other//other_define, C, loaded [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:     %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_19.2: init %C = class_init (), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_19.1, %.loc7_19.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
