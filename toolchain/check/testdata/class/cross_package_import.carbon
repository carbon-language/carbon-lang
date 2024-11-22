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

import Other library "other_define";

var c: Other.C = {};

// --- fail_extern.carbon

library "[[@TEST_NAME]]";

import Other library "other_extern";

// CHECK:STDERR: fail_extern.carbon:[[@LINE+8]]:8: error: variable has incomplete type `C` [IncompleteTypeInVarDecl]
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
// CHECK:STDERR: other_extern.carbon:4:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
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
// CHECK:STDERR: other_conflict.carbon:4:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn C() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE+4]]:1: in import [InImport]
// CHECK:STDERR: other_define.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
import Other library "other_define";
import Other library "other_conflict";

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+3]]:8: note: in name lookup for `C` [InNameLookup]
// CHECK:STDERR: var c: Other.C = {};
// CHECK:STDERR:        ^~~~~~~
var c: Other.C = {};

// CHECK:STDOUT: --- other_define.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
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
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- other_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [template = constants.%C] {} {}
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
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     import Other//other_define
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Other//other_define, inst+3, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref Other//other_define, inst+4, unloaded
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
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_19.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc6_19.2: init %C = class_init (), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_20: init %C = converted %.loc6_19.1, %.loc6_19.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .C = %import_ref
// CHECK:STDOUT:     import Other//other_extern
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: type = import_ref Other//other_extern, inst+3, loaded [template = constants.%C]
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
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14: %.1 = struct_literal ()
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_merge_define_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     import Other//other_define
// CHECK:STDOUT:     import Other//other_extern
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Other//other_define, inst+3, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref Other//other_define, inst+4, unloaded
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
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_19.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc19_19.2: init %C = class_init (), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc19_20: init %C = converted %.loc19_19.1, %.loc19_19.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc19_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     import Other//other_define
// CHECK:STDOUT:     import Other//other_conflict
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Other//other_define, inst+3, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref Other//other_define, inst+4, unloaded
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
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_19.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc18_19.2: init %C = class_init (), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc18_20: init %C = converted %.loc18_19.1, %.loc18_19.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc18_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
