// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/no_prelude/implicit_imports_entities.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/no_prelude/implicit_imports_entities.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- c1.carbon

library "[[@TEST_NAME]]";

class C1 {}

// --- c2.carbon

library "[[@TEST_NAME]]";

class C2 {}

// --- ns.carbon

library "[[@TEST_NAME]]";

namespace NS;
class NS.C {}

// --- o1.carbon

package Other library "[[@TEST_NAME]]";

class O1 {}

// --- o2.carbon

package Other library "[[@TEST_NAME]]";

class O2 {}

// --- local_other.carbon

library "[[@TEST_NAME]]";

class Other {}

// ============================================================================
// Test files
// ============================================================================

// --- mix_current_package.carbon

library "[[@TEST_NAME]]";

import library "c1";

// --- mix_current_package.impl.carbon

impl library "[[@TEST_NAME]]";

import library "c2";

var c1: C1 = {};
var c2: C2 = {};

// --- dup_c1.carbon

library "[[@TEST_NAME]]";

import library "c1";

// --- dup_c1.impl.carbon

impl library "[[@TEST_NAME]]";

import library "c1";

var c1: C1 = {};

// --- use_ns.carbon

library "[[@TEST_NAME]]";

import library "ns";

// --- use_ns.impl.carbon

impl library "[[@TEST_NAME]]";

var c: NS.C = {};

// --- mix_other.carbon

library "[[@TEST_NAME]]";

import Other library "o1";

// --- mix_other.impl.carbon

impl library "[[@TEST_NAME]]";

import Other library "o2";

var o1: Other.O1 = {};
var o2: Other.O2 = {};

// --- dup_o1.carbon

library "[[@TEST_NAME]]";

import Other library "o1";

// --- dup_o1.impl.carbon

impl library "[[@TEST_NAME]]";

import Other library "o1";

var o1: Other.O1 = {};

// --- import_conflict.carbon

library "[[@TEST_NAME]]";

import Other library "o1";

// --- fail_import_conflict.impl.carbon

// CHECK:STDERR: fail_import_conflict.impl.carbon:[[@LINE+9]]:1: in import [InImport]
// CHECK:STDERR: import_conflict.carbon:4:1: error: duplicate name `Other` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: import Other library "o1";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_import_conflict.impl.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: local_other.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: class Other {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl library "[[@TEST_NAME]]";

import library "local_other";

// --- import_conflict_reverse.carbon

library "[[@TEST_NAME]]";

import library "local_other";

// --- fail_import_conflict_reverse.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_conflict_reverse.impl.carbon:[[@LINE+9]]:1: error: duplicate name `Other` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: import Other library "o1";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_import_conflict_reverse.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: import_conflict_reverse.carbon:4:1: in import [InImport]
// CHECK:STDERR: local_other.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: class Other {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
import Other library "o1";

// CHECK:STDOUT: --- c1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C1: type = class_type @C1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C1 = %C1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1 [concrete = constants.%C1] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- c2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C2: type = class_type @C2 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C2 = %C2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2 [concrete = constants.%C2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ns.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- o1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %O1: type = class_type @O1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .O1 = %O1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %O1.decl: type = class_decl @O1 [concrete = constants.%O1] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @O1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%O1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- o2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %O2: type = class_type @O2 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .O2 = %O2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %O2.decl: type = class_decl @O2 [concrete = constants.%O2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @O2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%O2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- local_other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Other: type = class_type @Other [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = %Other.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.decl: type = class_decl @Other [concrete = constants.%Other] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Other {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Other
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- mix_current_package.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C1 = import_ref Main//c1, C1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C1 = imports.%Main.C1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- mix_current_package.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C1: type = class_type @C1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C1.val: %C1 = struct_value () [concrete]
// CHECK:STDOUT:   %C2: type = class_type @C2 [concrete]
// CHECK:STDOUT:   %C2.val: %C2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C1: type = import_ref Main//mix_current_package, C1, loaded [concrete = constants.%C1]
// CHECK:STDOUT:   %Main.C2: type = import_ref Main//c2, C2, loaded [concrete = constants.%C2]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//c1, loc4_11, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.eb7 = import_ref Main//c1, inst14 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//c2, loc4_11, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.5b0 = import_ref Main//c2, inst14 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C1 = imports.%Main.C1
// CHECK:STDOUT:     .C2 = imports.%Main.C2
// CHECK:STDOUT:     .c1 = %c1
// CHECK:STDOUT:     .c2 = %c2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_35.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_35.2 = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %C1 = binding_pattern c1
// CHECK:STDOUT:     %.loc6: %C1 = var_pattern %c1.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.var: ref %C1 = var c1
// CHECK:STDOUT:   %C1.ref: type = name_ref C1, imports.%Main.C1 [concrete = constants.%C1]
// CHECK:STDOUT:   %c1: ref %C1 = bind_name c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %C2 = binding_pattern c2
// CHECK:STDOUT:     %.loc7: %C2 = var_pattern %c2.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C2 = var c2
// CHECK:STDOUT:   %C2.ref: type = name_ref C2, imports.%Main.C2 [concrete = constants.%C2]
// CHECK:STDOUT:   %c2: ref %C2 = bind_name c2, %c2.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1 [from "c1.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.eb7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 [from "c2.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.5b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc6_15.2: init %C1 = class_init (), file.%c1.var [concrete = constants.%C1.val]
// CHECK:STDOUT:   %.loc6_1: init %C1 = converted %.loc6_15.1, %.loc6_15.2 [concrete = constants.%C1.val]
// CHECK:STDOUT:   assign file.%c1.var, %.loc6_1
// CHECK:STDOUT:   %.loc7_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_15.2: init %C2 = class_init (), file.%c2.var [concrete = constants.%C2.val]
// CHECK:STDOUT:   %.loc7_1: init %C2 = converted %.loc7_15.1, %.loc7_15.2 [concrete = constants.%C2.val]
// CHECK:STDOUT:   assign file.%c2.var, %.loc7_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- dup_c1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C1 = import_ref Main//c1, C1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C1 = imports.%Main.C1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- dup_c1.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C1: type = class_type @C1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C1.val: %C1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C1: type = import_ref Main//dup_c1, C1, loaded [concrete = constants.%C1]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//c1, loc4_11, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.eb7 = import_ref Main//c1, inst14 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C1 = imports.%Main.C1
// CHECK:STDOUT:     .c1 = %c1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_22.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_22.2 = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %C1 = binding_pattern c1
// CHECK:STDOUT:     %.loc6: %C1 = var_pattern %c1.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.var: ref %C1 = var c1
// CHECK:STDOUT:   %C1.ref: type = name_ref C1, imports.%Main.C1 [concrete = constants.%C1]
// CHECK:STDOUT:   %c1: ref %C1 = bind_name c1, %c1.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1 [from "c1.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.eb7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc6_15.2: init %C1 = class_init (), file.%c1.var [concrete = constants.%C1.val]
// CHECK:STDOUT:   %.loc6_1: init %C1 = converted %.loc6_15.1, %.loc6_15.2 [concrete = constants.%C1.val]
// CHECK:STDOUT:   assign file.%c1.var, %.loc6_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_ns.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//ns, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .C = %Main.C
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_ns.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//use_ns, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .C = %Main.C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.C: type = import_ref Main//use_ns, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//ns, loc5_13, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2dd = import_ref Main//ns, inst15 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_22.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_22.2 = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc4_1: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %.loc4_10: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:     %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "ns.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2dd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_16.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc4_16.2: init %C = class_init (), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc4_1: init %C = converted %.loc4_16.1, %.loc4_16.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc4_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- mix_other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     import Other//o1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- mix_other.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %O1: type = class_type @O1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %O1.val: %O1 = struct_value () [concrete]
// CHECK:STDOUT:   %O2: type = class_type @O2 [concrete]
// CHECK:STDOUT:   %O2.val: %O2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .O1 = %Other.O1
// CHECK:STDOUT:     .O2 = %Other.O2
// CHECK:STDOUT:     import Other//o2
// CHECK:STDOUT:     import Other//o1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.O1: type = import_ref Other//o1, O1, loaded [concrete = constants.%O1]
// CHECK:STDOUT:   %Other.import_ref.8f24d3.1: <witness> = import_ref Other//o1, loc4_11, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.481 = import_ref Other//o1, inst14 [no loc], unloaded
// CHECK:STDOUT:   %Other.O2: type = import_ref Other//o2, O2, loaded [concrete = constants.%O2]
// CHECK:STDOUT:   %Other.import_ref.8f24d3.2: <witness> = import_ref Other//o2, loc4_11, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.2eb = import_ref Other//o2, inst14 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .o1 = %o1
// CHECK:STDOUT:     .o2 = %o2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_25.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_25.2 = import <none>
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %o1.patt: %O1 = binding_pattern o1
// CHECK:STDOUT:     %.loc6_1: %O1 = var_pattern %o1.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o1.var: ref %O1 = var o1
// CHECK:STDOUT:   %.loc6_14: type = splice_block %O1.ref [concrete = constants.%O1] {
// CHECK:STDOUT:     %Other.ref.loc6: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:     %O1.ref: type = name_ref O1, imports.%Other.O1 [concrete = constants.%O1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o1: ref %O1 = bind_name o1, %o1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %o2.patt: %O2 = binding_pattern o2
// CHECK:STDOUT:     %.loc7_1: %O2 = var_pattern %o2.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o2.var: ref %O2 = var o2
// CHECK:STDOUT:   %.loc7_14: type = splice_block %O2.ref [concrete = constants.%O2] {
// CHECK:STDOUT:     %Other.ref.loc7: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:     %O2.ref: type = name_ref O2, imports.%Other.O2 [concrete = constants.%O2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o2: ref %O2 = bind_name o2, %o2.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @O1 [from "o1.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.481
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @O2 [from "o2.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.2eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc6_21.2: init %O1 = class_init (), file.%o1.var [concrete = constants.%O1.val]
// CHECK:STDOUT:   %.loc6_1: init %O1 = converted %.loc6_21.1, %.loc6_21.2 [concrete = constants.%O1.val]
// CHECK:STDOUT:   assign file.%o1.var, %.loc6_1
// CHECK:STDOUT:   %.loc7_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_21.2: init %O2 = class_init (), file.%o2.var [concrete = constants.%O2.val]
// CHECK:STDOUT:   %.loc7_1: init %O2 = converted %.loc7_21.1, %.loc7_21.2 [concrete = constants.%O2.val]
// CHECK:STDOUT:   assign file.%o2.var, %.loc7_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- dup_o1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     import Other//o1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- dup_o1.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %O1: type = class_type @O1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %O1.val: %O1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .O1 = %Other.O1
// CHECK:STDOUT:     import Other//o1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.O1: type = import_ref Other//o1, O1, loaded [concrete = constants.%O1]
// CHECK:STDOUT:   %Other.import_ref.8f2: <witness> = import_ref Other//o1, loc4_11, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.481 = import_ref Other//o1, inst14 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .o1 = %o1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_22.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_22.2 = import <none>
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %o1.patt: %O1 = binding_pattern o1
// CHECK:STDOUT:     %.loc6_1: %O1 = var_pattern %o1.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o1.var: ref %O1 = var o1
// CHECK:STDOUT:   %.loc6_14: type = splice_block %O1.ref [concrete = constants.%O1] {
// CHECK:STDOUT:     %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:     %O1.ref: type = name_ref O1, imports.%Other.O1 [concrete = constants.%O1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o1: ref %O1 = bind_name o1, %o1.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @O1 [from "o1.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.481
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc6_21.2: init %O1 = class_init (), file.%o1.var [concrete = constants.%O1.val]
// CHECK:STDOUT:   %.loc6_1: init %O1 = converted %.loc6_21.1, %.loc6_21.2 [concrete = constants.%O1.val]
// CHECK:STDOUT:   assign file.%o1.var, %.loc6_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     import Other//o1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_conflict.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     import Other//o1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc11_31.1 = import <none>
// CHECK:STDOUT:   %default.import.loc11_31.2 = import <none>
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_conflict_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Other = import_ref Main//local_other, Other, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Main.Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_conflict_reverse.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     import Other//o1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_39.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_39.2 = import <none>
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT: }
// CHECK:STDOUT:
