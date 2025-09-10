// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/export_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/export_name.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- base.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

// --- export.carbon

library "[[@TEST_NAME]]";

import library "base";

export D;

// --- export_orig.carbon

library "[[@TEST_NAME]]";

import library "base";

export C;

// ============================================================================
// Test files
// ============================================================================

// --- use_export.carbon

library "[[@TEST_NAME]]";

import library "export";

var d: D = {};

// --- fail_orig_name_not_in_export.carbon

library "[[@TEST_NAME]]";

import library "export";

// CHECK:STDERR: fail_orig_name_not_in_export.carbon:[[@LINE+4]]:8: error: name `C` not found [NameNotFound]
// CHECK:STDERR: var c: C = {};
// CHECK:STDERR:        ^
// CHECK:STDERR:
var c: C = {};

// --- indirect_compat.carbon

library "[[@TEST_NAME]]";

import library "export";
import library "export_orig";

var c: C = {};
var d: D* = &c;

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C = import_ref Main//base, C, unloaded
// CHECK:STDOUT:   %Main.D: type = import_ref Main//base, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//base, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//base, inst18 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %D: type = export D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_orig.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//base, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.D = import_ref Main//base, D, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//base, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//base, inst18 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C: type = export C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//export, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8db: <witness> = import_ref Main//export, inst24 [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.6a9 = import_ref Main//export, inst25 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %C = var %d.var_patt [concrete]
// CHECK:STDOUT:   %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:   %d: ref %C = bind_name d, %d.var [concrete = %d.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8db
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.6a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc6_13.2: init %C = class_init (), file.%d.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_1: init %C = converted %.loc6_13.1, %.loc6_13.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%d.var, %.loc6_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_orig_name_not_in_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D = import_ref Main//export, D, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: <error> = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: <error> = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref <error> = var %c.var_patt [concrete = <error>]
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [concrete = <error>]
// CHECK:STDOUT:   %c: <error> = bind_name c, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- indirect_compat.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of file.%c.var [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.f23: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.abf: %ptr.as.Copy.impl.Op.type.f23 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a94: <witness> = impl_witness imports.%Copy.impl_witness_table.a71, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.0f5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.fd1: %ptr.as.Copy.impl.Op.type.0f5 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.9a6: %Copy.type = facet_value %ptr.019, (%Copy.impl_witness.a94) [concrete]
// CHECK:STDOUT:   %.a98: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.9a6 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.fd1 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.fd1, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//export, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//export_orig, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8db: <witness> = import_ref Main//export_orig, inst24 [indirect], loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.6a9 = import_ref Main//export_orig, inst25 [indirect], unloaded
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.de9: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.f23) = import_ref Core//prelude/parts/copy, loc32_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.abf)]
// CHECK:STDOUT:   %Copy.impl_witness_table.a71 = impl_witness_table (%Core.import_ref.de9), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.c48 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt [concrete]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.44a = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.44a = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %ptr.019 = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc8: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     %ptr: type = ptr_type %D.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %ptr.019 = bind_name d, %d.var [concrete = %d.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export_orig.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8db
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.6a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_13.2: init %C = class_init (), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_13.1, %.loc7_13.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_1
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, file.%c [concrete = file.%c.var]
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %c.ref [concrete = constants.%addr]
// CHECK:STDOUT:   %impl.elem0: %.a98 = impl_witness_access constants.%Copy.impl_witness.a94, element0 [concrete = constants.%ptr.as.Copy.impl.Op.fd1]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %addr, %impl.elem0 [concrete = constants.%ptr.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%C) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %addr, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.019 = call %bound_method.loc8_13.2(%addr) [concrete = constants.%addr]
// CHECK:STDOUT:   assign file.%d.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
