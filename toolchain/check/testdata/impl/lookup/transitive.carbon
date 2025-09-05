// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/transitive.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/transitive.carbon

// --- i.carbon

library "[[@TEST_NAME]]";

interface I { fn F[self: Self](); }

// --- c.carbon

library "[[@TEST_NAME]]";

import library "i";

class C {}
impl C as I { fn F[self: Self]() {} }

// --- get.carbon

library "[[@TEST_NAME]]";

import library "c";

fn Get() -> C;

// --- call.carbon

library "[[@TEST_NAME]]";

import library "get";
import library "i";

fn Call() {
  // We find C in "get", but should look in "c" instead.
  Get().(I.F)();
}

// CHECK:STDOUT: --- i.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%I.F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.F.decl: %I.F.type = fn_decl @I.F [concrete = constants.%I.F] {
// CHECK:STDOUT:     %self.patt: @I.F.%pattern_type (%pattern_type) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @I.F.%pattern_type (%pattern_type) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @I.F.%Self.as_type.loc4_26.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_26.1: type = splice_block %.loc4_26.2 [symbolic = %Self.as_type.loc4_26.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc4_26.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_26.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc4_26.2: type = converted %Self.ref, %Self.as_type.loc4_26.2 [symbolic = %Self.as_type.loc4_26.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.F.%Self.as_type.loc4_26.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%I.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_26.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_26.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc4_26.1 [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.F.%Self.as_type.loc4_26.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_26.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.carbon
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
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %Self.as_type.b70 [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//i, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//i, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.507 = import_ref Main//i, loc4_33, unloaded
// CHECK:STDOUT:   %Main.F: %I.F.type = import_ref Main//i, F, loaded [concrete = constants.%I.F]
// CHECK:STDOUT:   %Main.import_ref.5dd: %I.type = import_ref Main//i, inst19 [no loc], loaded [symbolic = constants.%Self.826]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@C.as.I.impl.%C.as.I.impl.F.decl), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "i.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .F = imports.%Main.import_ref.507
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc6: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
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
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %I.ref {
// CHECK:STDOUT:   %C.as.I.impl.F.decl: %C.as.I.impl.F.type = fn_decl @C.as.I.impl.F [concrete = constants.%C.as.I.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.I.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.I.impl.F.decl
// CHECK:STDOUT:   witness = file.%I.impl_witness
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
// CHECK:STDOUT: generic fn @I.F(imports.%Main.import_ref.5dd: %I.type) [from "i.carbon"] {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.F(%self.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.b70
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- get.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Get.type: type = fn_type @Get [concrete]
// CHECK:STDOUT:   %Get: %Get.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//c, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//c, loc6_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//c, inst20 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Get = %Get.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Get.decl: %Get.type = fn_decl @Get [concrete = constants.%Get] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "c.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Get() -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %Get.type: type = fn_type @Get [concrete]
// CHECK:STDOUT:   %Get: %Get.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.3f3: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.e03 [concrete]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %Self.as_type.b70 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %.1b3: type = fn_type_with_self_type %I.F.type, %I.facet [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.0cc: <witness> = impl_witness imports.%Destroy.impl_witness_table.170 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Get: %Get.type = import_ref Main//get, Get, loaded [concrete = constants.%Get]
// CHECK:STDOUT:   %Main.I: type = import_ref Main//i, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8db: <witness> = import_ref Main//get, inst23 [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.6a9 = import_ref Main//get, inst24 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//i, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c44: %I.assoc_type = import_ref Main//i, loc4_33, loaded [concrete = constants.%assoc0.3f3]
// CHECK:STDOUT:   %Main.F = import_ref Main//i, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.e03: %I.F.type = import_ref Main//i, loc4_33, loaded [concrete = constants.%I.F]
// CHECK:STDOUT:   %Main.import_ref.5dd: %I.type = import_ref Main//i, inst19 [no loc], loaded [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %Main.import_ref.4c7: <witness> = import_ref Main//c, loc6_9, loaded [concrete = constants.%Destroy.impl_witness.0cc]
// CHECK:STDOUT:   %Main.import_ref.0ed: type = import_ref Main//c, loc6_9, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.cb9: type = import_ref Main//c, inst23 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.f0d: <witness> = import_ref Main//c, loc7_13, loaded [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.29a: type = import_ref Main//c, loc7_6, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.f50: type = import_ref Main//c, loc7_11, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.742: %C.as.I.impl.F.type = import_ref Main//c, loc7_34, loaded [concrete = constants.%C.as.I.impl.F]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.742), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.2e8: %C.as.Destroy.impl.Op.type = import_ref Main//c, loc6_9, loaded [concrete = constants.%C.as.Destroy.impl.Op]
// CHECK:STDOUT:   %Destroy.impl_witness_table.170 = impl_witness_table (%Main.import_ref.2e8), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Get = imports.%Main.Get
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "i.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .F = imports.%Main.import_ref.c44
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: imports.%Main.import_ref.0ed as imports.%Main.import_ref.cb9 [from "c.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.4c7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: imports.%Main.import_ref.29a as imports.%Main.import_ref.f50 [from "c.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.f0d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "get.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8db
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.6a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Get.ref: %Get.type = name_ref Get, imports.%Main.Get [concrete = constants.%Get]
// CHECK:STDOUT:   %.loc9_7.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Get.call: init %C = call %Get.ref() to %.loc9_7.1
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %I.assoc_type = name_ref F, imports.%Main.import_ref.c44 [concrete = constants.%assoc0.3f3]
// CHECK:STDOUT:   %impl.elem0: %.1b3 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%C.as.I.impl.F]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %Get.call, %impl.elem0
// CHECK:STDOUT:   %.loc9_7.2: ref %C = temporary %.loc9_7.1, %Get.call
// CHECK:STDOUT:   %.loc9_7.3: %C = bind_value %.loc9_7.2
// CHECK:STDOUT:   %C.as.I.impl.F.call: init %empty_tuple.type = call %bound_method(%.loc9_7.3)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc9_7.2, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc9_7.2
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Get [from "get.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(imports.%Main.import_ref.5dd: %I.type) [from "i.carbon"] {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.F [from "c.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op = "no_op" [from "c.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.b70
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
