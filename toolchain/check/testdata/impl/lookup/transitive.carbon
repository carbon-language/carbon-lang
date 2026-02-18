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
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F: %I.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%I.WithSelf.F.decl [concrete]
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
// CHECK:STDOUT:   %Self.loc4_13: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.%I.WithSelf.F.type (%I.WithSelf.F.type) = fn_decl @I.WithSelf.F [symbolic = @I.%I.WithSelf.F (constants.%I.WithSelf.F)] {
// CHECK:STDOUT:     %self.patt: @I.WithSelf.F.%pattern_type (%pattern_type) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @I.WithSelf.F.%pattern_type (%pattern_type) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @I.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_26.1: type = splice_block %.loc4_26.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self.loc4_13 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc4_26.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc4_13
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self.loc4_13: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.a1f: type = fn_type @I.WithSelf.F, @I(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.cc3: %I.WithSelf.F.type.a1f = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.422: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.31f: type = fn_type @I.WithSelf.F, @I(%I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.564: %I.WithSelf.F.type.31f = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//i, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.7f4 = import_ref Main//i, loc4_33, unloaded
// CHECK:STDOUT:   %Main.F: @I.%I.WithSelf.F.type (%I.WithSelf.F.type.a1f) = import_ref Main//i, F, loaded [symbolic = @I.%I.WithSelf.F (constants.%I.WithSelf.F.cc3)]
// CHECK:STDOUT:   %Main.import_ref.2362f8.2: %I.type = import_ref Main//i, loc4_13, loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Main.import_ref.c82 = import_ref Main//i, loc4_13, unloaded
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "i.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c82
// CHECK:STDOUT:   .F = imports.%Main.import_ref.7f4
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %I.ref {
// CHECK:STDOUT:   %C.as.I.impl.F.decl: %C.as.I.impl.F.type = fn_decl @C.as.I.impl.F [concrete = constants.%C.as.I.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.7c7 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7c7 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.I.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%C.as.I.impl.F.decl), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.I.impl.F.decl
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.2362f8.2: %I.type) [from "i.carbon"] {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.422)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.F(%self.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.a1f
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.cc3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.422
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.31f
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.564
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7c7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- get.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.768: Core.Form = init_form %C, call_param0 [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.743 = import_ref Main//c, inst{{[0-9A-F]+}} [no loc], unloaded
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
// CHECK:STDOUT:     %.loc6: Core.Form = init_form %C.ref, call_param0 [concrete = constants.%.768]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "c.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.743
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Get() -> out %return.param: %C;
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
// CHECK:STDOUT:   %Self.651: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.a1f: type = fn_type @I.WithSelf.F, @I(%Self.651) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.cc3: %I.WithSelf.F.type.a1f = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.3af: type = symbolic_binding_type Self, 0, %Self.651 [symbolic]
// CHECK:STDOUT:   %pattern_type.422: type = pattern_type %Self.binding.as_type.3af [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.a42: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.972 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.442: type = fn_type @I.WithSelf.F, @I(%I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.26e: %I.WithSelf.F.type.442 = struct_value () [concrete]
// CHECK:STDOUT:   %.c4c: type = fn_type_with_self_type %I.WithSelf.F.type.442, %I.facet [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Get: %Get.type = import_ref Main//get, Get, loaded [concrete = constants.%Get]
// CHECK:STDOUT:   %Main.I: type = import_ref Main//i, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8db: <witness> = import_ref Main//get, inst{{[0-9A-F]+}} [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.a60 = import_ref Main//get, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.a54: %I.assoc_type = import_ref Main//i, loc4_33, loaded [concrete = constants.%assoc0.a42]
// CHECK:STDOUT:   %Main.F = import_ref Main//i, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.2362f8.2: %I.type = import_ref Main//i, loc4_13, loaded [symbolic = constants.%Self.651]
// CHECK:STDOUT:   %Main.import_ref.c82 = import_ref Main//i, loc4_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.972: @I.%I.WithSelf.F.type (%I.WithSelf.F.type.a1f) = import_ref Main//i, loc4_33, loaded [symbolic = @I.%I.WithSelf.F (constants.%I.WithSelf.F.cc3)]
// CHECK:STDOUT:   %Main.import_ref.31c: <witness> = import_ref Main//c, loc7_13, loaded [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.61c: type = import_ref Main//c, loc7_6, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.91f: type = import_ref Main//c, loc7_11, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.706: %C.as.I.impl.F.type = import_ref Main//c, loc7_34, loaded [concrete = constants.%C.as.I.impl.F]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.706), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Get = imports.%Main.Get
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "i.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c82
// CHECK:STDOUT:   .F = imports.%Main.import_ref.a54
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: imports.%Main.import_ref.61c as imports.%Main.import_ref.91f [from "c.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.31c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "get.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8db
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.a60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Get.ref: %Get.type = name_ref Get, imports.%Main.Get [concrete = constants.%Get]
// CHECK:STDOUT:   %.loc9_7.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Get.call: init %C to %.loc9_7.1 = call %Get.ref()
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %I.assoc_type = name_ref F, imports.%Main.import_ref.a54 [concrete = constants.%assoc0.a42]
// CHECK:STDOUT:   %impl.elem0: %.c4c = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%C.as.I.impl.F]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %Get.call, %impl.elem0
// CHECK:STDOUT:   %.loc9_7.2: ref %C = temporary %.loc9_7.1, %Get.call
// CHECK:STDOUT:   %.loc9_7.3: %C = acquire_value %.loc9_7.2
// CHECK:STDOUT:   %C.as.I.impl.F.call: init %empty_tuple.type = call %bound_method(%.loc9_7.3)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc9_7.2, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc9_7.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Get [from "get.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.2362f8.2: %I.type) [from "i.carbon"] {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.651)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.3af)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.422)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.F [from "c.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%Self.651) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.651
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.a1f
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.cc3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.651) {
// CHECK:STDOUT:   %Self => constants.%Self.651
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.3af
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.422
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.442
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.26e
// CHECK:STDOUT: }
// CHECK:STDOUT:
