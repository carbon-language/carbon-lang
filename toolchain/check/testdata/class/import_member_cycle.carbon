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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/import_member_cycle.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/import_member_cycle.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

class Cycle {
  var a: Cycle*;
}

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

fn Run() {
  var a: Cycle*;
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [concrete]
// CHECK:STDOUT:   %ptr.257: type = ptr_type %Cycle [concrete]
// CHECK:STDOUT:   %Cycle.elem: type = unbound_element_type %Cycle, %ptr.257 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Cycle.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.d3d: type = pattern_type %ptr.257 [concrete]
// CHECK:STDOUT:   %Cycle.as.Destroy.impl.Op.type: type = fn_type @Cycle.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Cycle.as.Destroy.impl.Op: %Cycle.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %ptr.257} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete]
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
// CHECK:STDOUT:     .Cycle = %Cycle.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cycle.decl: type = class_decl @Cycle [concrete = constants.%Cycle] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Cycle.as.Destroy.impl: @Cycle.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Cycle.as.Destroy.impl.Op.decl: %Cycle.as.Destroy.impl.Op.type = fn_decl @Cycle.as.Destroy.impl.Op [concrete = constants.%Cycle.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.d3d = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d3d = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.257 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Cycle [concrete = constants.%Cycle]
// CHECK:STDOUT:     %self: %ptr.257 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Cycle.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Cycle.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle {
// CHECK:STDOUT:   %Cycle.ref: type = name_ref Cycle, file.%Cycle.decl [concrete = constants.%Cycle]
// CHECK:STDOUT:   %ptr: type = ptr_type %Cycle.ref [concrete = constants.%ptr.257]
// CHECK:STDOUT:   %.loc5: %Cycle.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Cycle [concrete = constants.%Cycle]
// CHECK:STDOUT:   impl_decl @Cycle.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Cycle.as.Destroy.impl.%Cycle.as.Destroy.impl.Op.decl), @Cycle.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Cycle
// CHECK:STDOUT:   .Cycle = <poisoned>
// CHECK:STDOUT:   .a = %.loc5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Cycle.as.Destroy.impl.Op(%self.param: %ptr.257) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [concrete]
// CHECK:STDOUT:   %ptr.257: type = ptr_type %Cycle [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %ptr.257} [concrete]
// CHECK:STDOUT:   %complete_type.e4b: <witness> = complete_type_witness %struct_type.a [concrete]
// CHECK:STDOUT:   %pattern_type.d3d: type = pattern_type %ptr.257 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.type.976: type = fn_type @ptr.as.Destroy.impl.Op, @ptr.as.Destroy.impl(%Cycle) [concrete]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.594: %ptr.as.Destroy.impl.Op.type.976 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d80: type = ptr_type %ptr.257 [concrete]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Destroy.impl.Op.594, @ptr.as.Destroy.impl.Op(%Cycle) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Cycle: type = import_ref Main//a, Cycle, loaded [concrete = constants.%Cycle]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.72d: <witness> = import_ref Main//a, loc6_1, loaded [concrete = constants.%complete_type.e4b]
// CHECK:STDOUT:   %Main.import_ref.3a6 = import_ref Main//a, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.4e0 = import_ref Main//a, loc5_8, unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.13a = import_ref Main//a, loc4_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.f64: type = import_ref Main//a, loc4_13, loaded [concrete = constants.%Cycle]
// CHECK:STDOUT:   %Main.import_ref.e45: type = import_ref Main//a, inst26 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cycle = imports.%Main.Cycle
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Cycle.as.Destroy.impl: imports.%Main.import_ref.f64 as imports.%Main.import_ref.e45 [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.13a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.72d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.3a6
// CHECK:STDOUT:   .a = imports.%Main.import_ref.4e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d3d = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d3d = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %ptr.257 = var %a.var_patt
// CHECK:STDOUT:   %.loc7: type = splice_block %ptr [concrete = constants.%ptr.257] {
// CHECK:STDOUT:     %Cycle.ref: type = name_ref Cycle, imports.%Main.Cycle [concrete = constants.%Cycle]
// CHECK:STDOUT:     %ptr: type = ptr_type %Cycle.ref [concrete = constants.%ptr.257]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %ptr.257 = bind_name a, %a.var
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%ptr.as.Destroy.impl.Op.594
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%ptr.as.Destroy.impl.Op.594, @ptr.as.Destroy.impl.Op(constants.%Cycle) [concrete = constants.%ptr.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.var, %ptr.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.d80 = addr_of %a.var
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
