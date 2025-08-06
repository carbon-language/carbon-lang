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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/import_struct_cyle.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/import_struct_cyle.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

class Cycle;

var a: {.b: Cycle*};

class Cycle {
  // The type here is equivalent to the `a` above, but on import can be resolved first.
  var c: {.b: Cycle*};
}

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

fn Run() {
  a.b = (*a.b).c.b;
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [concrete]
// CHECK:STDOUT:   %ptr.257: type = ptr_type %Cycle [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %ptr.257} [concrete]
// CHECK:STDOUT:   %pattern_type.afd: type = pattern_type %struct_type.b [concrete]
// CHECK:STDOUT:   %Cycle.elem: type = unbound_element_type %Cycle, %struct_type.b [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Cycle.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.d3d: type = pattern_type %ptr.257 [concrete]
// CHECK:STDOUT:   %Cycle.as.Destroy.impl.Op.type: type = fn_type @Cycle.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Cycle.as.Destroy.impl.Op: %Cycle.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %struct_type.b} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.c [concrete]
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
// CHECK:STDOUT:     .Cycle = %Cycle.decl.loc4
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cycle.decl.loc4: type = class_decl @Cycle [concrete = constants.%Cycle] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.afd = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.afd = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %struct_type.b = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc6: type = splice_block %struct_type.b [concrete = constants.%struct_type.b] {
// CHECK:STDOUT:     %Cycle.ref: type = name_ref Cycle, %Cycle.decl.loc4 [concrete = constants.%Cycle]
// CHECK:STDOUT:     %ptr: type = ptr_type %Cycle.ref [concrete = constants.%ptr.257]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %ptr.257} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %struct_type.b = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   %Cycle.decl.loc8: type = class_decl @Cycle [concrete = constants.%Cycle] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Cycle.as.Destroy.impl: constants.%Cycle as constants.%Destroy.type {
// CHECK:STDOUT:   %Cycle.as.Destroy.impl.Op.decl: %Cycle.as.Destroy.impl.Op.type = fn_decl @Cycle.as.Destroy.impl.Op [concrete = constants.%Cycle.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.d3d = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d3d = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
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
// CHECK:STDOUT:   %Cycle.ref: type = name_ref Cycle, file.%Cycle.decl.loc4 [concrete = constants.%Cycle]
// CHECK:STDOUT:   %ptr: type = ptr_type %Cycle.ref [concrete = constants.%ptr.257]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %ptr.257} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:   %.loc10: %Cycle.elem = field_decl c, element0 [concrete]
// CHECK:STDOUT:   impl_decl @Cycle.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Cycle.as.Destroy.impl.%Cycle.as.Destroy.impl.Op.decl), @Cycle.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %struct_type.b} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.c [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Cycle
// CHECK:STDOUT:   .Cycle = <poisoned>
// CHECK:STDOUT:   .c = %.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Cycle.as.Destroy.impl.Op(%self.param: %ptr.257) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [concrete]
// CHECK:STDOUT:   %ptr.257: type = ptr_type %Cycle [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %ptr.257} [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %struct_type.b} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.c [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %struct_type.b [concrete]
// CHECK:STDOUT:   %.46a: ref %ptr.257 = struct_access imports.%a.var, element0 [concrete]
// CHECK:STDOUT:   %Cycle.elem: type = unbound_element_type %Cycle, %struct_type.b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Cycle = import_ref Main//a, Cycle, unloaded
// CHECK:STDOUT:   %Main.a: ref %struct_type.b = import_ref Main//a, a, loaded [concrete = %a.var]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.b93: <witness> = import_ref Main//a, loc11_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.3a6 = import_ref Main//a, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.455: %Cycle.elem = import_ref Main//a, loc10_8, loaded [concrete = %.354]
// CHECK:STDOUT:   %a.patt: %pattern_type = binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %a.var: ref %struct_type.b = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.354: %Cycle.elem = field_decl c, element0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cycle = imports.%Main.Cycle
// CHECK:STDOUT:     .a = imports.%Main.a
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.b93
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.3a6
// CHECK:STDOUT:   .c = imports.%Main.import_ref.455
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref.loc7_3: ref %struct_type.b = name_ref a, imports.%Main.a [concrete = imports.%a.var]
// CHECK:STDOUT:   %.loc7_4: ref %ptr.257 = struct_access %a.ref.loc7_3, element0 [concrete = constants.%.46a]
// CHECK:STDOUT:   %a.ref.loc7_11: ref %struct_type.b = name_ref a, imports.%Main.a [concrete = imports.%a.var]
// CHECK:STDOUT:   %.loc7_12.1: ref %ptr.257 = struct_access %a.ref.loc7_11, element0 [concrete = constants.%.46a]
// CHECK:STDOUT:   %.loc7_12.2: %ptr.257 = bind_value %.loc7_12.1
// CHECK:STDOUT:   %.loc7_10: ref %Cycle = deref %.loc7_12.2
// CHECK:STDOUT:   %c.ref: %Cycle.elem = name_ref c, imports.%Main.import_ref.455 [concrete = imports.%.354]
// CHECK:STDOUT:   %.loc7_15: ref %struct_type.b = class_element_access %.loc7_10, element0
// CHECK:STDOUT:   %.loc7_17.1: ref %ptr.257 = struct_access %.loc7_15, element0
// CHECK:STDOUT:   %.loc7_17.2: %ptr.257 = bind_value %.loc7_17.1
// CHECK:STDOUT:   assign %.loc7_4, %.loc7_17.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
