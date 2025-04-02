// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %ptr: type = ptr_type %Cycle [concrete]
// CHECK:STDOUT:   %Cycle.elem: type = unbound_element_type %Cycle, %ptr [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %ptr} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete]
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
// CHECK:STDOUT:     .Cycle = %Cycle.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cycle.decl: type = class_decl @Cycle [concrete = constants.%Cycle] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle {
// CHECK:STDOUT:   %.loc5_8: %Cycle.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %Cycle.elem = var_pattern %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %Cycle.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Cycle
// CHECK:STDOUT:   .Cycle = <poisoned>
// CHECK:STDOUT:   .a = %.loc5_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Cycle [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %ptr} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Cycle: type = import_ref Main//a, Cycle, loaded [concrete = constants.%Cycle]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.72d: <witness> = import_ref Main//a, loc6_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.3a6 = import_ref Main//a, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.4e0 = import_ref Main//a, loc5_8, unloaded
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
// CHECK:STDOUT:     %a.patt: %ptr = binding_pattern a
// CHECK:STDOUT:     %.loc7_3: %ptr = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %ptr = var a
// CHECK:STDOUT:   %.loc7_15: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %Cycle.ref: type = name_ref Cycle, imports.%Main.Cycle [concrete = constants.%Cycle]
// CHECK:STDOUT:     %ptr: type = ptr_type %Cycle.ref [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %ptr = bind_name a, %a.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
