// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %ptr: type = ptr_type %Cycle [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %ptr} [concrete]
// CHECK:STDOUT:   %Cycle.elem: type = unbound_element_type %Cycle, %struct_type.b [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %struct_type.b} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.c [concrete]
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
// CHECK:STDOUT:     .Cycle = %Cycle.decl.loc4
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cycle.decl.loc4: type = class_decl @Cycle [concrete = constants.%Cycle] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %struct_type.b = binding_pattern a
// CHECK:STDOUT:     %.loc6_1: %struct_type.b = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %struct_type.b = var a
// CHECK:STDOUT:   %.loc6_19: type = splice_block %struct_type.b [concrete = constants.%struct_type.b] {
// CHECK:STDOUT:     %Cycle.ref: type = name_ref Cycle, %Cycle.decl.loc4 [concrete = constants.%Cycle]
// CHECK:STDOUT:     %ptr: type = ptr_type %Cycle [concrete = constants.%ptr]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %ptr} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %struct_type.b = bind_name a, %a.var
// CHECK:STDOUT:   %Cycle.decl.loc8: type = class_decl @Cycle [concrete = constants.%Cycle] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle {
// CHECK:STDOUT:   %.loc10_8: %Cycle.elem = field_decl c, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc10_3: %Cycle.elem = var_pattern %.loc10_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %Cycle.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.c [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Cycle
// CHECK:STDOUT:   .Cycle = <poisoned>
// CHECK:STDOUT:   .c = %.loc10_8
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %Cycle.elem: type = unbound_element_type %Cycle, %struct_type.b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Cycle = import_ref Main//a, Cycle, unloaded
// CHECK:STDOUT:   %Main.a: ref %struct_type.b = import_ref Main//a, a, loaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.b93: <witness> = import_ref Main//a, loc11_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.3a6 = import_ref Main//a, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.455: %Cycle.elem = import_ref Main//a, loc10_8, loaded [concrete = %.354]
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
// CHECK:STDOUT:   %a.ref.loc7_3: ref %struct_type.b = name_ref a, imports.%Main.a
// CHECK:STDOUT:   %.loc7_4: ref %ptr.257 = struct_access %a.ref.loc7_3, element0
// CHECK:STDOUT:   %a.ref.loc7_11: ref %struct_type.b = name_ref a, imports.%Main.a
// CHECK:STDOUT:   %.loc7_12.1: ref %ptr.257 = struct_access %a.ref.loc7_11, element0
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
