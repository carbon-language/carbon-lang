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
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [template]
// CHECK:STDOUT:   %.1: type = ptr_type %Cycle [template]
// CHECK:STDOUT:   %.2: type = struct_type {.b: %.1} [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %Cycle, %.2 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.c: %.2} [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [template]
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
// CHECK:STDOUT:     .Cycle = %Cycle.decl.loc4
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cycle.decl.loc4: type = class_decl @Cycle [template = constants.%Cycle] {} {}
// CHECK:STDOUT:   %Cycle.ref: type = name_ref Cycle, %Cycle.decl.loc4 [template = constants.%Cycle]
// CHECK:STDOUT:   %.loc6_18: type = ptr_type %Cycle [template = constants.%.1]
// CHECK:STDOUT:   %.loc6_19: type = struct_type {.b: %.1} [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref %.2 = var a
// CHECK:STDOUT:   %a: ref %.2 = bind_name a, %a.var
// CHECK:STDOUT:   %Cycle.decl.loc8: type = class_decl @Cycle [template = constants.%Cycle] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle {
// CHECK:STDOUT:   %Cycle.ref: type = name_ref Cycle, file.%Cycle.decl.loc4 [template = constants.%Cycle]
// CHECK:STDOUT:   %.loc10_20: type = ptr_type %Cycle [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_21: type = struct_type {.b: %.1} [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_8: %.3 = field_decl c, element0 [template]
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Cycle
// CHECK:STDOUT:   .c = %.loc10_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [template]
// CHECK:STDOUT:   %.1: type = ptr_type %Cycle [template]
// CHECK:STDOUT:   %.2: type = struct_type {.b: %.1} [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %Cycle, %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//a, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.2: ref %.2 = import_ref Main//a, inst+11, loaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//a, inst+4, unloaded
// CHECK:STDOUT:   %import_ref.4: %.6 = import_ref Main//a, inst+17, loaded [template = %.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Cycle = imports.%import_ref.1
// CHECK:STDOUT:     .a = imports.%import_ref.2
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .c = imports.%import_ref.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref.loc7_3: ref %.2 = name_ref a, imports.%import_ref.2
// CHECK:STDOUT:   %.loc7_4: ref %.1 = struct_access %a.ref.loc7_3, element0
// CHECK:STDOUT:   %a.ref.loc7_11: ref %.2 = name_ref a, imports.%import_ref.2
// CHECK:STDOUT:   %.loc7_12.1: ref %.1 = struct_access %a.ref.loc7_11, element0
// CHECK:STDOUT:   %.loc7_12.2: %.1 = bind_value %.loc7_12.1
// CHECK:STDOUT:   %.loc7_10: ref %Cycle = deref %.loc7_12.2
// CHECK:STDOUT:   %c.ref: %.6 = name_ref c, imports.%import_ref.4 [template = imports.%.1]
// CHECK:STDOUT:   %.loc7_15: ref %.2 = class_element_access %.loc7_10, element0
// CHECK:STDOUT:   %.loc7_17.1: ref %.1 = struct_access %.loc7_15, element0
// CHECK:STDOUT:   %.loc7_17.2: %.1 = bind_value %.loc7_17.1
// CHECK:STDOUT:   assign %.loc7_4, %.loc7_17.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
