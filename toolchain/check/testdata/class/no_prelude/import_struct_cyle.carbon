// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/import_struct_cyle.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/import_struct_cyle.carbon

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
  //@dump-sem-ir-begin
  a.b = (*a.b).c.b;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [concrete]
// CHECK:STDOUT:   %ptr.257: type = ptr_type %Cycle [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %ptr.257} [concrete]
// CHECK:STDOUT:   %Cycle.elem: type = unbound_element_type %Cycle, %struct_type.b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.a: ref %struct_type.b = import_ref Main//a, a, loaded
// CHECK:STDOUT:   %Main.import_ref.455: %Cycle.elem = import_ref Main//a, loc10_8, loaded [concrete = %.354]
// CHECK:STDOUT:   %.354: %Cycle.elem = field_decl c, element0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref.loc8_3: ref %struct_type.b = name_ref a, imports.%Main.a
// CHECK:STDOUT:   %.loc8_4: ref %ptr.257 = struct_access %a.ref.loc8_3, element0
// CHECK:STDOUT:   %a.ref.loc8_11: ref %struct_type.b = name_ref a, imports.%Main.a
// CHECK:STDOUT:   %.loc8_12.1: ref %ptr.257 = struct_access %a.ref.loc8_11, element0
// CHECK:STDOUT:   %.loc8_12.2: %ptr.257 = bind_value %.loc8_12.1
// CHECK:STDOUT:   %.loc8_10: ref %Cycle = deref %.loc8_12.2
// CHECK:STDOUT:   %c.ref: %Cycle.elem = name_ref c, imports.%Main.import_ref.455 [concrete = imports.%.354]
// CHECK:STDOUT:   %.loc8_15: ref %struct_type.b = class_element_access %.loc8_10, element0
// CHECK:STDOUT:   %.loc8_17.1: ref %ptr.257 = struct_access %.loc8_15, element0
// CHECK:STDOUT:   %.loc8_17.2: %ptr.257 = bind_value %.loc8_17.1
// CHECK:STDOUT:   assign %.loc8_4, %.loc8_17.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
