// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/val_form.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/val_form.carbon

// --- param.carbon
library "[[@TEST_NAME]]";

class C { var a: i32; }

//@dump-sem-ir-begin
fn F(c: C) ->? form(val C) {
  return c;
}
//@dump-sem-ir-end

// --- todo_fail_dangling_local.carbon
library "[[@TEST_NAME]]";

class C { var a: i32; }

// TODO: Reject returning a dangling reference to a local variable.
fn F() ->? form(val C) {
  var c: C = {.a = 1};
  return c;
}

// --- todo_fail_dangling_temporary.carbon
library "[[@TEST_NAME]]";

class C { var a: i32; }

// TODO: Reject returning a dangling reference to a temporary.
fn F() ->? form(val C) {
  return {.a = 1};
}

// CHECK:STDOUT: --- param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = wrapper_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %.fbb: Core.Form = value_form %C [concrete]
// CHECK:STDOUT:   %.483: %pattern_type.98b = value_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.98b = return_slot_pattern %.483, %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = wrapper_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:     %.loc6_21.1: %pattern_type.98b = value_return_pattern [concrete = constants.%.483]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b = return_slot_pattern %.loc6_21.1, %.loc6_21.3 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6_25: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc6_21.2: Core.Form = value_form %C.ref.loc6_25 [concrete = constants.%.fbb]
// CHECK:STDOUT:     %.loc6_21.3: type = type_component_of %.loc6_21.2 [concrete = constants.%C]
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc6_9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = wrapper_binding c, %c.param
// CHECK:STDOUT:     %.loc6_21.4: %C = value_return
// CHECK:STDOUT:     %return: %C = return_slot %.loc6_21.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) -> val %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   return %c.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
