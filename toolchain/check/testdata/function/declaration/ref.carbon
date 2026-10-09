// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/ref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/ref.carbon

// --- ref_return.carbon

//@include-in-dumps

fn F() -> ref i32;

fn G() {
  let unused ref x: i32 = F();
}

// --- ref_return_parenthesized_type.carbon

library "[[@TEST_NAME]]";

fn F() -> ref (i32);

// --- fail_todo_ref_return_parenthesized.carbon

library "[[@TEST_NAME]]";

// TODO: Per p5434, a return form starting with `(` is a compound form, which
// isn't supported yet.
// CHECK:STDERR: fail_todo_ref_return_parenthesized.carbon:[[@LINE+4]]:12: error: `ref` tag is not an argument to a `ref` parameter [RefTagNoRefParam]
// CHECK:STDERR: fn F() -> (ref i32);
// CHECK:STDERR:            ^~~~~~~
// CHECK:STDERR:
fn F() -> (ref i32);

// CHECK:STDOUT: --- ref_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.058: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.96d: %pattern_type.9d9 = ref_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.9d9 = return_slot_pattern %.96d, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.9d9 = ref_binding_pattern x [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_11.1: %pattern_type.9d9 = ref_return_pattern [concrete = constants.%.96d]
// CHECK:STDOUT:     %return.patt: %pattern_type.9d9 = return_slot_pattern %.loc4_11.1, %.loc4_11.3 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.2: Core.Form = ref_form %i32 [concrete = constants.%.058]
// CHECK:STDOUT:     %.loc4_11.3: type = type_component_of %.loc4_11.2 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.4: ref %i32 = ref_return
// CHECK:STDOUT:     %return: ref %i32 = return_slot %.loc4_11.4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> ref %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: ref %i32 = call %F.ref()
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: ref %i32 = wrapper_binding x, %F.call
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.9d9 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
