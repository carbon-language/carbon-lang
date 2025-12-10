// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/fail_convert_type_erased_type_to_facet.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/fail_convert_type_erased_type_to_facet.carbon

interface Animal {}

class Goat {}
impl Goat as Animal {}

fn WalkAnimal(a:! Animal) {}

fn F() {
  // CHECK:STDERR: fail_convert_type_erased_type_to_facet.carbon:[[@LINE+4]]:7: error: semantics TODO: `local `let :!` bindings are currently unsupported` [SemanticsTodo]
  // CHECK:STDERR:   let x:! type = Goat;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let x:! type = Goat;
  WalkAnimal(x);
}

// CHECK:STDOUT: --- fail_convert_type_erased_type_to_facet.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self: %Animal.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %Animal.type = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @<null name>: <unexpected>.inst{{[0-9A-F]+}}.loc18_6 as <unexpected>.inst{{[0-9A-F]+}}.loc18_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <unexpected>.inst{{[0-9A-F]+}}.loc18_21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Goat {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Goat
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WalkAnimal(<unexpected>.inst{{[0-9A-F]+}}.loc20_15: %Animal.type) {
// CHECK:STDOUT:   %a: %Animal.type = symbolic_binding a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WalkAnimal(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
