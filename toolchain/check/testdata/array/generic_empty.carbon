// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/generic_empty.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/generic_empty.carbon

fn G(T:! type) {
  // We can initialize this without knowing T.
  var arr: [T; 0] = ();
}

// CHECK:STDOUT: --- generic_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.28: type = array_type %.27, %T [symbolic]
// CHECK:STDOUT:   %array: %.28 = tuple_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc11_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_6.1, runtime_param<invalid> [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc11_6.1: type) {
// CHECK:STDOUT:   %T.loc11_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc13_17.2: type = array_type constants.%.27, @G.%T.loc11_6.2 (%T) [symbolic = %.loc13_17.2 (constants.%.28)]
// CHECK:STDOUT:   %array: @G.%.loc13_17.2 (%.28) = tuple_value () [symbolic = %array (constants.%array)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_6.1 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc13_16.1: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:     %.loc13_16.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc13_16.3: <bound method> = bound_method %.loc13_16.1, %.loc13_16.2 [template = constants.%.26]
// CHECK:STDOUT:     %int.convert_checked: init Core.IntLiteral = call %.loc13_16.3(%.loc13_16.1) [template = constants.%.27]
// CHECK:STDOUT:     %.loc13_16.4: Core.IntLiteral = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:     %.loc13_16.5: Core.IntLiteral = converted %.loc13_16.1, %.loc13_16.4 [template = constants.%.27]
// CHECK:STDOUT:     %.loc13_17.1: type = array_type %.loc13_16.5, %T [symbolic = %.loc13_17.2 (constants.%.28)]
// CHECK:STDOUT:     %arr.var: ref @G.%.loc13_17.2 (%.28) = var arr
// CHECK:STDOUT:     %arr: ref @G.%.loc13_17.2 (%.28) = bind_name arr, %arr.var
// CHECK:STDOUT:     %.loc13_22.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_22.2: init @G.%.loc13_17.2 (%.28) = array_init () to %arr.var [symbolic = %array (constants.%array)]
// CHECK:STDOUT:     %.loc13_23: init @G.%.loc13_17.2 (%.28) = converted %.loc13_22.1, %.loc13_22.2 [symbolic = %array (constants.%array)]
// CHECK:STDOUT:     assign %arr.var, %.loc13_23
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc11_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_6.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
