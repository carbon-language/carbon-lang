// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/choice/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/choice/generic.carbon

choice Always(T:! type) {
  Sunny
}

// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Always.type: type = generic_class_type @Always [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Always.generic: %Always.type = struct_value () [concrete]
// CHECK:STDOUT:   %Always: type = class_type @Always, @Always(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.discriminant: type = struct_type {.discriminant: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.discriminant [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Always [symbolic]
// CHECK:STDOUT:   %Always.val: %Always = struct_value (%empty_tuple) [symbolic]
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
// CHECK:STDOUT:     .Always = %Always.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Always.decl: %Always.type = class_decl @Always [concrete = constants.%Always.generic] {
// CHECK:STDOUT:     %T.patt.loc11_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc11_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Always(%T.loc11_15.1: type) {
// CHECK:STDOUT:   %T.loc11_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Always: type = class_type @Always, @Always(%T.loc11_15.2) [symbolic = %Always (constants.%Always)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Always.%Always (%Always) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Always.val: @Always.%Always (%Always) = struct_value (constants.%empty_tuple) [symbolic = %Always.val (constants.%Always.val)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.discriminant [concrete = constants.%complete_type]
// CHECK:STDOUT:     %.loc13_1.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13_1.2: %empty_tuple.type = converted %.loc13_1.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13_1.3: %struct_type.discriminant = struct_literal (%.loc13_1.2)
// CHECK:STDOUT:     %.loc13_1.4: ref @Always.%Always (%Always) = temporary_storage
// CHECK:STDOUT:     %.loc13_1.5: ref %empty_tuple.type = class_element_access %.loc13_1.4, element0
// CHECK:STDOUT:     %.loc13_1.6: init %empty_tuple.type = tuple_init () to %.loc13_1.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13_1.7: init %empty_tuple.type = converted %.loc13_1.2, %.loc13_1.6 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13_1.8: init @Always.%Always (%Always) = class_init (%.loc13_1.7), %.loc13_1.4 [symbolic = %Always.val (constants.%Always.val)]
// CHECK:STDOUT:     %.loc13_1.9: ref @Always.%Always (%Always) = temporary %.loc13_1.4, %.loc13_1.8
// CHECK:STDOUT:     %.loc13_1.10: ref @Always.%Always (%Always) = converted %.loc13_1.3, %.loc13_1.9
// CHECK:STDOUT:     %.loc13_1.11: @Always.%Always (%Always) = bind_value %.loc13_1.10
// CHECK:STDOUT:     %Sunny: @Always.%Always (%Always) = bind_name Sunny, %.loc13_1.11
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Always
// CHECK:STDOUT:     .Sunny = %Sunny
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Always(constants.%T) {
// CHECK:STDOUT:   %T.loc11_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_15.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Always(%T.loc11_15.2) {}
// CHECK:STDOUT:
