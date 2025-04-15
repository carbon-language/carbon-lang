// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/min_prelude/global_lookup_in_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/min_prelude/global_lookup_in_scope.carbon

// --- basic.carbon

var x: {.v: ()} = {.v = ()};

fn Main() {
  var y: {.v: ()} = x;
}

// CHECK:STDOUT: --- global_lookup_in_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.v = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %struct_type.v = binding_pattern x
// CHECK:STDOUT:     %.loc14_1: %struct_type.v = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.v = var x
// CHECK:STDOUT:   %.loc14_15: type = splice_block %struct_type.v [concrete = constants.%struct_type.v] {
// CHECK:STDOUT:     %.loc14_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc14_14.2: type = converted %.loc14_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete = constants.%struct_type.v]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.v = bind_name x, %x.var
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %struct_type.v = binding_pattern y
// CHECK:STDOUT:     %.loc17_3.1: %struct_type.v = var_pattern %y.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %struct_type.v = var y
// CHECK:STDOUT:   %x.ref: ref %struct_type.v = name_ref x, file.%x
// CHECK:STDOUT:   %.loc17_21.1: ref %empty_tuple.type = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc17_21.2: ref %empty_tuple.type = struct_access %y.var, element0
// CHECK:STDOUT:   %.loc17_21.3: init %empty_tuple.type = tuple_init () to %.loc17_21.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_21.4: init %empty_tuple.type = converted %.loc17_21.1, %.loc17_21.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_21.5: init %struct_type.v = struct_init (%.loc17_21.4) to %y.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc17_3.2: init %struct_type.v = converted %x.ref, %.loc17_21.5 [concrete = constants.%struct]
// CHECK:STDOUT:   assign %y.var, %.loc17_3.2
// CHECK:STDOUT:   %.loc17_17: type = splice_block %struct_type.v [concrete = constants.%struct_type.v] {
// CHECK:STDOUT:     %.loc17_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc17_16.2: type = converted %.loc17_16.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete = constants.%struct_type.v]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %struct_type.v = bind_name y, %y.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc14_27.1: %struct_type.v = struct_literal (%.loc14_26.1)
// CHECK:STDOUT:   %.loc14_27.2: ref %empty_tuple.type = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc14_26.2: init %empty_tuple.type = tuple_init () to %.loc14_27.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc14_27.3: init %empty_tuple.type = converted %.loc14_26.1, %.loc14_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc14_27.4: init %struct_type.v = struct_init (%.loc14_27.3) to file.%x.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc14_1: init %struct_type.v = converted %.loc14_27.1, %.loc14_27.4 [concrete = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc14_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/destroy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
