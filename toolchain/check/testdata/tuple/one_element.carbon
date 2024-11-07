// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/one_element.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/one_element.carbon

var x: (i32,) = (4,);
var y: (i32,) = x;

// CHECK:STDOUT: --- one_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.1: i32 = int_value 4 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_13.1: %tuple.type.1 = tuple_literal (%int.make_type_32.loc11)
// CHECK:STDOUT:   %.loc11_13.2: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_13.3: type = converted %int.make_type_32.loc11, %.loc11_13.2 [template = i32]
// CHECK:STDOUT:   %.loc11_13.4: type = converted %.loc11_13.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %x.var: ref %tuple.type.2 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.2 = bind_name x, %x.var
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_13.1: %tuple.type.1 = tuple_literal (%int.make_type_32.loc12)
// CHECK:STDOUT:   %.loc12_13.2: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_13.3: type = converted %int.make_type_32.loc12, %.loc12_13.2 [template = i32]
// CHECK:STDOUT:   %.loc12_13.4: type = converted %.loc12_13.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %y.var: ref %tuple.type.2 = var y
// CHECK:STDOUT:   %y: ref %tuple.type.2 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_18: i32 = int_value 4 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_20.1: %tuple.type.2 = tuple_literal (%.loc11_18)
// CHECK:STDOUT:   %.loc11_20.2: init %tuple.type.2 = tuple_init (%.loc11_18) to file.%x.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_21: init %tuple.type.2 = converted %.loc11_20.1, %.loc11_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_21
// CHECK:STDOUT:   %x.ref: ref %tuple.type.2 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc12_17.1: ref i32 = tuple_access %x.ref, element0
// CHECK:STDOUT:   %.loc12_17.2: i32 = bind_value %.loc12_17.1
// CHECK:STDOUT:   %.loc12_17.3: init %tuple.type.2 = tuple_init (%.loc12_17.2) to file.%y.var
// CHECK:STDOUT:   %.loc12_18: init %tuple.type.2 = converted %x.ref, %.loc12_17.3
// CHECK:STDOUT:   assign file.%y.var, %.loc12_18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
