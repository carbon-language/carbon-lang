// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_negative_indexing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_negative_indexing.carbon

var c: [i32; 2] = (42, 42);
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+3]]:16: error: cannot access member of interface `Negate` in type `i32` that does not implement that interface
// CHECK:STDERR: var d: i32 = c[-10];
// CHECK:STDERR:                ^~~
var d: i32 = c[-10];


// CHECK:STDOUT: --- fail_negative_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 42 [template]
// CHECK:STDOUT:   %.6: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.5, %.5) [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 10 [template]
// CHECK:STDOUT:   %Negate.type: type = interface_type @Negate [template]
// CHECK:STDOUT:   %Self: %Negate.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op [template]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %Negate.type, %Op.type [template]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .Negate = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Core//prelude/operators/arithmetic, inst+77, loaded [template = constants.%Negate.type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/arithmetic, inst+79, unloaded
// CHECK:STDOUT:   %import_ref.4: %.10 = import_ref Core//prelude/operators/arithmetic, inst+96, loaded [template = constants.%.11]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/arithmetic, inst+91, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/arithmetic, inst+91, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_14: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_32.loc11, %.loc11_9.1 [template = i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref %.3 = var c
// CHECK:STDOUT:   %c: ref %.3 = bind_name c, %c.var
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_32.loc15, %.loc15_8.1 [template = i32]
// CHECK:STDOUT:   %d.var: ref i32 = var d
// CHECK:STDOUT:   %d: ref i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Negate {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .Op = imports.%import_ref.4
// CHECK:STDOUT:   witness = (imports.%import_ref.5)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(constants.%Self: %Negate.type) {
// CHECK:STDOUT:   %Self: %Negate.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Op.%Self (%Self)]() -> @Op.%Self (%Self);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: i32 = int_literal 42 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_24: i32 = int_literal 42 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_26.1: %.6 = tuple_literal (%.loc11_20, %.loc11_24)
// CHECK:STDOUT:   %.loc11_26.2: i32 = int_literal 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_26.3: ref i32 = array_index file.%c.var, %.loc11_26.2
// CHECK:STDOUT:   %.loc11_26.4: init i32 = initialize_from %.loc11_20 to %.loc11_26.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_26.5: i32 = int_literal 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc11_26.6: ref i32 = array_index file.%c.var, %.loc11_26.5
// CHECK:STDOUT:   %.loc11_26.7: init i32 = initialize_from %.loc11_24 to %.loc11_26.6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_26.8: init %.3 = array_init (%.loc11_26.4, %.loc11_26.7) to file.%c.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_27: init %.3 = converted %.loc11_26.1, %.loc11_26.8 [template = constants.%array]
// CHECK:STDOUT:   assign file.%c.var, %.loc11_27
// CHECK:STDOUT:   %c.ref: ref %.3 = name_ref c, file.%c
// CHECK:STDOUT:   %.loc15_17: i32 = int_literal 10 [template = constants.%.9]
// CHECK:STDOUT:   %Op.ref: %.10 = name_ref Op, imports.%import_ref.4 [template = constants.%.11]
// CHECK:STDOUT:   %.loc15_19.1: ref i32 = array_index %c.ref, <error> [template = <error>]
// CHECK:STDOUT:   %.loc15_19.2: i32 = bind_value %.loc15_19.1
// CHECK:STDOUT:   assign file.%d.var, %.loc15_19.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
