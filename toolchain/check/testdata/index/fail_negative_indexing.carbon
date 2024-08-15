// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_negative_indexing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_negative_indexing.carbon

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+4]]:17: ERROR: Cannot access member of interface Negate in type i32 that does not implement that interface.
// CHECK:STDERR: var b: i32 = a.(-10);
// CHECK:STDERR:                 ^~~
// CHECK:STDERR:
var b: i32 = a.(-10);

var c: [i32; 2] = (42, 42);
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+3]]:16: ERROR: Cannot access member of interface Negate in type i32 that does not implement that interface.
// CHECK:STDERR: var d: i32 = c[-10];
// CHECK:STDERR:                ^~~
var d: i32 = c[-10];


// CHECK:STDOUT: --- fail_negative_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %tuple: %.3 = tuple_value (%.5, %.6) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 10 [template]
// CHECK:STDOUT:   %.8: type = interface_type @Negate [template]
// CHECK:STDOUT:   %Self: %.8 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op [template]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %.8, %Op.type [template]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.12: type = array_type %.11, i32 [template]
// CHECK:STDOUT:   %.13: type = ptr_type %.12 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 42 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.16: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %array: %.12 = tuple_value (%.14, %.14) [template]
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
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Core//prelude/operators/arithmetic, inst+72, loaded [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/arithmetic, inst+74, unloaded
// CHECK:STDOUT:   %import_ref.4: %.9 = import_ref Core//prelude/operators/arithmetic, inst+90, loaded [template = constants.%.10]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/arithmetic, inst+85, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/arithmetic, inst+85, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc11_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_17.1: %.2 = tuple_literal (%int.make_type_32.loc11_9, %int.make_type_32.loc11_14)
// CHECK:STDOUT:   %.loc11_17.2: type = value_of_initializer %int.make_type_32.loc11_9 [template = i32]
// CHECK:STDOUT:   %.loc11_17.3: type = converted %int.make_type_32.loc11_9, %.loc11_17.2 [template = i32]
// CHECK:STDOUT:   %.loc11_17.4: type = value_of_initializer %int.make_type_32.loc11_14 [template = i32]
// CHECK:STDOUT:   %.loc11_17.5: type = converted %int.make_type_32.loc11_14, %.loc11_17.4 [template = i32]
// CHECK:STDOUT:   %.loc11_17.6: type = converted %.loc11_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_8.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:   %.loc16_8.2: type = converted %int.make_type_32.loc16, %.loc16_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int.make_type_32.loc18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_14: i32 = int_literal 2 [template = constants.%.11]
// CHECK:STDOUT:   %.loc18_9.1: type = value_of_initializer %int.make_type_32.loc18 [template = i32]
// CHECK:STDOUT:   %.loc18_9.2: type = converted %int.make_type_32.loc18, %.loc18_9.1 [template = i32]
// CHECK:STDOUT:   %.loc18_15: type = array_type %.loc18_14, i32 [template = constants.%.12]
// CHECK:STDOUT:   %c.var: ref %.12 = var c
// CHECK:STDOUT:   %c: ref %.12 = bind_name c, %c.var
// CHECK:STDOUT:   %int.make_type_32.loc22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc22_8.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:   %.loc22_8.2: type = converted %int.make_type_32.loc22, %.loc22_8.1 [template = i32]
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
// CHECK:STDOUT: generic fn @Op(constants.%Self: %.8) {
// CHECK:STDOUT:   %Self: %.8 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Op.%Self (%Self)]() -> @Op.%Self (%Self);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_22: i32 = int_literal 12 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_26: i32 = int_literal 6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_27.1: %.3 = tuple_literal (%.loc11_22, %.loc11_26)
// CHECK:STDOUT:   %.loc11_27.2: ref i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_27.3: init i32 = initialize_from %.loc11_22 to %.loc11_27.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_27.4: ref i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_27.5: init i32 = initialize_from %.loc11_26 to %.loc11_27.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_27.6: init %.3 = tuple_init (%.loc11_27.3, %.loc11_27.5) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_28: init %.3 = converted %.loc11_27.1, %.loc11_27.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_28
// CHECK:STDOUT:   %a.ref: ref %.3 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc16_18: i32 = int_literal 10 [template = constants.%.7]
// CHECK:STDOUT:   %.loc16_15: ref <error> = tuple_index %a.ref, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   %.loc18_20: i32 = int_literal 42 [template = constants.%.14]
// CHECK:STDOUT:   %.loc18_24: i32 = int_literal 42 [template = constants.%.14]
// CHECK:STDOUT:   %.loc18_26.1: %.3 = tuple_literal (%.loc18_20, %.loc18_24)
// CHECK:STDOUT:   %.loc18_26.2: i32 = int_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %.loc18_26.3: ref i32 = array_index file.%c.var, %.loc18_26.2
// CHECK:STDOUT:   %.loc18_26.4: init i32 = initialize_from %.loc18_20 to %.loc18_26.3 [template = constants.%.14]
// CHECK:STDOUT:   %.loc18_26.5: i32 = int_literal 1 [template = constants.%.16]
// CHECK:STDOUT:   %.loc18_26.6: ref i32 = array_index file.%c.var, %.loc18_26.5
// CHECK:STDOUT:   %.loc18_26.7: init i32 = initialize_from %.loc18_24 to %.loc18_26.6 [template = constants.%.14]
// CHECK:STDOUT:   %.loc18_26.8: init %.12 = array_init (%.loc18_26.4, %.loc18_26.7) to file.%c.var [template = constants.%array]
// CHECK:STDOUT:   %.loc18_27: init %.12 = converted %.loc18_26.1, %.loc18_26.8 [template = constants.%array]
// CHECK:STDOUT:   assign file.%c.var, %.loc18_27
// CHECK:STDOUT:   %c.ref: ref %.12 = name_ref c, file.%c
// CHECK:STDOUT:   %.loc22_17: i32 = int_literal 10 [template = constants.%.7]
// CHECK:STDOUT:   %.loc22_19.1: ref i32 = array_index %c.ref, <error> [template = <error>]
// CHECK:STDOUT:   %.loc22_19.2: i32 = bind_value %.loc22_19.1
// CHECK:STDOUT:   assign file.%d.var, %.loc22_19.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
