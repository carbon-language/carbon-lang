// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_negative_indexing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_negative_indexing.carbon

let a: [i32; 2] = (12, 6);
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+4]]:16: ERROR: Cannot access member of interface Negate in type i32 that does not implement that interface.
// CHECK:STDERR: var b: i32 = a[-10];
// CHECK:STDERR:                ^~~
// CHECK:STDERR:
var b: i32 = a[-10];

let c: (i32, i32) = (1, 2);
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+7]]:14: ERROR: Member name of type `<error>` in compound member access is not an instance member or an interface member.
// CHECK:STDERR: let d: i32 = a.(-1);
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+3]]:17: ERROR: Cannot access member of interface Negate in type i32 that does not implement that interface.
// CHECK:STDERR: let d: i32 = a.(-1);
// CHECK:STDERR:                 ^~
let d: i32 = a.(-1);

// CHECK:STDOUT: --- fail_negative_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.5, %.6) [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 10 [template]
// CHECK:STDOUT:   %.11: type = interface_type @Negate [template]
// CHECK:STDOUT:   %Self: %.11 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op [template]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [template]
// CHECK:STDOUT:   %.12: type = assoc_entity_type @Negate, %Op.type [template]
// CHECK:STDOUT:   %.13: %.12 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.14: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.15: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %tuple: %.7 = tuple_value (%.9, %.2) [template]
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
// CHECK:STDOUT:   %import_ref.2: type = import_ref Core//prelude/operators/arithmetic, inst+72, loaded [template = constants.%.11]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/arithmetic, inst+74, unloaded
// CHECK:STDOUT:   %import_ref.4: %.12 = import_ref Core//prelude/operators/arithmetic, inst+90, loaded [template = constants.%.13]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/arithmetic, inst+85, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/arithmetic, inst+85, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:     .d = @__global_init.%d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_14: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_32.loc11, %.loc11_9.1 [template = i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_32.loc15, %.loc15_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int.make_type_32.loc17_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc17_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_17.1: %.14 = tuple_literal (%int.make_type_32.loc17_9, %int.make_type_32.loc17_14)
// CHECK:STDOUT:   %.loc17_17.2: type = value_of_initializer %int.make_type_32.loc17_9 [template = i32]
// CHECK:STDOUT:   %.loc17_17.3: type = converted %int.make_type_32.loc17_9, %.loc17_17.2 [template = i32]
// CHECK:STDOUT:   %.loc17_17.4: type = value_of_initializer %int.make_type_32.loc17_14 [template = i32]
// CHECK:STDOUT:   %.loc17_17.5: type = converted %int.make_type_32.loc17_14, %.loc17_17.4 [template = i32]
// CHECK:STDOUT:   %.loc17_17.6: type = converted %.loc17_17.1, constants.%.7 [template = constants.%.7]
// CHECK:STDOUT:   %int.make_type_32.loc18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_8.1: type = value_of_initializer %int.make_type_32.loc18 [template = i32]
// CHECK:STDOUT:   %.loc18_8.2: type = converted %int.make_type_32.loc18, %.loc18_8.1 [template = i32]
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
// CHECK:STDOUT: generic fn @Op(constants.%Self: %.11) {
// CHECK:STDOUT:   %Self: %.11 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: %Self]() -> %Self;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: i32 = int_literal 12 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_24: i32 = int_literal 6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_25.1: %.7 = tuple_literal (%.loc11_20, %.loc11_24)
// CHECK:STDOUT:   %.loc11_25.2: ref %.3 = temporary_storage
// CHECK:STDOUT:   %.loc11_25.3: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc11_25.4: ref i32 = array_index %.loc11_25.2, %.loc11_25.3
// CHECK:STDOUT:   %.loc11_25.5: init i32 = initialize_from %.loc11_20 to %.loc11_25.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_25.6: i32 = int_literal 1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_25.7: ref i32 = array_index %.loc11_25.2, %.loc11_25.6
// CHECK:STDOUT:   %.loc11_25.8: init i32 = initialize_from %.loc11_24 to %.loc11_25.7 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_25.9: init %.3 = array_init (%.loc11_25.5, %.loc11_25.8) to %.loc11_25.2 [template = constants.%array]
// CHECK:STDOUT:   %.loc11_26.1: init %.3 = converted %.loc11_25.1, %.loc11_25.9 [template = constants.%array]
// CHECK:STDOUT:   %.loc11_26.2: ref %.3 = temporary %.loc11_25.2, %.loc11_26.1
// CHECK:STDOUT:   %.loc11_26.3: %.3 = bind_value %.loc11_26.2
// CHECK:STDOUT:   %a: %.3 = bind_name a, %.loc11_26.3
// CHECK:STDOUT:   %a.ref.loc15: %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc15_17: i32 = int_literal 10 [template = constants.%.10]
// CHECK:STDOUT:   %.loc15_19.1: ref %.3 = value_as_ref %a.ref.loc15
// CHECK:STDOUT:   %.loc15_19.2: ref i32 = array_index %.loc15_19.1, <error> [template = <error>]
// CHECK:STDOUT:   %.loc15_19.3: i32 = bind_value %.loc15_19.2
// CHECK:STDOUT:   assign file.%b.var, %.loc15_19.3
// CHECK:STDOUT:   %.loc17_22: i32 = int_literal 1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc17_25: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc17_26: %.7 = tuple_literal (%.loc17_22, %.loc17_25)
// CHECK:STDOUT:   %tuple: %.7 = tuple_value (%.loc17_22, %.loc17_25) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc17_27: %.7 = converted %.loc17_26, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %c: %.7 = bind_name c, %.loc17_27
// CHECK:STDOUT:   %a.ref.loc18: %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc18: i32 = int_literal 1 [template = constants.%.9]
// CHECK:STDOUT:   %d: i32 = bind_name d, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
