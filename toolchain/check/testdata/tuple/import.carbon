// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/import.carbon

// --- implicit.carbon

package Implicit;

var a_ref: (i32,) = (0,);
var b_ref: (((i32,), i32), (i32, i32)) = (((0,), 1), (2, 3));

class C(X:! (i32, i32)) {}

fn F() -> C((1, 2));

// --- implicit.impl.carbon

impl package Implicit;

var a: (i32,) = a_ref;
var b: (((i32,), i32), (i32, i32)) = b_ref;
var c: C((1, 2)) = F();

// --- fail_bad_type.impl.carbon

impl package Implicit;

// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE+8]]:14: error: cannot initialize tuple of 2 element(s) from tuple with 3 element(s).
// CHECK:STDERR: var c_bad: C((1, 2, 3)) = F();
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE-5]]:6: in import
// CHECK:STDERR: implicit.carbon:7:9: note: initializing generic parameter `X` declared here
// CHECK:STDERR: class C(X:! (i32, i32)) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
var c_bad: C((1, 2, 3)) = F();

// --- fail_bad_value.impl.carbon

impl package Implicit;

// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+6]]:1: error: cannot implicitly convert from `C` to `C`
// CHECK:STDERR: var c_bad: C((3, 4)) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+3]]:1: note: type `C` does not implement interface `ImplicitAs`
// CHECK:STDERR: var c_bad: C((3, 4)) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
var c_bad: C((3, 4)) = F();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %tuple.1: %.3 = tuple_value (%.4) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (%.2, type) [template]
// CHECK:STDOUT:   %.6: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.7: type = tuple_type (%.5, %.6) [template]
// CHECK:STDOUT:   %.8: type = tuple_type (%.3, i32) [template]
// CHECK:STDOUT:   %.9: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.10: type = tuple_type (%.8, %.9) [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.9 [template]
// CHECK:STDOUT:   %.12: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %.13: type = tuple_type (%.12, %.11) [template]
// CHECK:STDOUT:   %.14: type = ptr_type %.13 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.16: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.17: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %tuple.2: %.8 = tuple_value (%tuple.1, %.15) [template]
// CHECK:STDOUT:   %tuple.3: %.9 = tuple_value (%.16, %.17) [template]
// CHECK:STDOUT:   %tuple.4: %.10 = tuple_value (%tuple.2, %tuple.3) [template]
// CHECK:STDOUT:   %X: %.9 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%X) [symbolic]
// CHECK:STDOUT:   %.18: type = struct_type {} [template]
// CHECK:STDOUT:   %.19: <witness> = complete_type_witness %.18 [template]
// CHECK:STDOUT:   %tuple.5: %.9 = tuple_value (%.15, %.16) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%tuple.5) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .b_ref = %b_ref
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc4_17.1: %.2 = tuple_literal (%int.make_type_32.loc4)
// CHECK:STDOUT:   %.loc4_17.2: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_17.3: type = converted %int.make_type_32.loc4, %.loc4_17.2 [template = i32]
// CHECK:STDOUT:   %.loc4_17.4: type = converted %.loc4_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %a_ref.var: ref %.3 = var a_ref
// CHECK:STDOUT:   %a_ref: ref %.3 = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %int.make_type_32.loc5_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_19: %.2 = tuple_literal (%int.make_type_32.loc5_15)
// CHECK:STDOUT:   %int.make_type_32.loc5_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_25: %.5 = tuple_literal (%.loc5_19, %int.make_type_32.loc5_22)
// CHECK:STDOUT:   %int.make_type_32.loc5_29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc5_34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_37: %.6 = tuple_literal (%int.make_type_32.loc5_29, %int.make_type_32.loc5_34)
// CHECK:STDOUT:   %.loc5_38.1: %.7 = tuple_literal (%.loc5_25, %.loc5_37)
// CHECK:STDOUT:   %.loc5_38.2: type = value_of_initializer %int.make_type_32.loc5_15 [template = i32]
// CHECK:STDOUT:   %.loc5_38.3: type = converted %int.make_type_32.loc5_15, %.loc5_38.2 [template = i32]
// CHECK:STDOUT:   %.loc5_38.4: type = converted %.loc5_19, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_38.5: type = value_of_initializer %int.make_type_32.loc5_22 [template = i32]
// CHECK:STDOUT:   %.loc5_38.6: type = converted %int.make_type_32.loc5_22, %.loc5_38.5 [template = i32]
// CHECK:STDOUT:   %.loc5_38.7: type = converted %.loc5_25, constants.%.8 [template = constants.%.8]
// CHECK:STDOUT:   %.loc5_38.8: type = value_of_initializer %int.make_type_32.loc5_29 [template = i32]
// CHECK:STDOUT:   %.loc5_38.9: type = converted %int.make_type_32.loc5_29, %.loc5_38.8 [template = i32]
// CHECK:STDOUT:   %.loc5_38.10: type = value_of_initializer %int.make_type_32.loc5_34 [template = i32]
// CHECK:STDOUT:   %.loc5_38.11: type = converted %int.make_type_32.loc5_34, %.loc5_38.10 [template = i32]
// CHECK:STDOUT:   %.loc5_38.12: type = converted %.loc5_37, constants.%.9 [template = constants.%.9]
// CHECK:STDOUT:   %.loc5_38.13: type = converted %.loc5_38.1, constants.%.10 [template = constants.%.10]
// CHECK:STDOUT:   %b_ref.var: ref %.10 = var b_ref
// CHECK:STDOUT:   %b_ref: ref %.10 = bind_name b_ref, %b_ref.var
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %X.patt: %.9 = symbolic_binding_pattern X, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc7_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc7_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_22.1: %.6 = tuple_literal (%int.make_type_32.loc7_14, %int.make_type_32.loc7_19)
// CHECK:STDOUT:     %.loc7_22.2: type = value_of_initializer %int.make_type_32.loc7_14 [template = i32]
// CHECK:STDOUT:     %.loc7_22.3: type = converted %int.make_type_32.loc7_14, %.loc7_22.2 [template = i32]
// CHECK:STDOUT:     %.loc7_22.4: type = value_of_initializer %int.make_type_32.loc7_19 [template = i32]
// CHECK:STDOUT:     %.loc7_22.5: type = converted %int.make_type_32.loc7_19, %.loc7_22.4 [template = i32]
// CHECK:STDOUT:     %.loc7_22.6: type = converted %.loc7_22.1, constants.%.9 [template = constants.%.9]
// CHECK:STDOUT:     %X.param: %.9 = param X, runtime_param<invalid>
// CHECK:STDOUT:     %X.loc7: %.9 = bind_symbolic_name X, 0, %X.param [symbolic = %X.1 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc9_14: i32 = int_literal 1 [template = constants.%.15]
// CHECK:STDOUT:     %.loc9_17: i32 = int_literal 2 [template = constants.%.16]
// CHECK:STDOUT:     %.loc9_18: %.9 = tuple_literal (%.loc9_14, %.loc9_17)
// CHECK:STDOUT:     %tuple: %.9 = tuple_value (%.loc9_14, %.loc9_17) [template = constants.%tuple.5]
// CHECK:STDOUT:     %.loc9_12: %.9 = converted %.loc9_18, %tuple [template = constants.%tuple.5]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple.5) [template = constants.%C.3]
// CHECK:STDOUT:     %return: ref %C.3 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%X.loc7: %.9) {
// CHECK:STDOUT:   %X.1: %.9 = bind_symbolic_name X, 0 [symbolic = %X.1 (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc7_26: <witness> = complete_type_witness %.18 [template = constants.%.19]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_22: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_24.1: %.3 = tuple_literal (%.loc4_22)
// CHECK:STDOUT:   %.loc4_24.2: init %.3 = tuple_init (%.loc4_22) to file.%a_ref.var [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc4_25: init %.3 = converted %.loc4_24.1, %.loc4_24.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc4_25
// CHECK:STDOUT:   %.loc5_45: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_47.1: %.3 = tuple_literal (%.loc5_45)
// CHECK:STDOUT:   %.loc5_50: i32 = int_literal 1 [template = constants.%.15]
// CHECK:STDOUT:   %.loc5_51.1: %.8 = tuple_literal (%.loc5_47.1, %.loc5_50)
// CHECK:STDOUT:   %.loc5_55: i32 = int_literal 2 [template = constants.%.16]
// CHECK:STDOUT:   %.loc5_58: i32 = int_literal 3 [template = constants.%.17]
// CHECK:STDOUT:   %.loc5_59.1: %.9 = tuple_literal (%.loc5_55, %.loc5_58)
// CHECK:STDOUT:   %.loc5_60.1: %.10 = tuple_literal (%.loc5_51.1, %.loc5_59.1)
// CHECK:STDOUT:   %.loc5_60.2: ref %.8 = tuple_access file.%b_ref.var, element0
// CHECK:STDOUT:   %.loc5_51.2: ref %.3 = tuple_access %.loc5_60.2, element0
// CHECK:STDOUT:   %.loc5_47.2: init %.3 = tuple_init (%.loc5_45) to %.loc5_51.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc5_51.3: init %.3 = converted %.loc5_47.1, %.loc5_47.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc5_51.4: init %.3 = initialize_from %.loc5_51.3 to %.loc5_51.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc5_51.5: ref i32 = tuple_access %.loc5_60.2, element1
// CHECK:STDOUT:   %.loc5_51.6: init i32 = initialize_from %.loc5_50 to %.loc5_51.5 [template = constants.%.15]
// CHECK:STDOUT:   %.loc5_51.7: init %.8 = tuple_init (%.loc5_51.4, %.loc5_51.6) to %.loc5_60.2 [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc5_60.3: init %.8 = converted %.loc5_51.1, %.loc5_51.7 [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc5_60.4: ref %.9 = tuple_access file.%b_ref.var, element1
// CHECK:STDOUT:   %.loc5_59.2: ref i32 = tuple_access %.loc5_60.4, element0
// CHECK:STDOUT:   %.loc5_59.3: init i32 = initialize_from %.loc5_55 to %.loc5_59.2 [template = constants.%.16]
// CHECK:STDOUT:   %.loc5_59.4: ref i32 = tuple_access %.loc5_60.4, element1
// CHECK:STDOUT:   %.loc5_59.5: init i32 = initialize_from %.loc5_58 to %.loc5_59.4 [template = constants.%.17]
// CHECK:STDOUT:   %.loc5_59.6: init %.9 = tuple_init (%.loc5_59.3, %.loc5_59.5) to %.loc5_60.4 [template = constants.%tuple.3]
// CHECK:STDOUT:   %.loc5_60.5: init %.9 = converted %.loc5_59.1, %.loc5_59.6 [template = constants.%tuple.3]
// CHECK:STDOUT:   %.loc5_60.6: init %.10 = tuple_init (%.loc5_60.3, %.loc5_60.5) to file.%b_ref.var [template = constants.%tuple.4]
// CHECK:STDOUT:   %.loc5_61: init %.10 = converted %.loc5_60.1, %.loc5_60.6 [template = constants.%tuple.4]
// CHECK:STDOUT:   assign file.%b_ref.var, %.loc5_61
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X.1 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.5) {
// CHECK:STDOUT:   %X.1 => constants.%tuple.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (%.2, type) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.6: type = tuple_type (%.4, %.5) [template]
// CHECK:STDOUT:   %.7: type = tuple_type (%.3, i32) [template]
// CHECK:STDOUT:   %.8: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.9: type = tuple_type (%.7, %.8) [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %.12: type = tuple_type (%.11, %.10) [template]
// CHECK:STDOUT:   %.13: type = ptr_type %.12 [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.14: type = struct_type {} [template]
// CHECK:STDOUT:   %.15: <witness> = complete_type_witness %.14 [template]
// CHECK:STDOUT:   %X: %.8 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%X) [symbolic]
// CHECK:STDOUT:   %.16: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.17: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %tuple: %.8 = tuple_value (%.16, %.17) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%tuple) [template]
// CHECK:STDOUT:   %.18: type = ptr_type %.14 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: ref %.3 = import_ref Implicit//default, inst+17, loaded
// CHECK:STDOUT:   %import_ref.2: ref %.9 = import_ref Implicit//default, inst+57, loaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, inst+104, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, inst+122, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.6 = import_ref Implicit//default, inst+108, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.1
// CHECK:STDOUT:     .b_ref = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .F = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc4_13.1: %.2 = tuple_literal (%int.make_type_32.loc4)
// CHECK:STDOUT:   %.loc4_13.2: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_13.3: type = converted %int.make_type_32.loc4, %.loc4_13.2 [template = i32]
// CHECK:STDOUT:   %.loc4_13.4: type = converted %.loc4_13.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc5_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_15: %.2 = tuple_literal (%int.make_type_32.loc5_11)
// CHECK:STDOUT:   %int.make_type_32.loc5_18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_21: %.4 = tuple_literal (%.loc5_15, %int.make_type_32.loc5_18)
// CHECK:STDOUT:   %int.make_type_32.loc5_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc5_30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_33: %.5 = tuple_literal (%int.make_type_32.loc5_25, %int.make_type_32.loc5_30)
// CHECK:STDOUT:   %.loc5_34.1: %.6 = tuple_literal (%.loc5_21, %.loc5_33)
// CHECK:STDOUT:   %.loc5_34.2: type = value_of_initializer %int.make_type_32.loc5_11 [template = i32]
// CHECK:STDOUT:   %.loc5_34.3: type = converted %int.make_type_32.loc5_11, %.loc5_34.2 [template = i32]
// CHECK:STDOUT:   %.loc5_34.4: type = converted %.loc5_15, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_34.5: type = value_of_initializer %int.make_type_32.loc5_18 [template = i32]
// CHECK:STDOUT:   %.loc5_34.6: type = converted %int.make_type_32.loc5_18, %.loc5_34.5 [template = i32]
// CHECK:STDOUT:   %.loc5_34.7: type = converted %.loc5_21, constants.%.7 [template = constants.%.7]
// CHECK:STDOUT:   %.loc5_34.8: type = value_of_initializer %int.make_type_32.loc5_25 [template = i32]
// CHECK:STDOUT:   %.loc5_34.9: type = converted %int.make_type_32.loc5_25, %.loc5_34.8 [template = i32]
// CHECK:STDOUT:   %.loc5_34.10: type = value_of_initializer %int.make_type_32.loc5_30 [template = i32]
// CHECK:STDOUT:   %.loc5_34.11: type = converted %int.make_type_32.loc5_30, %.loc5_34.10 [template = i32]
// CHECK:STDOUT:   %.loc5_34.12: type = converted %.loc5_33, constants.%.8 [template = constants.%.8]
// CHECK:STDOUT:   %.loc5_34.13: type = converted %.loc5_34.1, constants.%.9 [template = constants.%.9]
// CHECK:STDOUT:   %b.var: ref %.9 = var b
// CHECK:STDOUT:   %b: ref %.9 = bind_name b, %b.var
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc6_11: i32 = int_literal 1 [template = constants.%.16]
// CHECK:STDOUT:   %.loc6_14: i32 = int_literal 2 [template = constants.%.17]
// CHECK:STDOUT:   %.loc6_15: %.8 = tuple_literal (%.loc6_11, %.loc6_14)
// CHECK:STDOUT:   %tuple: %.8 = tuple_value (%.loc6_11, %.loc6_14) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_9: %.8 = converted %.loc6_15, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%tuple) [template = constants.%C.3]
// CHECK:STDOUT:   %c.var: ref %C.3 = var c
// CHECK:STDOUT:   %c: ref %C.3 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%X: %.8) {
// CHECK:STDOUT:   %X: %.8 = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %.3 = name_ref a_ref, imports.%import_ref.1
// CHECK:STDOUT:   %.loc4_17.1: ref i32 = tuple_access %a_ref.ref, element0
// CHECK:STDOUT:   %.loc4_17.2: i32 = bind_value %.loc4_17.1
// CHECK:STDOUT:   %.loc4_17.3: init %.3 = tuple_init (%.loc4_17.2) to file.%a.var
// CHECK:STDOUT:   %.loc4_22: init %.3 = converted %a_ref.ref, %.loc4_17.3
// CHECK:STDOUT:   assign file.%a.var, %.loc4_22
// CHECK:STDOUT:   %b_ref.ref: ref %.9 = name_ref b_ref, imports.%import_ref.2
// CHECK:STDOUT:   %.loc5_38.1: ref %.7 = tuple_access %b_ref.ref, element0
// CHECK:STDOUT:   %.loc5_38.2: ref %.3 = tuple_access %.loc5_38.1, element0
// CHECK:STDOUT:   %.loc5_38.3: ref i32 = tuple_access %.loc5_38.2, element0
// CHECK:STDOUT:   %.loc5_38.4: i32 = bind_value %.loc5_38.3
// CHECK:STDOUT:   %.loc5_38.5: ref %.7 = tuple_access file.%b.var, element0
// CHECK:STDOUT:   %.loc5_38.6: ref %.3 = tuple_access %.loc5_38.5, element0
// CHECK:STDOUT:   %.loc5_38.7: init %.3 = tuple_init (%.loc5_38.4) to %.loc5_38.6
// CHECK:STDOUT:   %.loc5_38.8: init %.3 = converted %.loc5_38.2, %.loc5_38.7
// CHECK:STDOUT:   %.loc5_38.9: init %.3 = initialize_from %.loc5_38.8 to %.loc5_38.6
// CHECK:STDOUT:   %.loc5_38.10: ref i32 = tuple_access %.loc5_38.1, element1
// CHECK:STDOUT:   %.loc5_38.11: i32 = bind_value %.loc5_38.10
// CHECK:STDOUT:   %.loc5_38.12: ref i32 = tuple_access %.loc5_38.5, element1
// CHECK:STDOUT:   %.loc5_38.13: init i32 = initialize_from %.loc5_38.11 to %.loc5_38.12
// CHECK:STDOUT:   %.loc5_38.14: init %.7 = tuple_init (%.loc5_38.9, %.loc5_38.13) to %.loc5_38.5
// CHECK:STDOUT:   %.loc5_38.15: init %.7 = converted %.loc5_38.1, %.loc5_38.14
// CHECK:STDOUT:   %.loc5_38.16: ref %.8 = tuple_access %b_ref.ref, element1
// CHECK:STDOUT:   %.loc5_38.17: ref i32 = tuple_access %.loc5_38.16, element0
// CHECK:STDOUT:   %.loc5_38.18: i32 = bind_value %.loc5_38.17
// CHECK:STDOUT:   %.loc5_38.19: ref %.8 = tuple_access file.%b.var, element1
// CHECK:STDOUT:   %.loc5_38.20: ref i32 = tuple_access %.loc5_38.19, element0
// CHECK:STDOUT:   %.loc5_38.21: init i32 = initialize_from %.loc5_38.18 to %.loc5_38.20
// CHECK:STDOUT:   %.loc5_38.22: ref i32 = tuple_access %.loc5_38.16, element1
// CHECK:STDOUT:   %.loc5_38.23: i32 = bind_value %.loc5_38.22
// CHECK:STDOUT:   %.loc5_38.24: ref i32 = tuple_access %.loc5_38.19, element1
// CHECK:STDOUT:   %.loc5_38.25: init i32 = initialize_from %.loc5_38.23 to %.loc5_38.24
// CHECK:STDOUT:   %.loc5_38.26: init %.8 = tuple_init (%.loc5_38.21, %.loc5_38.25) to %.loc5_38.19
// CHECK:STDOUT:   %.loc5_38.27: init %.8 = converted %.loc5_38.16, %.loc5_38.26
// CHECK:STDOUT:   %.loc5_38.28: init %.9 = tuple_init (%.loc5_38.15, %.loc5_38.27) to file.%b.var
// CHECK:STDOUT:   %.loc5_43: init %.9 = converted %b_ref.ref, %.loc5_38.28
// CHECK:STDOUT:   assign file.%b.var, %.loc5_43
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.4 [template = constants.%F]
// CHECK:STDOUT:   %.loc6: ref %C.3 = splice_block file.%c.var {}
// CHECK:STDOUT:   %F.call: init %C.3 = call %F.ref() to %.loc6
// CHECK:STDOUT:   assign file.%c.var, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple) {
// CHECK:STDOUT:   %X => constants.%tuple
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_type.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %X: %.4 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%X) [symbolic]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.8: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %tuple: %.4 = tuple_value (%.5, %.6) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%tuple) [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Implicit//default, inst+17, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Implicit//default, inst+57, unloaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, inst+104, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, inst+122, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5 = import_ref Implicit//default, inst+108, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.1
// CHECK:STDOUT:     .b_ref = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .F = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c_bad = %c_bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc12_15: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_18: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_21: i32 = int_literal 3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc12_22: %.8 = tuple_literal (%.loc12_15, %.loc12_18, %.loc12_21)
// CHECK:STDOUT:   %c_bad.var: ref <error> = var c_bad
// CHECK:STDOUT:   %c_bad: ref <error> = bind_name c_bad, %c_bad.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%X: %.4) {
// CHECK:STDOUT:   %X: %.4 = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.4 [template = constants.%F]
// CHECK:STDOUT:   %.loc12: ref %C.3 = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.3 = call %F.ref() to %.loc12
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple) {
// CHECK:STDOUT:   %X => constants.%tuple
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_value.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %X: %.4 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%X) [symbolic]
// CHECK:STDOUT:   %.5: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %tuple.1: %.4 = tuple_value (%.5, %.6) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%tuple.1) [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %tuple.2: %.4 = tuple_value (%.10, %.9) [template]
// CHECK:STDOUT:   %C.4: type = class_type @C, @C(%tuple.2) [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.10 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = interface_type @ImplicitAs, @ImplicitAs(%C.3) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%C.3) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.13: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.14: %.13 = assoc_entity element0, imports.%import_ref.10 [template]
// CHECK:STDOUT:   %.15: %.11 = assoc_entity element0, imports.%import_ref.11 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Implicit//default, inst+17, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Implicit//default, inst+57, unloaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, inst+104, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, inst+122, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5 = import_ref Implicit//default, inst+108, unloaded
// CHECK:STDOUT:   %import_ref.6: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.8: @ImplicitAs.%.1 (%.11) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.2 (constants.%.15)]
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.1
// CHECK:STDOUT:     .b_ref = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .F = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c_bad = %c_bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc10_15: i32 = int_literal 3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_18: i32 = int_literal 4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc10_19: %.4 = tuple_literal (%.loc10_15, %.loc10_18)
// CHECK:STDOUT:   %tuple: %.4 = tuple_value (%.loc10_15, %.loc10_18) [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc10_13: %.4 = converted %.loc10_19, %tuple [template = constants.%tuple.1]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%tuple.1) [template = constants.%C.3]
// CHECK:STDOUT:   %c_bad.var: ref %C.3 = var c_bad
// CHECK:STDOUT:   %c_bad: ref %C.3 = bind_name c_bad, %c_bad.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.11) = assoc_entity element0, imports.%import_ref.10 [symbolic = %.2 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.7
// CHECK:STDOUT:     .Convert = imports.%import_ref.8
// CHECK:STDOUT:     witness = (imports.%import_ref.9)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%X: %.4) {
// CHECK:STDOUT:   %X: %.4 = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.4;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.4 [template = constants.%F]
// CHECK:STDOUT:   %.loc10_25.1: ref %C.4 = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.4 = call %F.ref() to %.loc10_25.1
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(constants.%C.3) [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %.loc10_27.1: %.13 = specific_constant imports.%import_ref.8, @ImplicitAs(constants.%C.3) [template = constants.%.14]
// CHECK:STDOUT:   %Convert.ref: %.13 = name_ref Convert, %.loc10_27.1 [template = constants.%.14]
// CHECK:STDOUT:   %.loc10_25.2: ref %C.4 = temporary %.loc10_25.1, %F.call
// CHECK:STDOUT:   %.loc10_27.2: %C.3 = converted %F.call, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.1) {
// CHECK:STDOUT:   %X => constants.%tuple.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.2) {
// CHECK:STDOUT:   %X => constants.%tuple.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%C.3) {
// CHECK:STDOUT:   %Dest => constants.%C.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT:   %.2 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
