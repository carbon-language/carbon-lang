// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_type_mismatch.carbon

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:19: error: cannot implicitly convert from `String` to `i32`
// CHECK:STDERR: var a: [i32; 3] = (1, "Hello", "World");
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:19: note: type `String` does not implement interface `ImplicitAs`
// CHECK:STDERR: var a: [i32; 3] = (1, "Hello", "World");
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: [i32; 3] = (1, "Hello", "World");

var t1: (i32, String, String);
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:19: error: cannot implicitly convert from `String` to `i32`
// CHECK:STDERR: var b: [i32; 3] = t1;
// CHECK:STDERR:                   ^~
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:19: note: type `String` does not implement interface `ImplicitAs`
// CHECK:STDERR: var b: [i32; 3] = t1;
// CHECK:STDERR:                   ^~
// CHECK:STDERR:
var b: [i32; 3] = t1;

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:19: error: cannot initialize array of 3 element(s) from 2 initializer(s)
// CHECK:STDERR: var c: [i32; 3] = (1, 2);
// CHECK:STDERR:                   ^~~~~~
// CHECK:STDERR:
var c: [i32; 3] = (1, 2);

var t2: (i32, i32);
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:19: error: cannot initialize array of 3 element(s) from tuple with 2 element(s).
// CHECK:STDERR: var d: [i32; 3] = t2;
// CHECK:STDERR:                   ^~
var d: [i32; 3] = t2;

// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: type = ptr_type String [template]
// CHECK:STDOUT:   %.7: String = string_literal "Hello" [template]
// CHECK:STDOUT:   %.8: String = string_literal "World" [template]
// CHECK:STDOUT:   %.9: type = tuple_type (i32, String, String) [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = interface_type @ImplicitAs, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.13: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.14: %.13 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.15: %.11 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %.16: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.17: type = tuple_type (i32, %.6, %.6) [template]
// CHECK:STDOUT:   %.18: type = ptr_type %.17 [template]
// CHECK:STDOUT:   %.19: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.20: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.21: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.22: type = ptr_type %.20 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
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
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.11) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.2 (constants.%.15)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .t1 = %t1
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .t2 = %t2
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_14: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_9.1: type = value_of_initializer %int.make_type_32.loc18 [template = i32]
// CHECK:STDOUT:   %.loc18_9.2: type = converted %int.make_type_32.loc18, %.loc18_9.1 [template = i32]
// CHECK:STDOUT:   %.loc18_15: type = array_type %.loc18_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc20: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc20_29.1: %.16 = tuple_literal (%int.make_type_32.loc20, String, String)
// CHECK:STDOUT:   %.loc20_29.2: type = value_of_initializer %int.make_type_32.loc20 [template = i32]
// CHECK:STDOUT:   %.loc20_29.3: type = converted %int.make_type_32.loc20, %.loc20_29.2 [template = i32]
// CHECK:STDOUT:   %.loc20_29.4: type = converted %.loc20_29.1, constants.%.9 [template = constants.%.9]
// CHECK:STDOUT:   %t1.var: ref %.9 = var t1
// CHECK:STDOUT:   %t1: ref %.9 = bind_name t1, %t1.var
// CHECK:STDOUT:   %int.make_type_32.loc28: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc28_14: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc28_9.1: type = value_of_initializer %int.make_type_32.loc28 [template = i32]
// CHECK:STDOUT:   %.loc28_9.2: type = converted %int.make_type_32.loc28, %.loc28_9.1 [template = i32]
// CHECK:STDOUT:   %.loc28_15: type = array_type %.loc28_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %b.var: ref %.3 = var b
// CHECK:STDOUT:   %b: ref %.3 = bind_name b, %b.var
// CHECK:STDOUT:   %int.make_type_32.loc34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc34_14: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc34_9.1: type = value_of_initializer %int.make_type_32.loc34 [template = i32]
// CHECK:STDOUT:   %.loc34_9.2: type = converted %int.make_type_32.loc34, %.loc34_9.1 [template = i32]
// CHECK:STDOUT:   %.loc34_15: type = array_type %.loc34_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref %.3 = var c
// CHECK:STDOUT:   %c: ref %.3 = bind_name c, %c.var
// CHECK:STDOUT:   %int.make_type_32.loc36_10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc36_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc36_18.1: %.21 = tuple_literal (%int.make_type_32.loc36_10, %int.make_type_32.loc36_15)
// CHECK:STDOUT:   %.loc36_18.2: type = value_of_initializer %int.make_type_32.loc36_10 [template = i32]
// CHECK:STDOUT:   %.loc36_18.3: type = converted %int.make_type_32.loc36_10, %.loc36_18.2 [template = i32]
// CHECK:STDOUT:   %.loc36_18.4: type = value_of_initializer %int.make_type_32.loc36_15 [template = i32]
// CHECK:STDOUT:   %.loc36_18.5: type = converted %int.make_type_32.loc36_15, %.loc36_18.4 [template = i32]
// CHECK:STDOUT:   %.loc36_18.6: type = converted %.loc36_18.1, constants.%.20 [template = constants.%.20]
// CHECK:STDOUT:   %t2.var: ref %.20 = var t2
// CHECK:STDOUT:   %t2: ref %.20 = bind_name t2, %t2.var
// CHECK:STDOUT:   %int.make_type_32.loc40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc40_14: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc40_9.1: type = value_of_initializer %int.make_type_32.loc40 [template = i32]
// CHECK:STDOUT:   %.loc40_9.2: type = converted %int.make_type_32.loc40, %.loc40_9.1 [template = i32]
// CHECK:STDOUT:   %.loc40_15: type = array_type %.loc40_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %d.var: ref %.3 = var d
// CHECK:STDOUT:   %d: ref %.3 = bind_name d, %d.var
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
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.11) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
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
// CHECK:STDOUT:   %.loc18_20: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc18_23: String = string_literal "Hello" [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_32: String = string_literal "World" [template = constants.%.8]
// CHECK:STDOUT:   %.loc18_39.1: %.9 = tuple_literal (%.loc18_20, %.loc18_23, %.loc18_32)
// CHECK:STDOUT:   %.loc18_39.2: i32 = int_literal 0 [template = constants.%.10]
// CHECK:STDOUT:   %.loc18_39.3: ref i32 = array_index file.%a.var, %.loc18_39.2
// CHECK:STDOUT:   %.loc18_39.4: init i32 = initialize_from %.loc18_20 to %.loc18_39.3 [template = constants.%.5]
// CHECK:STDOUT:   %ImplicitAs.type.loc18: type = interface_type @ImplicitAs, @ImplicitAs(i32) [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %.loc18_39.5: %.13 = specific_constant imports.%import_ref.4, @ImplicitAs(i32) [template = constants.%.14]
// CHECK:STDOUT:   %Convert.ref.loc18: %.13 = name_ref Convert, %.loc18_39.5 [template = constants.%.14]
// CHECK:STDOUT:   %.loc18_39.6: i32 = converted %.loc18_23, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   %t1.ref: ref %.9 = name_ref t1, file.%t1
// CHECK:STDOUT:   %.loc28_19.1: ref i32 = tuple_access %t1.ref, element0
// CHECK:STDOUT:   %.loc28_19.2: i32 = bind_value %.loc28_19.1
// CHECK:STDOUT:   %.loc28_19.3: i32 = int_literal 0 [template = constants.%.10]
// CHECK:STDOUT:   %.loc28_19.4: ref i32 = array_index file.%b.var, %.loc28_19.3
// CHECK:STDOUT:   %.loc28_19.5: init i32 = initialize_from %.loc28_19.2 to %.loc28_19.4
// CHECK:STDOUT:   %.loc28_19.6: ref String = tuple_access %t1.ref, element1
// CHECK:STDOUT:   %ImplicitAs.type.loc28: type = interface_type @ImplicitAs, @ImplicitAs(i32) [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %.loc28_19.7: %.13 = specific_constant imports.%import_ref.4, @ImplicitAs(i32) [template = constants.%.14]
// CHECK:STDOUT:   %Convert.ref.loc28: %.13 = name_ref Convert, %.loc28_19.7 [template = constants.%.14]
// CHECK:STDOUT:   %.loc28_19.8: i32 = converted %.loc28_19.6, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   %.loc34_20: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc34_23: i32 = int_literal 2 [template = constants.%.19]
// CHECK:STDOUT:   %.loc34_24: %.20 = tuple_literal (%.loc34_20, %.loc34_23)
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   %t2.ref: ref %.20 = name_ref t2, file.%t2
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   return
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
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %Dest => i32
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
