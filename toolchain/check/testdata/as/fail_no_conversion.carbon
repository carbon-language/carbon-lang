// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/fail_no_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/fail_no_conversion.carbon

// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+6]]:21: ERROR: Cannot convert from `i32` to `(i32, i32)` with `as`.
// CHECK:STDERR: let n: (i32, i32) = 1 as (i32, i32);
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+3]]:21: Type `i32` does not implement interface `As`.
// CHECK:STDERR: let n: (i32, i32) = 1 as (i32, i32);
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~
let n: (i32, i32) = 1 as (i32, i32);

// CHECK:STDOUT: --- fail_no_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %As.type: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic]
// CHECK:STDOUT:   %.6: type = interface_type @As, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @As.%.1 (%.6) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %.6 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.6, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %.9: type = interface_type @As, @As(%.3) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @As(%.3) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %.9, %Convert.type.2 [template]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.12: %.7 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .As = %import_ref.2
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
// CHECK:STDOUT:   %import_ref.2: %As.type = import_ref Core//prelude/operators/as, inst+4, loaded [template = constants.%As]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+10, unloaded
// CHECK:STDOUT:   %import_ref.4: @As.%.2 (%.7) = import_ref Core//prelude/operators/as, inst+27, loaded [symbolic = @As.%.3 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+20, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+20, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+20, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .n = @__global_init.%n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc17_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc17_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_17.1: %.2 = tuple_literal (%int.make_type_32.loc17_9, %int.make_type_32.loc17_14)
// CHECK:STDOUT:   %.loc17_17.2: type = value_of_initializer %int.make_type_32.loc17_9 [template = i32]
// CHECK:STDOUT:   %.loc17_17.3: type = converted %int.make_type_32.loc17_9, %.loc17_17.2 [template = i32]
// CHECK:STDOUT:   %.loc17_17.4: type = value_of_initializer %int.make_type_32.loc17_14 [template = i32]
// CHECK:STDOUT:   %.loc17_17.5: type = converted %int.make_type_32.loc17_14, %.loc17_17.4 [template = i32]
// CHECK:STDOUT:   %.loc17_17.6: type = converted %.loc17_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @As, @As(%Dest) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self: %.6 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @As.%.1 (%.6), @As.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.7)]
// CHECK:STDOUT:   %.3: @As.%.2 (%.7) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.3 (constants.%.8)]
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
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @As.%.1 (%.6)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @As, @As(%Dest) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self: %.6 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_21: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %int.make_type_32.loc17_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc17_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_35.1: %.2 = tuple_literal (%int.make_type_32.loc17_27, %int.make_type_32.loc17_32)
// CHECK:STDOUT:   %.loc17_35.2: type = value_of_initializer %int.make_type_32.loc17_27 [template = i32]
// CHECK:STDOUT:   %.loc17_35.3: type = converted %int.make_type_32.loc17_27, %.loc17_35.2 [template = i32]
// CHECK:STDOUT:   %.loc17_35.4: type = value_of_initializer %int.make_type_32.loc17_32 [template = i32]
// CHECK:STDOUT:   %.loc17_35.5: type = converted %int.make_type_32.loc17_32, %.loc17_35.4 [template = i32]
// CHECK:STDOUT:   %.loc17_35.6: type = converted %.loc17_35.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc17_23.1: type = interface_type @As, @As(constants.%.3) [template = constants.%.9]
// CHECK:STDOUT:   %.loc17_23.2: %.10 = specific_constant imports.%import_ref.4, @As(constants.%.3) [template = constants.%.11]
// CHECK:STDOUT:   %Convert.ref: %.10 = name_ref Convert, %.loc17_23.2 [template = constants.%.11]
// CHECK:STDOUT:   %.loc17_23.3: %.3 = converted %.loc17_21, <error> [template = <error>]
// CHECK:STDOUT:   %.loc17_23.4: i32 = tuple_access %.loc17_23.3, element0 [template = <error>]
// CHECK:STDOUT:   %n: %.3 = bind_name n, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@As.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.3) {
// CHECK:STDOUT:   %Dest => constants.%.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.2 => constants.%.10
// CHECK:STDOUT:   %.3 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
