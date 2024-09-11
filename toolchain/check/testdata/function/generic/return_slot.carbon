// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/return_slot.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/return_slot.carbon

class Wrap(T:! type) {
  fn Make() -> T;
}

class C { var arr: [i32; 100]; }

fn G() {
  var a: i32 = Wrap(i32).Make();
  var b: () = Wrap(()).Make();
  var c: C = Wrap(C).Make();
}

// CHECK:STDOUT: --- return_slot.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Wrap.type: type = generic_class_type @Wrap [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Wrap.1: %Wrap.type = struct_value () [template]
// CHECK:STDOUT:   %Wrap.2: type = class_type @Wrap, @Wrap(%T) [symbolic]
// CHECK:STDOUT:   %Make.type.1: type = fn_type @Make, @Wrap(%T) [symbolic]
// CHECK:STDOUT:   %Make.1: %Make.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 100 [template]
// CHECK:STDOUT:   %.4: type = array_type %.3, i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %C, %.4 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.arr: %.4} [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Wrap.3: type = class_type @Wrap, @Wrap(i32) [template]
// CHECK:STDOUT:   %Make.type.2: type = fn_type @Make, @Wrap(i32) [template]
// CHECK:STDOUT:   %Make.2: %Make.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %Wrap.4: type = class_type @Wrap, @Wrap(%.1) [template]
// CHECK:STDOUT:   %Make.type.3: type = fn_type @Make, @Wrap(%.1) [template]
// CHECK:STDOUT:   %Make.3: %Make.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.9: type = struct_type {.arr: %.5} [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %Wrap.5: type = class_type @Wrap, @Wrap(%C) [template]
// CHECK:STDOUT:   %Make.type.4: type = fn_type @Make, @Wrap(%C) [template]
// CHECK:STDOUT:   %Make.4: %Make.type.4 = struct_value () [template]
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
// CHECK:STDOUT:     .Wrap = %Wrap.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Wrap.decl: %Wrap.type = class_decl @Wrap [template = constants.%Wrap.1] {
// CHECK:STDOUT:     %T.loc11_12.1: type = param T
// CHECK:STDOUT:     %T.loc11_12.2: type = bind_symbolic_name T 0, %T.loc11_12.1 [symbolic = @Wrap.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Wrap(file.%T.loc11_12.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Wrap(%T) [symbolic = %Make.type (constants.%Make.type.1)]
// CHECK:STDOUT:   %Make: @Wrap.%Make.type (%Make.type.1) = struct_value () [symbolic = %Make (constants.%Make.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Make.decl: @Wrap.%Make.type (%Make.type.1) = fn_decl @Make [symbolic = %Make (constants.%Make.1)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, file.%T.loc11_12.2 [symbolic = @Make.%T (constants.%T)]
// CHECK:STDOUT:       %return.var: ref @Make.%T (%T) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Wrap.2
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_26: i32 = int_literal 100 [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_21.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc15_21.2: type = converted %int.make_type_32, %.loc15_21.1 [template = i32]
// CHECK:STDOUT:   %.loc15_29: type = array_type %.loc15_26, i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc15_18: %.6 = field_decl arr, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .arr = %.loc15_18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(file.%T.loc11_12.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc18_10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_10.1: type = value_of_initializer %int.make_type_32.loc18_10 [template = i32]
// CHECK:STDOUT:   %.loc18_10.2: type = converted %int.make_type_32.loc18_10, %.loc18_10.1 [template = i32]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %Wrap.ref.loc18: %Wrap.type = name_ref Wrap, file.%Wrap.decl [template = constants.%Wrap.1]
// CHECK:STDOUT:   %int.make_type_32.loc18_21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_20.1: type = value_of_initializer %int.make_type_32.loc18_21 [template = i32]
// CHECK:STDOUT:   %.loc18_20.2: type = converted %int.make_type_32.loc18_21, %.loc18_20.1 [template = i32]
// CHECK:STDOUT:   %.loc18_20.3: init type = call %Wrap.ref.loc18(%.loc18_20.2) [template = constants.%Wrap.3]
// CHECK:STDOUT:   %.loc18_25: %Make.type.2 = specific_constant @Wrap.%Make.decl, @Wrap(i32) [template = constants.%Make.2]
// CHECK:STDOUT:   %Make.ref.loc18: %Make.type.2 = name_ref Make, %.loc18_25 [template = constants.%Make.2]
// CHECK:STDOUT:   %Make.call.loc18: init i32 = call %Make.ref.loc18()
// CHECK:STDOUT:   assign %a.var, %Make.call.loc18
// CHECK:STDOUT:   %.loc19_11.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc19_11.2: type = converted %.loc19_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %b.var: ref %.1 = var b
// CHECK:STDOUT:   %b: ref %.1 = bind_name b, %b.var
// CHECK:STDOUT:   %Wrap.ref.loc19: %Wrap.type = name_ref Wrap, file.%Wrap.decl [template = constants.%Wrap.1]
// CHECK:STDOUT:   %.loc19_21: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc19_19.1: type = converted %.loc19_21, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc19_19.2: init type = call %Wrap.ref.loc19(%.loc19_19.1) [template = constants.%Wrap.4]
// CHECK:STDOUT:   %.loc19_23: %Make.type.3 = specific_constant @Wrap.%Make.decl, @Wrap(constants.%.1) [template = constants.%Make.3]
// CHECK:STDOUT:   %Make.ref.loc19: %Make.type.3 = name_ref Make, %.loc19_23 [template = constants.%Make.3]
// CHECK:STDOUT:   %Make.call.loc19: init %.1 = call %Make.ref.loc19()
// CHECK:STDOUT:   assign %b.var, %Make.call.loc19
// CHECK:STDOUT:   %C.ref.loc20_10: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   %Wrap.ref.loc20: %Wrap.type = name_ref Wrap, file.%Wrap.decl [template = constants.%Wrap.1]
// CHECK:STDOUT:   %C.ref.loc20_19: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc20_18: init type = call %Wrap.ref.loc20(%C.ref.loc20_19) [template = constants.%Wrap.5]
// CHECK:STDOUT:   %.loc20_21: %Make.type.4 = specific_constant @Wrap.%Make.decl, @Wrap(constants.%C) [template = constants.%Make.4]
// CHECK:STDOUT:   %Make.ref.loc20: %Make.type.4 = name_ref Make, %.loc20_21 [template = constants.%Make.4]
// CHECK:STDOUT:   %.loc20_7: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %Make.call.loc20: init %C = call %Make.ref.loc20() to %.loc20_7
// CHECK:STDOUT:   assign %c.var, %Make.call.loc20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(@Wrap.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.2
// CHECK:STDOUT:   %Make => constants.%Make.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%.1) {
// CHECK:STDOUT:   %T => constants.%.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.3
// CHECK:STDOUT:   %Make => constants.%Make.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%.1) {
// CHECK:STDOUT:   %T => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.4
// CHECK:STDOUT:   %Make => constants.%Make.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
