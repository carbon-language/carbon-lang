// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/method.carbon

interface I {
  fn F[self: Self](other: Self) -> Self;
}

impl i32 as I {
  fn F[self: i32](other: i32) -> i32 = "int.sadd";
}

var arr: [i32; 1.(I.F)(2)];

// CHECK:STDOUT: --- method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %I.type, %F.type.1 [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: <bound method> = bound_method %.5, %F.2 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.9: type = array_type %.8, i32 [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.9 [template]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_6.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc15_6.2: type = converted %int.make_type_32, %.loc15_6.1 [template = i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc19_16: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %I.ref: type = name_ref I, %I.decl [template = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %.2 = name_ref F, @I.%.loc12 [template = constants.%.3]
// CHECK:STDOUT:   %.loc19_17.1: %F.type.1 = interface_witness_access constants.%.4, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %.loc19_17.2: <bound method> = bound_method %.loc19_16, %.loc19_17.1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc19_24: i32 = int_literal 2 [template = constants.%.7]
// CHECK:STDOUT:   %int.sadd: init i32 = call %.loc19_17.2(%.loc19_16, %.loc19_24) [template = constants.%.8]
// CHECK:STDOUT:   %.loc19_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc19_11.2: type = converted %int.make_type_32, %.loc19_11.1 [template = i32]
// CHECK:STDOUT:   %.loc19_26: type = array_type %int.sadd, i32 [template = constants.%.9]
// CHECK:STDOUT:   %arr.var: ref %.9 = var arr
// CHECK:STDOUT:   %arr: ref %.9 = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self (%Self) = binding_pattern self
// CHECK:STDOUT:     %other.patt: @F.1.%Self (%Self) = binding_pattern other
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc12_14: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc12_14.1: type = facet_type_access %Self.ref.loc12_14 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc12_14.2: type = converted %Self.ref.loc12_14, %.loc12_14.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %self.param: @F.1.%Self (%Self) = param self, runtime_param0
// CHECK:STDOUT:     %self: @F.1.%Self (%Self) = bind_name self, %self.param
// CHECK:STDOUT:     %Self.ref.loc12_27: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc12_27.1: type = facet_type_access %Self.ref.loc12_27 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc12_27.2: type = converted %Self.ref.loc12_27, %.loc12_27.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %other.param: @F.1.%Self (%Self) = param other, runtime_param1
// CHECK:STDOUT:     %other: @F.1.%Self (%Self) = bind_name other, %other.param
// CHECK:STDOUT:     %Self.ref.loc12_36: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc12_36.1: type = facet_type_access %Self.ref.loc12_36 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc12_36.2: type = converted %Self.ref.loc12_36, %.loc12_36.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %return: ref @F.1.%Self (%Self) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: %.2 = assoc_entity element0, %F.decl [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc12
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc15_6.2 as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: i32 = binding_pattern self
// CHECK:STDOUT:     %other.patt: i32 = binding_pattern other
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc16_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_14.1: type = value_of_initializer %int.make_type_32.loc16_14 [template = i32]
// CHECK:STDOUT:     %.loc16_14.2: type = converted %int.make_type_32.loc16_14, %.loc16_14.1 [template = i32]
// CHECK:STDOUT:     %self.param: i32 = param self, runtime_param0
// CHECK:STDOUT:     %self: i32 = bind_name self, %self.param
// CHECK:STDOUT:     %int.make_type_32.loc16_26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_26.1: type = value_of_initializer %int.make_type_32.loc16_26 [template = i32]
// CHECK:STDOUT:     %.loc16_26.2: type = converted %int.make_type_32.loc16_26, %.loc16_26.1 [template = i32]
// CHECK:STDOUT:     %other.param: i32 = param other, runtime_param1
// CHECK:STDOUT:     %other: i32 = bind_name other, %other.param
// CHECK:STDOUT:     %int.make_type_32.loc16_34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_34.1: type = value_of_initializer %int.make_type_32.loc16_34 [template = i32]
// CHECK:STDOUT:     %.loc16_34.2: type = converted %int.make_type_32.loc16_34, %.loc16_34.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_15: <witness> = interface_witness (%F.decl) [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc15_15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @F.1.%Self (%Self)](%other: @F.1.%Self (%Self)) -> @F.1.%Self (%Self);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self: i32](%other: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(i32) {
// CHECK:STDOUT:   %Self => i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
