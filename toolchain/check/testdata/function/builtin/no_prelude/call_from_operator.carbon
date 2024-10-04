// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/no_prelude/call_from_operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/no_prelude/call_from_operator.carbon

// --- core.carbon

package Core;

fn Int32() -> type = "int.make_type_32";

interface Add {
  fn Op[self: Self](other: Self) -> Self;
}

// --- user.carbon

import Core;

// TODO: This should be in `Core`, but currently impl lookup only looks in the
// current file.
impl i32 as Core.Add {
  fn Op[self: Self](other: Self) -> Self = "int.sadd";
}

var arr: [i32; 1 + 2] = (3, 4, 3 + 4);

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Add.type: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op [template]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %Add.type, %Op.type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @Add.%Op.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int32 = %Int32.decl
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32.decl: %Int32.type = fn_decl @Int32 [template = constants.%Int32] {} {
// CHECK:STDOUT:     %return: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.decl: type = interface_decl @Add [template = constants.%Add.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Op.decl: %Op.type = fn_decl @Op [template = constants.%Op] {
// CHECK:STDOUT:     %self.patt: @Op.%Self (%Self) = binding_pattern self
// CHECK:STDOUT:     %other.patt: @Op.%Self (%Self) = binding_pattern other
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc7_15: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc7_15.1: type = facet_type_access %Self.ref.loc7_15 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc7_15.2: type = converted %Self.ref.loc7_15, %.loc7_15.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %self.param: @Op.%Self (%Self) = param self, runtime_param0
// CHECK:STDOUT:     %self: @Op.%Self (%Self) = bind_name self, %self.param
// CHECK:STDOUT:     %Self.ref.loc7_28: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc7_28.1: type = facet_type_access %Self.ref.loc7_28 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc7_28.2: type = converted %Self.ref.loc7_28, %.loc7_28.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %other.param: @Op.%Self (%Self) = param other, runtime_param1
// CHECK:STDOUT:     %other: @Op.%Self (%Self) = bind_name other, %other.param
// CHECK:STDOUT:     %Self.ref.loc7_37: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc7_37.1: type = facet_type_access %Self.ref.loc7_37 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc7_37.2: type = converted %Self.ref.loc7_37, %.loc7_37.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %return: ref @Op.%Self (%Self) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: %.2 = assoc_entity element0, %Op.decl [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc7
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(@Add.%Self: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Op.%Self (%Self)](%other: @Op.%Self (%Self)) -> @Op.%Self (%Self);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Add.type: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Op.1) [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %Add.type, %Op.type.2 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.7: <bound method> = bound_method %.3, %Op.1 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.9: type = array_type %.8, i32 [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.9 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.12: <bound method> = bound_method %.8, %Op.1 [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.14: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array: %.9 = tuple_value (%.8, %.11, %.13) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .Add = %import_ref.2
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//default, inst+2, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Core//default, inst+6, loaded [template = constants.%Add.type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//default, inst+8, unloaded
// CHECK:STDOUT:   %import_ref.4: %.5 = import_ref Core//default, inst+31, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.5: %Op.type.2 = import_ref Core//default, inst+26, loaded [template = constants.%Op.2]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//default, inst+26, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_6.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc6_6.2: type = converted %int.make_type_32, %.loc6_6.1 [template = i32]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, imports.%import_ref.2 [template = constants.%Add.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc10_16: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_20: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %Op.ref: %.5 = name_ref Op, imports.%import_ref.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc10_18.1: %Op.type.2 = interface_witness_access constants.%.2, element0 [template = constants.%Op.1]
// CHECK:STDOUT:   %.loc10_18.2: <bound method> = bound_method %.loc10_16, %.loc10_18.1 [template = constants.%.7]
// CHECK:STDOUT:   %int.sadd: init i32 = call %.loc10_18.2(%.loc10_16, %.loc10_20) [template = constants.%.8]
// CHECK:STDOUT:   %.loc10_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc10_11.2: type = converted %int.make_type_32, %.loc10_11.1 [template = i32]
// CHECK:STDOUT:   %.loc10_21: type = array_type %int.sadd, i32 [template = constants.%.9]
// CHECK:STDOUT:   %arr.var: ref %.9 = var arr
// CHECK:STDOUT:   %arr: ref %.9 = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .Op = imports.%import_ref.4
// CHECK:STDOUT:   witness = (imports.%import_ref.5)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc6_6.2 as %Add.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.1 = fn_decl @Op.1 [template = constants.%Op.1] {
// CHECK:STDOUT:     %self.patt: i32 = binding_pattern self
// CHECK:STDOUT:     %other.patt: i32 = binding_pattern other
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc7_15: type = name_ref Self, @impl.%.loc6_6.2 [template = i32]
// CHECK:STDOUT:     %self.param: i32 = param self, runtime_param0
// CHECK:STDOUT:     %self: i32 = bind_name self, %self.param
// CHECK:STDOUT:     %Self.ref.loc7_28: type = name_ref Self, @impl.%.loc6_6.2 [template = i32]
// CHECK:STDOUT:     %other.param: i32 = param other, runtime_param1
// CHECK:STDOUT:     %other: i32 = bind_name other, %other.param
// CHECK:STDOUT:     %Self.ref.loc7_37: type = name_ref Self, @impl.%.loc6_6.2 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_22: <witness> = interface_witness (%Op.decl) [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.loc6_22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[%self: i32](%other: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(constants.%Self: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Op.2.%Self (%Self)](%other: @Op.2.%Self (%Self)) -> @Op.2.%Self (%Self);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_26: i32 = int_literal 3 [template = constants.%.8]
// CHECK:STDOUT:   %.loc10_29: i32 = int_literal 4 [template = constants.%.11]
// CHECK:STDOUT:   %.loc10_32: i32 = int_literal 3 [template = constants.%.8]
// CHECK:STDOUT:   %.loc10_36: i32 = int_literal 4 [template = constants.%.11]
// CHECK:STDOUT:   %Op.ref: %.5 = name_ref Op, imports.%import_ref.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc10_34.1: %Op.type.2 = interface_witness_access constants.%.2, element0 [template = constants.%Op.1]
// CHECK:STDOUT:   %.loc10_34.2: <bound method> = bound_method %.loc10_32, %.loc10_34.1 [template = constants.%.12]
// CHECK:STDOUT:   %int.sadd: init i32 = call %.loc10_34.2(%.loc10_32, %.loc10_36) [template = constants.%.13]
// CHECK:STDOUT:   %.loc10_37.1: %.14 = tuple_literal (%.loc10_26, %.loc10_29, %int.sadd)
// CHECK:STDOUT:   %.loc10_37.2: i32 = int_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %.loc10_37.3: ref i32 = array_index file.%arr.var, %.loc10_37.2
// CHECK:STDOUT:   %.loc10_37.4: init i32 = initialize_from %.loc10_26 to %.loc10_37.3 [template = constants.%.8]
// CHECK:STDOUT:   %.loc10_37.5: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_37.6: ref i32 = array_index file.%arr.var, %.loc10_37.5
// CHECK:STDOUT:   %.loc10_37.7: init i32 = initialize_from %.loc10_29 to %.loc10_37.6 [template = constants.%.11]
// CHECK:STDOUT:   %.loc10_37.8: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_37.9: ref i32 = array_index file.%arr.var, %.loc10_37.8
// CHECK:STDOUT:   %.loc10_37.10: init i32 = initialize_from %int.sadd to %.loc10_37.9 [template = constants.%.13]
// CHECK:STDOUT:   %.loc10_37.11: init %.9 = array_init (%.loc10_37.4, %.loc10_37.7, %.loc10_37.10) to file.%arr.var [template = constants.%array]
// CHECK:STDOUT:   %.loc10_38: init %.9 = converted %.loc10_37.1, %.loc10_37.11 [template = constants.%array]
// CHECK:STDOUT:   assign file.%arr.var, %.loc10_38
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(i32) {
// CHECK:STDOUT:   %Self => i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
