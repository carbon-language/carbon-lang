// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/fail_non_tuple_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/fail_non_tuple_access.carbon

fn Main() {
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+4]]:3: error: type `i32` does not support indexing [TypeNotIndexable]
  // CHECK:STDERR:   0[1];
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  0[1];

  var non_tuple: [i32; 2] = (5, 5);
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+3]]:20: error: type `[i32; 2]` does not support tuple indexing; only tuples can be indexed that way [TupleIndexOnANonTupleType]
  // CHECK:STDERR:   var first: i32 = non_tuple.0;
  // CHECK:STDERR:                    ^~~~~~~~~~~
  var first: i32 = non_tuple.0;
}

// CHECK:STDOUT: --- fail_non_tuple_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %IndexWith.type.1: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %IndexWith: %IndexWith.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.2: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2) = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %ElementType.patt: type = symbolic_binding_pattern ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %At.type: type = fn_type @At, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %At: %At.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %IndexWith.type.2, %At.type [symbolic]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: type = array_type %.6, i32 [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.10: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %array: %.7 = tuple_value (%.9, %.9) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IndexWith = %import_ref.1
// CHECK:STDOUT:     .Int32 = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IndexWith.type.1 = import_ref Core//prelude/operators/index, inst+13, loaded [template = constants.%IndexWith]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/index, inst+22, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/index, inst+50, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/index, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/index, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @IndexWith(constants.%SubscriptType: type, constants.%ElementType: type) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic = %SubscriptType.patt (constants.%SubscriptType.patt)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType, 1 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:   %ElementType.patt: type = symbolic_binding_pattern ElementType, 1 [symbolic = %ElementType.patt (constants.%ElementType.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %At.type: type = fn_type @At, @IndexWith(%SubscriptType, %ElementType) [symbolic = %At.type (constants.%At.type)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type) = struct_value () [symbolic = %At (constants.%At)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @IndexWith.%IndexWith.type (%IndexWith.type.2), @IndexWith.%At.type (%At.type) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %.2: @IndexWith.%.1 (%.4) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:     .At = imports.%import_ref.3
// CHECK:STDOUT:     witness = (imports.%import_ref.4)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_3: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_5: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_32.loc18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_24: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc18_19.1: type = value_of_initializer %int.make_type_32.loc18 [template = i32]
// CHECK:STDOUT:   %.loc18_19.2: type = converted %int.make_type_32.loc18, %.loc18_19.1 [template = i32]
// CHECK:STDOUT:   %.loc18_25: type = array_type %.loc18_24, i32 [template = constants.%.7]
// CHECK:STDOUT:   %non_tuple.var: ref %.7 = var non_tuple
// CHECK:STDOUT:   %non_tuple: ref %.7 = bind_name non_tuple, %non_tuple.var
// CHECK:STDOUT:   %.loc18_30: i32 = int_literal 5 [template = constants.%.9]
// CHECK:STDOUT:   %.loc18_33: i32 = int_literal 5 [template = constants.%.9]
// CHECK:STDOUT:   %.loc18_34.1: %.10 = tuple_literal (%.loc18_30, %.loc18_33)
// CHECK:STDOUT:   %.loc18_34.2: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_34.3: ref i32 = array_index %non_tuple.var, %.loc18_34.2
// CHECK:STDOUT:   %.loc18_34.4: init i32 = initialize_from %.loc18_30 to %.loc18_34.3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc18_34.5: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc18_34.6: ref i32 = array_index %non_tuple.var, %.loc18_34.5
// CHECK:STDOUT:   %.loc18_34.7: init i32 = initialize_from %.loc18_33 to %.loc18_34.6 [template = constants.%.9]
// CHECK:STDOUT:   %.loc18_34.8: init %.7 = array_init (%.loc18_34.4, %.loc18_34.7) to %non_tuple.var [template = constants.%array]
// CHECK:STDOUT:   %.loc18_35: init %.7 = converted %.loc18_34.1, %.loc18_34.8 [template = constants.%array]
// CHECK:STDOUT:   assign %non_tuple.var, %.loc18_35
// CHECK:STDOUT:   %int.make_type_32.loc22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc22_14.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:   %.loc22_14.2: type = converted %int.make_type_32.loc22, %.loc22_14.1 [template = i32]
// CHECK:STDOUT:   %first.var: ref i32 = var first
// CHECK:STDOUT:   %first: ref i32 = bind_name first, %first.var
// CHECK:STDOUT:   %non_tuple.ref: ref %.7 = name_ref non_tuple, %non_tuple
// CHECK:STDOUT:   %.loc22_30: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   assign %first.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @At(constants.%SubscriptType: type, constants.%ElementType: type, constants.%Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2)) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType, 1 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:   %IndexWith.type: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @At.%Self (%Self.2)](%subscript.param_patt: @At.%SubscriptType (%SubscriptType)) -> @At.%ElementType (%ElementType);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType, constants.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %SubscriptType.patt => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT:   %ElementType.patt => constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@IndexWith.%SubscriptType, @IndexWith.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %SubscriptType.patt => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT:   %ElementType.patt => constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@At.%SubscriptType, @At.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %SubscriptType.patt => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT:   %ElementType.patt => constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At(constants.%SubscriptType, constants.%ElementType, constants.%Self.1) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
