// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_invalid_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_invalid_base.carbon

namespace N;
// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+4]]:14: error: expression cannot be used as a value [UseOfNonExprAsValue]
// CHECK:STDERR: var a: i32 = N[0];
// CHECK:STDERR:              ^
// CHECK:STDERR:
var a: i32 = N[0];

fn F();
// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+4]]:14: error: expression cannot be used as a value [UseOfNonExprAsValue]
// CHECK:STDERR: var b: i32 = F[1];
// CHECK:STDERR:              ^
// CHECK:STDERR:
var b: i32 = F[1];

// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+4]]:14: error: type `{.a: i32, .b: i32}` does not support indexing [TypeNotIndexable]
// CHECK:STDERR: var c: i32 = {.a = 1, .b = 2}[0];
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c: i32 = {.a = 1, .b = 2}[0];

// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+3]]:14: error: type `type` does not support indexing [TypeNotIndexable]
// CHECK:STDERR: var d: i32 = {.a: i32, .b: i32}[0];
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~
var d: i32 = {.a: i32, .b: i32}[0];

// CHECK:STDOUT: --- fail_invalid_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 0 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %struct: %.4 = struct_value (%.2, %.3) [template]
// CHECK:STDOUT:   %IndexWith.type.1: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %IndexWith: %IndexWith.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.2: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic]
// CHECK:STDOUT:   %Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2) = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %ElementType.patt: type = symbolic_binding_pattern ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %At.type: type = fn_type @At, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %At: %At.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %IndexWith.type.2, %At.type [symbolic]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .IndexWith = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %IndexWith.type.1 = import_ref Core//prelude/operators/index, inst+13, loaded [template = constants.%IndexWith]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/index, inst+22, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/index, inst+50, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/index, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/index, inst+42, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %int.make_type_32.loc16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_8.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:   %.loc16_8.2: type = converted %int.make_type_32.loc16, %.loc16_8.1 [template = i32]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %int.make_type_32.loc23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_8.1: type = value_of_initializer %int.make_type_32.loc23 [template = i32]
// CHECK:STDOUT:   %.loc23_8.2: type = converted %int.make_type_32.loc23, %.loc23_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int.make_type_32.loc29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc29_8.1: type = value_of_initializer %int.make_type_32.loc29 [template = i32]
// CHECK:STDOUT:   %.loc29_8.2: type = converted %int.make_type_32.loc29, %.loc29_8.1 [template = i32]
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT:   %int.make_type_32.loc34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc34_8.1: type = value_of_initializer %int.make_type_32.loc34 [template = i32]
// CHECK:STDOUT:   %.loc34_8.2: type = converted %int.make_type_32.loc34, %.loc34_8.1 [template = i32]
// CHECK:STDOUT:   %d.var: ref i32 = var d
// CHECK:STDOUT:   %d: ref i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @IndexWith(constants.%SubscriptType: type, constants.%ElementType: type) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic = %SubscriptType.patt (constants.%SubscriptType.patt)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType, 1 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:   %ElementType.patt: type = symbolic_binding_pattern ElementType, 1 [symbolic = %ElementType.patt (constants.%ElementType.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %At.type: type = fn_type @At, @IndexWith(%SubscriptType, %ElementType) [symbolic = %At.type (constants.%At.type)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type) = struct_value () [symbolic = %At (constants.%At)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @IndexWith.%IndexWith.type (%IndexWith.type.2), @IndexWith.%At.type (%At.type) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %.2: @IndexWith.%.1 (%.6) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .At = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @At(constants.%SubscriptType: type, constants.%ElementType: type, constants.%Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2)) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType, 1 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @At.%Self (%Self.2)](%subscript.param_patt: @At.%SubscriptType (%SubscriptType)) -> @At.%ElementType (%ElementType);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %N.ref: <namespace> = name_ref N, file.%N [template = file.%N]
// CHECK:STDOUT:   %.loc16: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc23: i32 = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   %.loc29_20: i32 = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc29_28: i32 = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc29_29.1: %.4 = struct_literal (%.loc29_20, %.loc29_28)
// CHECK:STDOUT:   %.loc29_31: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %struct: %.4 = struct_value (%.loc29_20, %.loc29_28) [template = constants.%struct]
// CHECK:STDOUT:   %.loc29_29.2: %.4 = converted %.loc29_29.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   %int.make_type_32.loc34_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc34_19.1: type = value_of_initializer %int.make_type_32.loc34_19 [template = i32]
// CHECK:STDOUT:   %.loc34_19.2: type = converted %int.make_type_32.loc34_19, %.loc34_19.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc34_28: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc34_28.1: type = value_of_initializer %int.make_type_32.loc34_28 [template = i32]
// CHECK:STDOUT:   %.loc34_28.2: type = converted %int.make_type_32.loc34_28, %.loc34_28.1 [template = i32]
// CHECK:STDOUT:   %.loc34_31: type = struct_type {.a: i32, .b: i32} [template = constants.%.4]
// CHECK:STDOUT:   %.loc34_33: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
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
