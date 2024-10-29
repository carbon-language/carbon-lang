// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/index.carbon

// --- overloaded_index.carbon

library "[[@TEST_NAME]]";

class C {}
class ElementType {}
class SubscriptType {}

impl C as Core.IndexWith(SubscriptType, ElementType) {
  fn At[self: Self](subscript: SubscriptType) -> ElementType {
    return {};
  }
}

let s: SubscriptType = {};
let c: C = {};
let x: ElementType = c[s];

// --- overloaded_builtin.carbon

library "[[@TEST_NAME]]";

impl (i32, i32) as Core.IndexWith(i32, i32) {
  fn At[self: Self](subscript: i32) -> i32 {
    return self.0;
  }
}

let s: (i32, i32) = (1, 5);
let e: i32 = s[0];

// --- fail_invalid_subscript_type.carbon

library "[[@TEST_NAME]]";

class C { }
class ElementType {}
class SubscriptType {}

impl C as Core.IndexWith(SubscriptType, ElementType) {
  fn At[self: Self](subscript: SubscriptType) -> ElementType {
    return {};
  }
}

let c: C = {};
// CHECK:STDERR: fail_invalid_subscript_type.carbon:[[@LINE+7]]:22: error: cannot implicitly convert from `i32` to `SubscriptType` [ImplicitAsConversionFailure]
// CHECK:STDERR: let x: ElementType = c[0];
// CHECK:STDERR:                      ^~~~
// CHECK:STDERR: fail_invalid_subscript_type.carbon:[[@LINE+4]]:22: note: type `i32` does not implement interface `ImplicitAs` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let x: ElementType = c[0];
// CHECK:STDERR:                      ^~~~
// CHECK:STDERR:
let x: ElementType = c[0];

// --- fail_index_with_not_implemented.carbon

library "[[@TEST_NAME]]";

class C { }

let c: C = {};
// CHECK:STDERR: fail_index_with_not_implemented.carbon:[[@LINE+3]]:14: error: type `C` does not support indexing [TypeNotIndexable]
// CHECK:STDERR: let x: i32 = c[0];
// CHECK:STDERR:              ^~~~
let x: i32 = c[0];


// CHECK:STDOUT: --- overloaded_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %ElementType.1: type = class_type @ElementType [template]
// CHECK:STDOUT:   %SubscriptType.1: type = class_type @SubscriptType [template]
// CHECK:STDOUT:   %IndexWith.type.1: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %IndexWith: %IndexWith.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ElementType.2: type = bind_symbolic_name ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.2: type = bind_symbolic_name SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.2: type = interface_type @IndexWith, @IndexWith(%SubscriptType.2, %ElementType.2) [symbolic]
// CHECK:STDOUT:   %Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2) = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %ElementType.patt: type = symbolic_binding_pattern ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %At.type.1: type = fn_type @At.1, @IndexWith(%SubscriptType.2, %ElementType.2) [symbolic]
// CHECK:STDOUT:   %At.1: %At.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %IndexWith.type.2, %At.type.1 [symbolic]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.3: type = interface_type @IndexWith, @IndexWith(%SubscriptType.1, %ElementType.1) [template]
// CHECK:STDOUT:   %At.type.2: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.2: %At.type.2 = struct_value () [template]
// CHECK:STDOUT:   %At.type.3: type = fn_type @At.1, @IndexWith(%SubscriptType.1, %ElementType.1) [template]
// CHECK:STDOUT:   %At.3: %At.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %IndexWith.type.3, %At.type.3 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%At.2) [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct.1: %ElementType.1 = struct_value () [template]
// CHECK:STDOUT:   %struct.2: %SubscriptType.1 = struct_value () [template]
// CHECK:STDOUT:   %struct.3: %C = struct_value () [template]
// CHECK:STDOUT:   %.10: %.4 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IndexWith = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/index
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IndexWith.type.1 = import_ref Core//prelude/operators/index, inst+13, loaded [template = constants.%IndexWith]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/index, inst+22, unloaded
// CHECK:STDOUT:   %import_ref.3: @IndexWith.%.1 (%.4) = import_ref Core//prelude/operators/index, inst+50, loaded [symbolic = @IndexWith.%.2 (constants.%.10)]
// CHECK:STDOUT:   %import_ref.4: @IndexWith.%At.type (%At.type.1) = import_ref Core//prelude/operators/index, inst+42, loaded [symbolic = @IndexWith.%At (constants.%At.1)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/index, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/index, inst+42, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .ElementType = %ElementType.decl
// CHECK:STDOUT:     .SubscriptType = %SubscriptType.decl
// CHECK:STDOUT:     .s = @__global_init.%s
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:     .x = @__global_init.%x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %ElementType.decl: type = class_decl @ElementType [template = constants.%ElementType.1] {} {}
// CHECK:STDOUT:   %SubscriptType.decl: type = class_decl @SubscriptType [template = constants.%SubscriptType.1] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.1 = name_ref IndexWith, imports.%import_ref.1 [template = constants.%IndexWith]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT:     %IndexWith.type: type = interface_type @IndexWith, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) [template = constants.%IndexWith.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SubscriptType.ref: type = name_ref SubscriptType, %SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %ElementType.ref: type = name_ref ElementType, %ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @IndexWith(constants.%SubscriptType.2: type, constants.%ElementType.2: type) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType.2)]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic = %SubscriptType.patt (constants.%SubscriptType.patt)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType, 1 [symbolic = %ElementType (constants.%ElementType.2)]
// CHECK:STDOUT:   %ElementType.patt: type = symbolic_binding_pattern ElementType, 1 [symbolic = %ElementType.patt (constants.%ElementType.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %At.type: type = fn_type @At.1, @IndexWith(%SubscriptType, %ElementType) [symbolic = %At.type (constants.%At.type.1)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type.1) = struct_value () [symbolic = %At (constants.%At.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @IndexWith.%IndexWith.type (%IndexWith.type.2), @IndexWith.%At.type (%At.type.1) [symbolic = %.1 (constants.%.4)]
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
// CHECK:STDOUT: impl @impl: %C.ref as %IndexWith.type {
// CHECK:STDOUT:   %At.decl: %At.type.2 = fn_decl @At.2 [template = constants.%At.2] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %subscript.patt: %SubscriptType.1 = binding_pattern subscript
// CHECK:STDOUT:     %subscript.param_patt: %SubscriptType.1 = value_param_pattern %subscript.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %ElementType.1 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ElementType.1 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%C.ref [template = constants.%C]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %subscript.param: %SubscriptType.1 = value_param runtime_param1
// CHECK:STDOUT:     %subscript: %SubscriptType.1 = bind_name subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %ElementType.1 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %ElementType.1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = interface_witness (%At.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ElementType {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ElementType.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType {
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SubscriptType.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @At.1(constants.%SubscriptType.2: type, constants.%ElementType.2: type, constants.%Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2)) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType.2)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType, 1 [symbolic = %ElementType (constants.%ElementType.2)]
// CHECK:STDOUT:   %IndexWith.type: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @At.1.%Self (%Self.2)](%subscript.param_patt: @At.1.%SubscriptType (%SubscriptType.2)) -> @At.1.%ElementType (%ElementType.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %C](%subscript.param_patt: %SubscriptType.1) -> %return: %ElementType.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc10_13.2: init %ElementType.1 = class_init (), %return [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc10_14: init %ElementType.1 = converted %.loc10_13.1, %.loc10_13.2 [template = constants.%struct.1]
// CHECK:STDOUT:   return %.loc10_14 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_25.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc14_25.2: ref %SubscriptType.1 = temporary_storage
// CHECK:STDOUT:   %.loc14_25.3: init %SubscriptType.1 = class_init (), %.loc14_25.2 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc14_25.4: ref %SubscriptType.1 = temporary %.loc14_25.2, %.loc14_25.3
// CHECK:STDOUT:   %.loc14_26.1: ref %SubscriptType.1 = converted %.loc14_25.1, %.loc14_25.4
// CHECK:STDOUT:   %.loc14_26.2: %SubscriptType.1 = bind_value %.loc14_26.1
// CHECK:STDOUT:   %s: %SubscriptType.1 = bind_name s, %.loc14_26.2
// CHECK:STDOUT:   %.loc15_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc15_13.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc15_13.3: init %C = class_init (), %.loc15_13.2 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc15_13.4: ref %C = temporary %.loc15_13.2, %.loc15_13.3
// CHECK:STDOUT:   %.loc15_14.1: ref %C = converted %.loc15_13.1, %.loc15_13.4
// CHECK:STDOUT:   %.loc15_14.2: %C = bind_value %.loc15_14.1
// CHECK:STDOUT:   %c: %C = bind_name c, %.loc15_14.2
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %s.ref: %SubscriptType.1 = name_ref s, %s
// CHECK:STDOUT:   %IndexWith.type: type = interface_type @IndexWith, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) [template = constants.%IndexWith.type.3]
// CHECK:STDOUT:   %.loc16_25.1: %.6 = specific_constant imports.%import_ref.3, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) [template = constants.%.7]
// CHECK:STDOUT:   %At.ref: %.6 = name_ref At, %.loc16_25.1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc16_25.2: %At.type.3 = interface_witness_access constants.%.8, element0 [template = constants.%At.2]
// CHECK:STDOUT:   %.loc16_25.3: <bound method> = bound_method %c.ref, %.loc16_25.2
// CHECK:STDOUT:   %.loc16_25.4: ref %ElementType.1 = temporary_storage
// CHECK:STDOUT:   %At.call: init %ElementType.1 = call %.loc16_25.3(%c.ref, %s.ref) to %.loc16_25.4
// CHECK:STDOUT:   %.loc16_25.5: ref %ElementType.1 = temporary %.loc16_25.4, %At.call
// CHECK:STDOUT:   %.loc16_25.6: %ElementType.1 = bind_value %.loc16_25.5
// CHECK:STDOUT:   %x: %ElementType.1 = bind_name x, %.loc16_25.6
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType.2, constants.%ElementType.2) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %SubscriptType.patt => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT:   %ElementType.patt => constants.%ElementType.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@IndexWith.%SubscriptType, @IndexWith.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %SubscriptType.patt => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT:   %ElementType.patt => constants.%ElementType.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@At.1.%SubscriptType, @At.1.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %SubscriptType.patt => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT:   %ElementType.patt => constants.%ElementType.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(constants.%SubscriptType.2, constants.%ElementType.2, constants.%Self.1) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.1
// CHECK:STDOUT:   %SubscriptType.patt => constants.%SubscriptType.1
// CHECK:STDOUT:   %ElementType => constants.%ElementType.1
// CHECK:STDOUT:   %ElementType.patt => constants.%ElementType.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %At.type => constants.%At.type.3
// CHECK:STDOUT:   %At => constants.%At.3
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(constants.%SubscriptType.1, constants.%ElementType.1, constants.%C) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.1
// CHECK:STDOUT:   %ElementType => constants.%ElementType.1
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.3
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overloaded_builtin.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %IndexWith.type.1: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %IndexWith: %IndexWith.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.2: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2) = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %ElementType.patt: type = symbolic_binding_pattern ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %At.type.1: type = fn_type @At.1, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %At.1: %At.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %IndexWith.type.2, %At.type.1 [symbolic]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.3: type = interface_type @IndexWith, @IndexWith(i32, i32) [template]
// CHECK:STDOUT:   %At.type.2: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.2: %At.type.2 = struct_value () [template]
// CHECK:STDOUT:   %At.type.3: type = fn_type @At.1, @IndexWith(i32, i32) [template]
// CHECK:STDOUT:   %At.3: %At.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %IndexWith.type.3, %At.type.3 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%At.2) [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.12: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %tuple: %.3 = tuple_value (%.11, %.12) [template]
// CHECK:STDOUT:   %.13: %.4 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .IndexWith = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/index
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %IndexWith.type.1 = import_ref Core//prelude/operators/index, inst+13, loaded [template = constants.%IndexWith]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/index, inst+22, unloaded
// CHECK:STDOUT:   %import_ref.4: @IndexWith.%.1 (%.4) = import_ref Core//prelude/operators/index, inst+50, loaded [symbolic = @IndexWith.%.2 (constants.%.13)]
// CHECK:STDOUT:   %import_ref.5: @IndexWith.%At.type (%At.type.1) = import_ref Core//prelude/operators/index, inst+42, loaded [symbolic = @IndexWith.%At (constants.%At.1)]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/index, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/index, inst+42, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .s = @__global_init.%s
// CHECK:STDOUT:     .e = @__global_init.%e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %int.make_type_32.loc4_7: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc4_12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_15.1: %.2 = tuple_literal (%int.make_type_32.loc4_7, %int.make_type_32.loc4_12)
// CHECK:STDOUT:     %.loc4_15.2: type = value_of_initializer %int.make_type_32.loc4_7 [template = i32]
// CHECK:STDOUT:     %.loc4_15.3: type = converted %int.make_type_32.loc4_7, %.loc4_15.2 [template = i32]
// CHECK:STDOUT:     %.loc4_15.4: type = value_of_initializer %int.make_type_32.loc4_12 [template = i32]
// CHECK:STDOUT:     %.loc4_15.5: type = converted %int.make_type_32.loc4_12, %.loc4_15.4 [template = i32]
// CHECK:STDOUT:     %.loc4_15.6: type = converted %.loc4_15.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.1 = name_ref IndexWith, imports.%import_ref.2 [template = constants.%IndexWith]
// CHECK:STDOUT:     %int.make_type_32.loc4_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc4_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_34.1: type = value_of_initializer %int.make_type_32.loc4_35 [template = i32]
// CHECK:STDOUT:     %.loc4_34.2: type = converted %int.make_type_32.loc4_35, %.loc4_34.1 [template = i32]
// CHECK:STDOUT:     %.loc4_34.3: type = value_of_initializer %int.make_type_32.loc4_40 [template = i32]
// CHECK:STDOUT:     %.loc4_34.4: type = converted %int.make_type_32.loc4_40, %.loc4_34.3 [template = i32]
// CHECK:STDOUT:     %IndexWith.type: type = interface_type @IndexWith, @IndexWith(i32, i32) [template = constants.%IndexWith.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc10_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc10_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc10_17.1: %.2 = tuple_literal (%int.make_type_32.loc10_9, %int.make_type_32.loc10_14)
// CHECK:STDOUT:   %.loc10_17.2: type = value_of_initializer %int.make_type_32.loc10_9 [template = i32]
// CHECK:STDOUT:   %.loc10_17.3: type = converted %int.make_type_32.loc10_9, %.loc10_17.2 [template = i32]
// CHECK:STDOUT:   %.loc10_17.4: type = value_of_initializer %int.make_type_32.loc10_14 [template = i32]
// CHECK:STDOUT:   %.loc10_17.5: type = converted %int.make_type_32.loc10_14, %.loc10_17.4 [template = i32]
// CHECK:STDOUT:   %.loc10_17.6: type = converted %.loc10_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_8.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_8.2: type = converted %int.make_type_32.loc11, %.loc11_8.1 [template = i32]
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
// CHECK:STDOUT:   %At.type: type = fn_type @At.1, @IndexWith(%SubscriptType, %ElementType) [symbolic = %At.type (constants.%At.type.1)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type.1) = struct_value () [symbolic = %At (constants.%At.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @IndexWith.%IndexWith.type (%IndexWith.type.2), @IndexWith.%At.type (%At.type.1) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %.2: @IndexWith.%.1 (%.4) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .At = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc4_15.6 as %IndexWith.type {
// CHECK:STDOUT:   %At.decl: %At.type.2 = fn_decl @At.2 [template = constants.%At.2] {
// CHECK:STDOUT:     %self.patt: %.3 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.3 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %subscript.patt: i32 = binding_pattern subscript
// CHECK:STDOUT:     %subscript.param_patt: i32 = value_param_pattern %subscript.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%.loc4_15.6 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_32.loc5_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_32.1: type = value_of_initializer %int.make_type_32.loc5_32 [template = i32]
// CHECK:STDOUT:     %.loc5_32.2: type = converted %int.make_type_32.loc5_32, %.loc5_32.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc5_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_40.1: type = value_of_initializer %int.make_type_32.loc5_40 [template = i32]
// CHECK:STDOUT:     %.loc5_40.2: type = converted %int.make_type_32.loc5_40, %.loc5_40.1 [template = i32]
// CHECK:STDOUT:     %self.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.3 = bind_name self, %self.param
// CHECK:STDOUT:     %subscript.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %subscript: i32 = bind_name subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_45: <witness> = interface_witness (%At.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = %.loc4_45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @At.1(constants.%SubscriptType: type, constants.%ElementType: type, constants.%Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2)) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType, 1 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:   %IndexWith.type: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @At.1.%Self (%Self.2)](%subscript.param_patt: @At.1.%SubscriptType (%SubscriptType)) -> @At.1.%ElementType (%ElementType);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %.3](%subscript.param_patt: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %.3 = name_ref self, %self
// CHECK:STDOUT:   %.loc6_17: i32 = int_literal 0 [template = constants.%.10]
// CHECK:STDOUT:   %.loc6_16: i32 = tuple_access %self.ref, element0
// CHECK:STDOUT:   return %.loc6_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_22: i32 = int_literal 1 [template = constants.%.11]
// CHECK:STDOUT:   %.loc10_25: i32 = int_literal 5 [template = constants.%.12]
// CHECK:STDOUT:   %.loc10_26: %.3 = tuple_literal (%.loc10_22, %.loc10_25)
// CHECK:STDOUT:   %tuple: %.3 = tuple_value (%.loc10_22, %.loc10_25) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc10_27: %.3 = converted %.loc10_26, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %s: %.3 = bind_name s, %.loc10_27
// CHECK:STDOUT:   %s.ref: %.3 = name_ref s, %s
// CHECK:STDOUT:   %.loc11_16: i32 = int_literal 0 [template = constants.%.10]
// CHECK:STDOUT:   %IndexWith.type: type = interface_type @IndexWith, @IndexWith(i32, i32) [template = constants.%IndexWith.type.3]
// CHECK:STDOUT:   %.loc11_17.1: %.6 = specific_constant imports.%import_ref.4, @IndexWith(i32, i32) [template = constants.%.7]
// CHECK:STDOUT:   %At.ref: %.6 = name_ref At, %.loc11_17.1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_17.2: %At.type.3 = interface_witness_access constants.%.8, element0 [template = constants.%At.2]
// CHECK:STDOUT:   %.loc11_17.3: <bound method> = bound_method %s.ref, %.loc11_17.2
// CHECK:STDOUT:   %At.call: init i32 = call %.loc11_17.3(%s.ref, %.loc11_16)
// CHECK:STDOUT:   %.loc11_18.1: i32 = value_of_initializer %At.call
// CHECK:STDOUT:   %.loc11_18.2: i32 = converted %At.call, %.loc11_18.1
// CHECK:STDOUT:   %e: i32 = bind_name e, %.loc11_18.2
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
// CHECK:STDOUT: specific @IndexWith(@At.1.%SubscriptType, @At.1.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %SubscriptType.patt => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT:   %ElementType.patt => constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(constants.%SubscriptType, constants.%ElementType, constants.%Self.1) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(i32, i32) {
// CHECK:STDOUT:   %SubscriptType => i32
// CHECK:STDOUT:   %SubscriptType.patt => i32
// CHECK:STDOUT:   %ElementType => i32
// CHECK:STDOUT:   %ElementType.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %At.type => constants.%At.type.3
// CHECK:STDOUT:   %At => constants.%At.3
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(i32, i32, constants.%.3) {
// CHECK:STDOUT:   %SubscriptType => i32
// CHECK:STDOUT:   %ElementType => i32
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.3
// CHECK:STDOUT:   %Self => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_subscript_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %ElementType.1: type = class_type @ElementType [template]
// CHECK:STDOUT:   %SubscriptType.1: type = class_type @SubscriptType [template]
// CHECK:STDOUT:   %IndexWith.type.1: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %IndexWith: %IndexWith.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ElementType.2: type = bind_symbolic_name ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.2: type = bind_symbolic_name SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.2: type = interface_type @IndexWith, @IndexWith(%SubscriptType.2, %ElementType.2) [symbolic]
// CHECK:STDOUT:   %Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2) = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %ElementType.patt: type = symbolic_binding_pattern ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %At.type.1: type = fn_type @At.1, @IndexWith(%SubscriptType.2, %ElementType.2) [symbolic]
// CHECK:STDOUT:   %At.1: %At.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %IndexWith.type.2, %At.type.1 [symbolic]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.3: type = interface_type @IndexWith, @IndexWith(%SubscriptType.1, %ElementType.1) [template]
// CHECK:STDOUT:   %At.type.2: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.2: %At.type.2 = struct_value () [template]
// CHECK:STDOUT:   %At.type.3: type = fn_type @At.1, @IndexWith(%SubscriptType.1, %ElementType.1) [template]
// CHECK:STDOUT:   %At.3: %At.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %IndexWith.type.3, %At.type.3 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%At.2) [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct.1: %ElementType.1 = struct_value () [template]
// CHECK:STDOUT:   %struct.2: %C = struct_value () [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.3: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.4: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.10 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = interface_type @ImplicitAs, @ImplicitAs(%SubscriptType.1) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%SubscriptType.1) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.13: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.14: %.13 = assoc_entity element0, imports.%import_ref.10 [template]
// CHECK:STDOUT:   %.15: %.11 = assoc_entity element0, imports.%import_ref.11 [symbolic]
// CHECK:STDOUT:   %.16: %.4 = assoc_entity element0, imports.%import_ref.12 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IndexWith = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/index
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IndexWith.type.1 = import_ref Core//prelude/operators/index, inst+13, loaded [template = constants.%IndexWith]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/index, inst+22, unloaded
// CHECK:STDOUT:   %import_ref.3: @IndexWith.%.1 (%.4) = import_ref Core//prelude/operators/index, inst+50, loaded [symbolic = @IndexWith.%.2 (constants.%.16)]
// CHECK:STDOUT:   %import_ref.4: @IndexWith.%At.type (%At.type.1) = import_ref Core//prelude/operators/index, inst+42, loaded [symbolic = @IndexWith.%At (constants.%At.1)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/index, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.6: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+49, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+55, unloaded
// CHECK:STDOUT:   %import_ref.8: @ImplicitAs.%.1 (%.11) = import_ref Core//prelude/operators/as, inst+77, loaded [symbolic = @ImplicitAs.%.2 (constants.%.15)]
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/as, inst+70, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//prelude/operators/as, inst+70, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//prelude/operators/as, inst+70, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//prelude/operators/index, inst+42, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .ElementType = %ElementType.decl
// CHECK:STDOUT:     .SubscriptType = %SubscriptType.decl
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:     .x = @__global_init.%x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %ElementType.decl: type = class_decl @ElementType [template = constants.%ElementType.1] {} {}
// CHECK:STDOUT:   %SubscriptType.decl: type = class_decl @SubscriptType [template = constants.%SubscriptType.1] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.1 = name_ref IndexWith, imports.%import_ref.1 [template = constants.%IndexWith]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT:     %IndexWith.type: type = interface_type @IndexWith, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) [template = constants.%IndexWith.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %ElementType.ref: type = name_ref ElementType, %ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @IndexWith(constants.%SubscriptType.2: type, constants.%ElementType.2: type) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType.2)]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic = %SubscriptType.patt (constants.%SubscriptType.patt)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType, 1 [symbolic = %ElementType (constants.%ElementType.2)]
// CHECK:STDOUT:   %ElementType.patt: type = symbolic_binding_pattern ElementType, 1 [symbolic = %ElementType.patt (constants.%ElementType.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %At.type: type = fn_type @At.1, @IndexWith(%SubscriptType, %ElementType) [symbolic = %At.type (constants.%At.type.1)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type.1) = struct_value () [symbolic = %At (constants.%At.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @IndexWith.%IndexWith.type (%IndexWith.type.2), @IndexWith.%At.type (%At.type.1) [symbolic = %.1 (constants.%.4)]
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
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
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
// CHECK:STDOUT: impl @impl: %C.ref as %IndexWith.type {
// CHECK:STDOUT:   %At.decl: %At.type.2 = fn_decl @At.2 [template = constants.%At.2] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %subscript.patt: %SubscriptType.1 = binding_pattern subscript
// CHECK:STDOUT:     %subscript.param_patt: %SubscriptType.1 = value_param_pattern %subscript.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %ElementType.1 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ElementType.1 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%C.ref [template = constants.%C]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %subscript.param: %SubscriptType.1 = value_param runtime_param1
// CHECK:STDOUT:     %subscript: %SubscriptType.1 = bind_name subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %ElementType.1 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %ElementType.1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = interface_witness (%At.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ElementType {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ElementType.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType {
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SubscriptType.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @At.1(constants.%SubscriptType.2: type, constants.%ElementType.2: type, constants.%Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2)) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType.2)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType, 1 [symbolic = %ElementType (constants.%ElementType.2)]
// CHECK:STDOUT:   %IndexWith.type: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @At.1.%Self (%Self.2)](%subscript.param_patt: @At.1.%SubscriptType (%SubscriptType.2)) -> @At.1.%ElementType (%ElementType.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %C](%subscript.param_patt: %SubscriptType.1) -> %return: %ElementType.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc10_13.2: init %ElementType.1 = class_init (), %return [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc10_14: init %ElementType.1 = converted %.loc10_13.1, %.loc10_13.2 [template = constants.%struct.1]
// CHECK:STDOUT:   return %.loc10_14 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.3: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self (%Self.4)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc14_13.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc14_13.3: init %C = class_init (), %.loc14_13.2 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc14_13.4: ref %C = temporary %.loc14_13.2, %.loc14_13.3
// CHECK:STDOUT:   %.loc14_14.1: ref %C = converted %.loc14_13.1, %.loc14_13.4
// CHECK:STDOUT:   %.loc14_14.2: %C = bind_value %.loc14_14.1
// CHECK:STDOUT:   %c: %C = bind_name c, %.loc14_14.2
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc22_24: i32 = int_literal 0 [template = constants.%.10]
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(constants.%SubscriptType.1) [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %.loc22_25.1: %.13 = specific_constant imports.%import_ref.8, @ImplicitAs(constants.%SubscriptType.1) [template = constants.%.14]
// CHECK:STDOUT:   %Convert.ref: %.13 = name_ref Convert, %.loc22_25.1 [template = constants.%.14]
// CHECK:STDOUT:   %.loc22_25.2: %SubscriptType.1 = converted %.loc22_24, <error> [template = <error>]
// CHECK:STDOUT:   %IndexWith.type: type = interface_type @IndexWith, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) [template = constants.%IndexWith.type.3]
// CHECK:STDOUT:   %.loc22_25.3: %.6 = specific_constant imports.%import_ref.3, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) [template = constants.%.7]
// CHECK:STDOUT:   %At.ref: %.6 = name_ref At, %.loc22_25.3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc22_25.4: %At.type.3 = interface_witness_access constants.%.8, element0 [template = constants.%At.2]
// CHECK:STDOUT:   %.loc22_25.5: <bound method> = bound_method %c.ref, %.loc22_25.4
// CHECK:STDOUT:   %.loc22_25.6: ref %ElementType.1 = temporary_storage
// CHECK:STDOUT:   %At.call: init %ElementType.1 = call %.loc22_25.5(%c.ref, <error>) to %.loc22_25.6
// CHECK:STDOUT:   %.loc22_25.7: ref %ElementType.1 = temporary %.loc22_25.6, %At.call
// CHECK:STDOUT:   %.loc22_25.8: %ElementType.1 = bind_value %.loc22_25.7
// CHECK:STDOUT:   %x: %ElementType.1 = bind_name x, %.loc22_25.8
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType.2, constants.%ElementType.2) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %SubscriptType.patt => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT:   %ElementType.patt => constants.%ElementType.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@IndexWith.%SubscriptType, @IndexWith.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %SubscriptType.patt => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT:   %ElementType.patt => constants.%ElementType.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@At.1.%SubscriptType, @At.1.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %SubscriptType.patt => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT:   %ElementType.patt => constants.%ElementType.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(constants.%SubscriptType.2, constants.%ElementType.2, constants.%Self.1) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.1
// CHECK:STDOUT:   %SubscriptType.patt => constants.%SubscriptType.1
// CHECK:STDOUT:   %ElementType => constants.%ElementType.1
// CHECK:STDOUT:   %ElementType.patt => constants.%ElementType.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %At.type => constants.%At.type.3
// CHECK:STDOUT:   %At => constants.%At.3
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(constants.%SubscriptType.1, constants.%ElementType.1, constants.%C) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.1
// CHECK:STDOUT:   %ElementType => constants.%ElementType.1
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.3
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%SubscriptType.1) {
// CHECK:STDOUT:   %Dest => constants.%SubscriptType.1
// CHECK:STDOUT:   %Dest.patt => constants.%SubscriptType.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT:   %.2 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_index_with_not_implemented.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 0 [template]
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
// CHECK:STDOUT:   %.6: type = assoc_entity_type %IndexWith.type.2, %At.type [symbolic]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .IndexWith = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/index
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:     .x = @__global_init.%x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc10_8.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc10_8.2: type = converted %int.make_type_32, %.loc10_8.1 [template = i32]
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
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
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
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc6_13.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc6_13.3: init %C = class_init (), %.loc6_13.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_13.4: ref %C = temporary %.loc6_13.2, %.loc6_13.3
// CHECK:STDOUT:   %.loc6_14.1: ref %C = converted %.loc6_13.1, %.loc6_13.4
// CHECK:STDOUT:   %.loc6_14.2: %C = bind_value %.loc6_14.1
// CHECK:STDOUT:   %c: %C = bind_name c, %.loc6_14.2
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc10: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %x: i32 = bind_name x, <error>
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
