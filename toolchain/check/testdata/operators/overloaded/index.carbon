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
// CHECK:STDERR: fail_invalid_subscript_type.carbon:[[@LINE+4]]:22: note: type `i32` does not implement interface `ImplicitAs(SubscriptType)` [MissingImplInMemberAccessNote]
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %IndexWith: %IndexWith.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ElementType.2: type = bind_symbolic_name ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.2: type = bind_symbolic_name SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.2: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.2, %ElementType.2)> [symbolic]
// CHECK:STDOUT:   %Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2) = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %ElementType.patt: type = symbolic_binding_pattern ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %At.type.1: type = fn_type @At.1, @IndexWith(%SubscriptType.2, %ElementType.2) [symbolic]
// CHECK:STDOUT:   %At.1: %At.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %IndexWith.type.2, %At.type.1 [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.3: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.1, %ElementType.1)> [template]
// CHECK:STDOUT:   %At.type.2: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.2: %At.type.2 = struct_value () [template]
// CHECK:STDOUT:   %At.type.3: type = fn_type @At.1, @IndexWith(%SubscriptType.1, %ElementType.1) [template]
// CHECK:STDOUT:   %At.3: %At.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %IndexWith.type.3, %At.type.3 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.7: <witness> = interface_witness (%At.2) [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct.1: %ElementType.1 = struct_value () [template]
// CHECK:STDOUT:   %struct.2: %SubscriptType.1 = struct_value () [template]
// CHECK:STDOUT:   %struct.3: %C = struct_value () [template]
// CHECK:STDOUT:   %.9: %.3 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IndexWith = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IndexWith.type.1 = import_ref Core//prelude/operators/index, inst+13, loaded [template = constants.%IndexWith]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/index, inst+22, unloaded
// CHECK:STDOUT:   %import_ref.3: @IndexWith.%.1 (%.3) = import_ref Core//prelude/operators/index, inst+50, loaded [symbolic = @IndexWith.%.2 (constants.%.9)]
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
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1)> [template = constants.%IndexWith.type.3]
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
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %At.type: type = fn_type @At.1, @IndexWith(%SubscriptType, %ElementType) [symbolic = %At.type (constants.%At.type.1)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type.1) = struct_value () [symbolic = %At (constants.%At.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @IndexWith.%IndexWith.type (%IndexWith.type.2), @IndexWith.%At.type (%At.type.1) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %.2: @IndexWith.%.1 (%.3) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.2 (constants.%.4)]
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
// CHECK:STDOUT:   %.loc8: <witness> = interface_witness (%At.decl) [template = constants.%.7]
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
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
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
// CHECK:STDOUT:   %.loc16_25.1: %At.type.3 = interface_witness_access constants.%.7, element0 [template = constants.%At.2]
// CHECK:STDOUT:   %.loc16_25.2: <bound method> = bound_method %c.ref, %.loc16_25.1
// CHECK:STDOUT:   %.loc16_25.3: ref %ElementType.1 = temporary_storage
// CHECK:STDOUT:   %At.call: init %ElementType.1 = call %.loc16_25.2(%c.ref, %s.ref) to %.loc16_25.3
// CHECK:STDOUT:   %.loc16_25.4: ref %ElementType.1 = temporary %.loc16_25.3, %At.call
// CHECK:STDOUT:   %.loc16_25.5: %ElementType.1 = bind_value %.loc16_25.4
// CHECK:STDOUT:   %x: %ElementType.1 = bind_name x, %.loc16_25.5
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
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT:   %.2 => constants.%.6
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %IndexWith.type.1: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %IndexWith: %IndexWith.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.2: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic]
// CHECK:STDOUT:   %Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2) = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %ElementType.patt: type = symbolic_binding_pattern ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %At.type.1: type = fn_type @At.1, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %At.1: %At.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %IndexWith.type.2, %At.type.1 [symbolic]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.3: type = facet_type <@IndexWith, @IndexWith(i32, i32)> [template]
// CHECK:STDOUT:   %At.type.2: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.2: %At.type.2 = struct_value () [template]
// CHECK:STDOUT:   %At.type.3: type = fn_type @At.1, @IndexWith(i32, i32) [template]
// CHECK:STDOUT:   %At.3: %At.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %IndexWith.type.3, %At.type.3 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%At.2) [template]
// CHECK:STDOUT:   %.6: type = ptr_type %tuple.type.2 [template]
// CHECK:STDOUT:   %.7: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.8: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.9: i32 = int_value 5 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.8, %.9) [template]
// CHECK:STDOUT:   %.10: %.1 = assoc_entity element0, imports.%import_ref.7 [symbolic]
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
// CHECK:STDOUT:   %import_ref.4: @IndexWith.%.1 (%.1) = import_ref Core//prelude/operators/index, inst+50, loaded [symbolic = @IndexWith.%.2 (constants.%.10)]
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
// CHECK:STDOUT:     %.loc4_15.1: %tuple.type.1 = tuple_literal (%int.make_type_32.loc4_7, %int.make_type_32.loc4_12)
// CHECK:STDOUT:     %.loc4_15.2: type = value_of_initializer %int.make_type_32.loc4_7 [template = i32]
// CHECK:STDOUT:     %.loc4_15.3: type = converted %int.make_type_32.loc4_7, %.loc4_15.2 [template = i32]
// CHECK:STDOUT:     %.loc4_15.4: type = value_of_initializer %int.make_type_32.loc4_12 [template = i32]
// CHECK:STDOUT:     %.loc4_15.5: type = converted %int.make_type_32.loc4_12, %.loc4_15.4 [template = i32]
// CHECK:STDOUT:     %.loc4_15.6: type = converted %.loc4_15.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.1 = name_ref IndexWith, imports.%import_ref.2 [template = constants.%IndexWith]
// CHECK:STDOUT:     %int.make_type_32.loc4_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc4_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_34.1: type = value_of_initializer %int.make_type_32.loc4_35 [template = i32]
// CHECK:STDOUT:     %.loc4_34.2: type = converted %int.make_type_32.loc4_35, %.loc4_34.1 [template = i32]
// CHECK:STDOUT:     %.loc4_34.3: type = value_of_initializer %int.make_type_32.loc4_40 [template = i32]
// CHECK:STDOUT:     %.loc4_34.4: type = converted %int.make_type_32.loc4_40, %.loc4_34.3 [template = i32]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(i32, i32)> [template = constants.%IndexWith.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc10_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc10_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc10_17.1: %tuple.type.1 = tuple_literal (%int.make_type_32.loc10_9, %int.make_type_32.loc10_14)
// CHECK:STDOUT:   %.loc10_17.2: type = value_of_initializer %int.make_type_32.loc10_9 [template = i32]
// CHECK:STDOUT:   %.loc10_17.3: type = converted %int.make_type_32.loc10_9, %.loc10_17.2 [template = i32]
// CHECK:STDOUT:   %.loc10_17.4: type = value_of_initializer %int.make_type_32.loc10_14 [template = i32]
// CHECK:STDOUT:   %.loc10_17.5: type = converted %int.make_type_32.loc10_14, %.loc10_17.4 [template = i32]
// CHECK:STDOUT:   %.loc10_17.6: type = converted %.loc10_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
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
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %At.type: type = fn_type @At.1, @IndexWith(%SubscriptType, %ElementType) [symbolic = %At.type (constants.%At.type.1)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type.1) = struct_value () [symbolic = %At (constants.%At.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @IndexWith.%IndexWith.type (%IndexWith.type.2), @IndexWith.%At.type (%At.type.1) [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:   %.2: @IndexWith.%.1 (%.1) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.2)]
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
// CHECK:STDOUT:     %self.patt: %tuple.type.2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %tuple.type.2 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %subscript.patt: i32 = binding_pattern subscript
// CHECK:STDOUT:     %subscript.param_patt: i32 = value_param_pattern %subscript.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%.loc4_15.6 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %int.make_type_32.loc5_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_32.1: type = value_of_initializer %int.make_type_32.loc5_32 [template = i32]
// CHECK:STDOUT:     %.loc5_32.2: type = converted %int.make_type_32.loc5_32, %.loc5_32.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc5_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_40.1: type = value_of_initializer %int.make_type_32.loc5_40 [template = i32]
// CHECK:STDOUT:     %.loc5_40.2: type = converted %int.make_type_32.loc5_40, %.loc5_40.1 [template = i32]
// CHECK:STDOUT:     %self.param: %tuple.type.2 = value_param runtime_param0
// CHECK:STDOUT:     %self: %tuple.type.2 = bind_name self, %self.param
// CHECK:STDOUT:     %subscript.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %subscript: i32 = bind_name subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_45: <witness> = interface_witness (%At.decl) [template = constants.%.5]
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
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @At.1.%Self (%Self.2)](%subscript.param_patt: @At.1.%SubscriptType (%SubscriptType)) -> @At.1.%ElementType (%ElementType);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %tuple.type.2](%subscript.param_patt: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %tuple.type.2 = name_ref self, %self
// CHECK:STDOUT:   %.loc6_17: i32 = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc6_16: i32 = tuple_access %self.ref, element0
// CHECK:STDOUT:   return %.loc6_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_22: i32 = int_value 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc10_25: i32 = int_value 5 [template = constants.%.9]
// CHECK:STDOUT:   %.loc10_26: %tuple.type.2 = tuple_literal (%.loc10_22, %.loc10_25)
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.loc10_22, %.loc10_25) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc10_27: %tuple.type.2 = converted %.loc10_26, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %s: %tuple.type.2 = bind_name s, %.loc10_27
// CHECK:STDOUT:   %s.ref: %tuple.type.2 = name_ref s, %s
// CHECK:STDOUT:   %.loc11_16: i32 = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_17.1: %At.type.3 = interface_witness_access constants.%.5, element0 [template = constants.%At.2]
// CHECK:STDOUT:   %.loc11_17.2: <bound method> = bound_method %s.ref, %.loc11_17.1
// CHECK:STDOUT:   %At.call: init i32 = call %.loc11_17.2(%s.ref, %.loc11_16)
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
// CHECK:STDOUT:   %.1 => constants.%.3
// CHECK:STDOUT:   %.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(i32, i32, constants.%tuple.type.2) {
// CHECK:STDOUT:   %SubscriptType => i32
// CHECK:STDOUT:   %ElementType => i32
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.3
// CHECK:STDOUT:   %Self => constants.%tuple.type.2
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %IndexWith: %IndexWith.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ElementType.2: type = bind_symbolic_name ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.2: type = bind_symbolic_name SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.2: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.2, %ElementType.2)> [symbolic]
// CHECK:STDOUT:   %Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2) = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %ElementType.patt: type = symbolic_binding_pattern ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %At.type.1: type = fn_type @At.1, @IndexWith(%SubscriptType.2, %ElementType.2) [symbolic]
// CHECK:STDOUT:   %At.1: %At.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %IndexWith.type.2, %At.type.1 [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.3: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.1, %ElementType.1)> [template]
// CHECK:STDOUT:   %At.type.2: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.2: %At.type.2 = struct_value () [template]
// CHECK:STDOUT:   %At.type.3: type = fn_type @At.1, @IndexWith(%SubscriptType.1, %ElementType.1) [template]
// CHECK:STDOUT:   %At.3: %At.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %IndexWith.type.3, %At.type.3 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.7: <witness> = interface_witness (%At.2) [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct.1: %ElementType.1 = struct_value () [template]
// CHECK:STDOUT:   %struct.2: %C = struct_value () [template]
// CHECK:STDOUT:   %.9: i32 = int_value 0 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.4: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.10 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%SubscriptType.1)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%SubscriptType.1) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.12: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.13: %.12 = assoc_entity element0, imports.%import_ref.10 [template]
// CHECK:STDOUT:   %.14: %.10 = assoc_entity element0, imports.%import_ref.11 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.15: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.3 [template]
// CHECK:STDOUT:   %.16: %.15 = assoc_entity element0, imports.%import_ref.15 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.17: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%.17)> [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.2, @impl.3(%N) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.18: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %.19: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(%.19)> [symbolic]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.3, @impl.4(%N) [symbolic]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [symbolic]
// CHECK:STDOUT:   %.20: <witness> = interface_witness (%Convert.5) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.21: type = assoc_entity_type %ImplicitAs.type.7, %Convert.type.6 [template]
// CHECK:STDOUT:   %.22: %.21 = assoc_entity element0, imports.%import_ref.25 [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.4, @impl.6(%N) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.23: <witness> = interface_witness (%Convert.7) [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.5, @impl.7(%N) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.24: <witness> = interface_witness (%Convert.8) [symbolic]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.5: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(i32)> [template]
// CHECK:STDOUT:   %Self.6: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [symbolic]
// CHECK:STDOUT:   %.25: type = assoc_entity_type %As.type.2, %Convert.type.9 [symbolic]
// CHECK:STDOUT:   %.26: %.25 = assoc_entity element0, imports.%import_ref.38 [symbolic]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.6, @As(i32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %.27: type = assoc_entity_type %As.type.3, %Convert.type.10 [template]
// CHECK:STDOUT:   %.28: %.27 = assoc_entity element0, imports.%import_ref.39 [template]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%.17)> [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.7, @impl.9(%N) [symbolic]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [symbolic]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.11) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(%.19)> [symbolic]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.8, @impl.10(%N) [symbolic]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [symbolic]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.12) [symbolic]
// CHECK:STDOUT:   %As.type.6: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [template]
// CHECK:STDOUT:   %.31: type = assoc_entity_type %As.type.6, %Convert.type.13 [template]
// CHECK:STDOUT:   %.32: %.31 = assoc_entity element0, imports.%import_ref.49 [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.9, @impl.12(%N) [symbolic]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [symbolic]
// CHECK:STDOUT:   %.33: <witness> = interface_witness (%Convert.14) [symbolic]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.10, @impl.13(%N) [symbolic]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [symbolic]
// CHECK:STDOUT:   %.34: <witness> = interface_witness (%Convert.15) [symbolic]
// CHECK:STDOUT:   %.35: %.3 = assoc_entity element0, imports.%import_ref.56 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IndexWith = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IndexWith.type.1 = import_ref Core//prelude/operators/index, inst+13, loaded [template = constants.%IndexWith]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/index, inst+22, unloaded
// CHECK:STDOUT:   %import_ref.3: @IndexWith.%.1 (%.3) = import_ref Core//prelude/operators/index, inst+50, loaded [symbolic = @IndexWith.%.2 (constants.%.35)]
// CHECK:STDOUT:   %import_ref.4: @IndexWith.%At.type (%At.type.1) = import_ref Core//prelude/operators/index, inst+42, loaded [symbolic = @IndexWith.%At (constants.%At.1)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/index, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.6: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+55, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.8: @ImplicitAs.%.1 (%.10) = import_ref Core//prelude/operators/as, inst+83, loaded [symbolic = @ImplicitAs.%.2 (constants.%.14)]
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//prelude/operators/as, inst+100, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.13: type = import_ref Core//prelude/operators/as, inst+111, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.14 = import_ref Core//prelude/operators/as, inst+133, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.16: type = import_ref Core//prelude/operators/as, inst+147, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.17: type = import_ref Core//prelude/operators/as, inst+165, loaded [symbolic = @impl.3.%ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:   %import_ref.18 = import_ref Core//prelude/operators/as, inst+196, unloaded
// CHECK:STDOUT:   %import_ref.19: type = import_ref Core//prelude/operators/as, inst+211, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.20: type = import_ref Core//prelude/operators/as, inst+229, loaded [symbolic = @impl.4.%ImplicitAs.type (constants.%ImplicitAs.type.6)]
// CHECK:STDOUT:   %import_ref.21 = import_ref Core//prelude/operators/as, inst+260, unloaded
// CHECK:STDOUT:   %import_ref.22: type = import_ref Core//prelude/operators/as, inst+267, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.23: type = import_ref Core//prelude/operators/as, inst+273, loaded [template = constants.%ImplicitAs.type.7]
// CHECK:STDOUT:   %import_ref.24 = import_ref Core//prelude/operators/as, inst+296, unloaded
// CHECK:STDOUT:   %import_ref.25 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.26: type = import_ref Core//prelude/operators/as, inst+309, loaded [symbolic = @impl.6.%.1 (constants.%.17)]
// CHECK:STDOUT:   %import_ref.27: type = import_ref Core//prelude/operators/as, inst+315, loaded [template = constants.%ImplicitAs.type.7]
// CHECK:STDOUT:   %import_ref.28 = import_ref Core//prelude/operators/as, inst+339, unloaded
// CHECK:STDOUT:   %import_ref.29: type = import_ref Core//prelude/operators/as, inst+355, loaded [symbolic = @impl.7.%.1 (constants.%.19)]
// CHECK:STDOUT:   %import_ref.30: type = import_ref Core//prelude/operators/as, inst+361, loaded [template = constants.%ImplicitAs.type.7]
// CHECK:STDOUT:   %import_ref.31 = import_ref Core//prelude/operators/as, inst+385, unloaded
// CHECK:STDOUT:   %import_ref.32 = import_ref Core//prelude/operators/as, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.33 = import_ref Core//prelude/operators/as, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.34 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.35: type = import_ref Core//prelude/operators/as, inst+393, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.36: type = import_ref Core//prelude/operators/as, inst+398, loaded [template = constants.%As.type.3]
// CHECK:STDOUT:   %import_ref.37 = import_ref Core//prelude/operators/as, inst+420, unloaded
// CHECK:STDOUT:   %import_ref.38 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.39 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.40: type = import_ref Core//prelude/operators/as, inst+432, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.41: type = import_ref Core//prelude/operators/as, inst+439, loaded [symbolic = @impl.9.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.42 = import_ref Core//prelude/operators/as, inst+470, unloaded
// CHECK:STDOUT:   %import_ref.43: type = import_ref Core//prelude/operators/as, inst+485, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.44: type = import_ref Core//prelude/operators/as, inst+492, loaded [symbolic = @impl.10.%As.type (constants.%As.type.5)]
// CHECK:STDOUT:   %import_ref.45 = import_ref Core//prelude/operators/as, inst+523, unloaded
// CHECK:STDOUT:   %import_ref.46: type = import_ref Core//prelude/operators/as, inst+530, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.47: type = import_ref Core//prelude/operators/as, inst+536, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.48 = import_ref Core//prelude/operators/as, inst+559, unloaded
// CHECK:STDOUT:   %import_ref.49 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.50: type = import_ref Core//prelude/operators/as, inst+572, loaded [symbolic = @impl.12.%.1 (constants.%.17)]
// CHECK:STDOUT:   %import_ref.51: type = import_ref Core//prelude/operators/as, inst+578, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.52 = import_ref Core//prelude/operators/as, inst+602, unloaded
// CHECK:STDOUT:   %import_ref.53: type = import_ref Core//prelude/operators/as, inst+618, loaded [symbolic = @impl.13.%.1 (constants.%.19)]
// CHECK:STDOUT:   %import_ref.54: type = import_ref Core//prelude/operators/as, inst+624, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.55 = import_ref Core//prelude/operators/as, inst+648, unloaded
// CHECK:STDOUT:   %import_ref.56 = import_ref Core//prelude/operators/index, inst+42, unloaded
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
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.1 = name_ref IndexWith, imports.%import_ref.1 [template = constants.%IndexWith]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1)> [template = constants.%IndexWith.type.3]
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
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %At.type: type = fn_type @At.1, @IndexWith(%SubscriptType, %ElementType) [symbolic = %At.type (constants.%At.type.1)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type.1) = struct_value () [symbolic = %At (constants.%At.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @IndexWith.%IndexWith.type (%IndexWith.type.2), @IndexWith.%At.type (%At.type.1) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %.2: @IndexWith.%.1 (%.3) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.2 (constants.%.4)]
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
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.10) = assoc_entity element0, imports.%import_ref.10 [symbolic = %.2 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.7
// CHECK:STDOUT:     .Convert = imports.%import_ref.8
// CHECK:STDOUT:     witness = (imports.%import_ref.9)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.6)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.9)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.9) = struct_value () [symbolic = %Convert (constants.%Convert.9)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.9) [symbolic = %.1 (constants.%.25)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.25) = assoc_entity element0, imports.%import_ref.38 [symbolic = %.2 (constants.%.26)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.32
// CHECK:STDOUT:     .Convert = imports.%import_ref.33
// CHECK:STDOUT:     witness = (imports.%import_ref.34)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C.ref as %IndexWith.type {
// CHECK:STDOUT:   %At.decl: %At.type.2 = fn_decl @At.2 [template = constants.%At.2] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %subscript.patt: %SubscriptType.1 = binding_pattern subscript
// CHECK:STDOUT:     %subscript.param_patt: %SubscriptType.1 = value_param_pattern %subscript.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %ElementType.1 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ElementType.1 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.1.%C.ref [template = constants.%C]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %subscript.param: %SubscriptType.1 = value_param runtime_param1
// CHECK:STDOUT:     %subscript: %SubscriptType.1 = bind_name subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %ElementType.1 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %ElementType.1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = interface_witness (%At.decl) [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.12 as imports.%import_ref.13 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.17)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.18)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.16 as imports.%import_ref.17 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.18
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.4(%N) [symbolic = %Convert.type (constants.%Convert.type.5)]
// CHECK:STDOUT:   %Convert: @impl.4.%Convert.type (%Convert.type.5) = struct_value () [symbolic = %Convert (constants.%Convert.5)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.19 as imports.%import_ref.20 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.21
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: imports.%import_ref.22 as imports.%import_ref.23 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.23)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.26 as imports.%import_ref.27 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.28
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.7(%N) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @impl.7.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.24)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.29 as imports.%import_ref.30 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.31
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: imports.%import_ref.35 as imports.%import_ref.36 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.17)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.9(%N) [symbolic = %Convert.type (constants.%Convert.type.11)]
// CHECK:STDOUT:   %Convert: @impl.9.%Convert.type (%Convert.type.11) = struct_value () [symbolic = %Convert (constants.%Convert.11)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.29)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.40 as imports.%import_ref.41 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.42
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.10(%N) [symbolic = %Convert.type (constants.%Convert.type.12)]
// CHECK:STDOUT:   %Convert: @impl.10.%Convert.type (%Convert.type.12) = struct_value () [symbolic = %Convert (constants.%Convert.12)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.30)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.43 as imports.%import_ref.44 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.45
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.11: imports.%import_ref.46 as imports.%import_ref.47 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.12(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.12(%N) [symbolic = %Convert.type (constants.%Convert.type.14)]
// CHECK:STDOUT:   %Convert: @impl.12.%Convert.type (%Convert.type.14) = struct_value () [symbolic = %Convert (constants.%Convert.14)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.33)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.50 as imports.%import_ref.51 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.52
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.13(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.13(%N) [symbolic = %Convert.type (constants.%Convert.type.15)]
// CHECK:STDOUT:   %Convert: @impl.13.%Convert.type (%Convert.type.15) = struct_value () [symbolic = %Convert (constants.%Convert.15)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.34)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.53 as imports.%import_ref.54 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.55
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
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
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
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.3: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.4)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%.1 (%.17) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%.1 (%.19) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%.1 (%.17)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%.1 (%.19)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.6(constants.%Dest: type, constants.%Self.5: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.6.%Self (%Self.6)]() -> @Convert.6.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%.1 (%.17) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%.1 (%.19) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%.1 (%.17)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%.1 (%.19)]() -> Core.IntLiteral = "int.convert_checked";
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
// CHECK:STDOUT:   %.loc22_24: i32 = int_value 0 [template = constants.%.9]
// CHECK:STDOUT:   %.loc22_25.1: %SubscriptType.1 = converted %.loc22_24, <error> [template = <error>]
// CHECK:STDOUT:   %.loc22_25.2: %At.type.3 = interface_witness_access constants.%.7, element0 [template = constants.%At.2]
// CHECK:STDOUT:   %.loc22_25.3: <bound method> = bound_method %c.ref, %.loc22_25.2
// CHECK:STDOUT:   %.loc22_25.4: ref %ElementType.1 = temporary_storage
// CHECK:STDOUT:   %At.call: init %ElementType.1 = call %.loc22_25.3(%c.ref, <error>) to %.loc22_25.4
// CHECK:STDOUT:   %.loc22_25.5: ref %ElementType.1 = temporary %.loc22_25.4, %At.call
// CHECK:STDOUT:   %.loc22_25.6: %ElementType.1 = bind_value %.loc22_25.5
// CHECK:STDOUT:   %x: %ElementType.1 = bind_name x, %.loc22_25.6
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
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT:   %.2 => constants.%.6
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
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.3) {
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
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %.2 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.3
// CHECK:STDOUT:   %Convert => constants.%Convert.3
// CHECK:STDOUT:   %.1 => constants.%.15
// CHECK:STDOUT:   %.2 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.17) {
// CHECK:STDOUT:   %Dest => constants.%.17
// CHECK:STDOUT:   %Dest.patt => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.3.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.17
// CHECK:STDOUT:   %Dest.patt => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(@impl.3.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.19) {
// CHECK:STDOUT:   %Dest => constants.%.19
// CHECK:STDOUT:   %Dest.patt => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.4.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.19
// CHECK:STDOUT:   %Dest.patt => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(@impl.4.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.6
// CHECK:STDOUT:   %Convert => constants.%Convert.6
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %.2 => constants.%.22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(@impl.6.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(@impl.7.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT:   %Self => constants.%Self.6
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.10
// CHECK:STDOUT:   %Convert => constants.%Convert.10
// CHECK:STDOUT:   %.1 => constants.%.27
// CHECK:STDOUT:   %.2 => constants.%.28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@As.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.6.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.6(constants.%Dest, constants.%Self.5) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.17) {
// CHECK:STDOUT:   %Dest => constants.%.17
// CHECK:STDOUT:   %Dest.patt => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.9.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.17
// CHECK:STDOUT:   %Dest.patt => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(@impl.9.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.19) {
// CHECK:STDOUT:   %Dest => constants.%.19
// CHECK:STDOUT:   %Dest.patt => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.10.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.19
// CHECK:STDOUT:   %Dest.patt => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.10(@impl.10.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.6
// CHECK:STDOUT:   %Self => constants.%Self.6
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.13
// CHECK:STDOUT:   %Convert => constants.%Convert.13
// CHECK:STDOUT:   %.1 => constants.%.31
// CHECK:STDOUT:   %.2 => constants.%.32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(@impl.12.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.13(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.13(@impl.13.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_index_with_not_implemented.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_value 0 [template]
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
// CHECK:STDOUT:   %.5: type = assoc_entity_type %IndexWith.type.2, %At.type [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.6 [symbolic]
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
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %At.type: type = fn_type @At, @IndexWith(%SubscriptType, %ElementType) [symbolic = %At.type (constants.%At.type)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type) = struct_value () [symbolic = %At (constants.%At)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @IndexWith.%IndexWith.type (%IndexWith.type.2), @IndexWith.%At.type (%At.type) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: @IndexWith.%.1 (%.5) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.6)]
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
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
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
// CHECK:STDOUT:   %.loc10: i32 = int_value 0 [template = constants.%.4]
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
