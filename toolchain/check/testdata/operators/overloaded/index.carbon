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
// CHECK:STDERR: fail_invalid_subscript_type.carbon:[[@LINE+7]]:22: error: cannot implicitly convert from `i32` to `SubscriptType`
// CHECK:STDERR: let x: ElementType = c[0];
// CHECK:STDERR:                      ^~~~
// CHECK:STDERR: fail_invalid_subscript_type.carbon:[[@LINE+4]]:22: type `i32` does not implement interface `ImplicitAs`
// CHECK:STDERR: let x: ElementType = c[0];
// CHECK:STDERR:                      ^~~~
// CHECK:STDERR:
let x: ElementType = c[0];

// --- fail_index_with_not_implemented.carbon

library "[[@TEST_NAME]]";

class C { }

let c: C = {};
// CHECK:STDERR: fail_index_with_not_implemented.carbon:[[@LINE+3]]:14: error: type `C` does not support indexing
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
// CHECK:STDOUT:   %IndexWith.type: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %IndexWith: %IndexWith.type = struct_value () [template]
// CHECK:STDOUT:   %ElementType.2: type = bind_symbolic_name ElementType 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.2: type = bind_symbolic_name SubscriptType 0 [symbolic]
// CHECK:STDOUT:   %.4: type = interface_type @IndexWith, @IndexWith(%SubscriptType.2, %ElementType.2) [symbolic]
// CHECK:STDOUT:   %Self.1: @IndexWith.%.1 (%.4) = bind_symbolic_name Self 2 [symbolic]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 2 [symbolic]
// CHECK:STDOUT:   %At.type.1: type = fn_type @At.1, @IndexWith(%SubscriptType.2, %ElementType.2) [symbolic]
// CHECK:STDOUT:   %At.1: %At.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.4, %At.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %.7: type = interface_type @IndexWith, @IndexWith(%SubscriptType.1, %ElementType.1) [template]
// CHECK:STDOUT:   %At.type.2: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.2: %At.type.2 = struct_value () [template]
// CHECK:STDOUT:   %At.type.3: type = fn_type @At.1, @IndexWith(%SubscriptType.1, %ElementType.1) [template]
// CHECK:STDOUT:   %At.3: %At.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %.7, %At.type.3 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%At.2) [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct.1: %ElementType.1 = struct_value () [template]
// CHECK:STDOUT:   %struct.2: %SubscriptType.1 = struct_value () [template]
// CHECK:STDOUT:   %struct.3: %C = struct_value () [template]
// CHECK:STDOUT:   %.12: %.5 = assoc_entity element0, imports.%import_ref.6 [symbolic]
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
// CHECK:STDOUT:   %import_ref.1: %IndexWith.type = import_ref Core//prelude/operators/index, inst+7, loaded [template = constants.%IndexWith]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/index, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.3: @IndexWith.%.2 (%.5) = import_ref Core//prelude/operators/index, inst+35, loaded [symbolic = @IndexWith.%.3 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.4: @IndexWith.%At.type (%At.type.1) = import_ref Core//prelude/operators/index, inst+27, loaded [symbolic = @IndexWith.%At (constants.%At.1)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/index, inst+27, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/index, inst+27, unloaded
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
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %ElementType.decl: type = class_decl @ElementType [template = constants.%ElementType.1] {}
// CHECK:STDOUT:   %SubscriptType.decl: type = class_decl @SubscriptType [template = constants.%SubscriptType.1] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type = name_ref IndexWith, imports.%import_ref.1 [template = constants.%IndexWith]
// CHECK:STDOUT:     %SubscriptType.ref.loc8: type = name_ref SubscriptType, %SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:     %ElementType.ref.loc8: type = name_ref ElementType, %ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT:     %.loc8: type = interface_type @IndexWith, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) [template = constants.%.7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SubscriptType.ref.loc14: type = name_ref SubscriptType, %SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:   %C.ref.loc15: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %ElementType.ref.loc16: type = name_ref ElementType, %ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @IndexWith(constants.%SubscriptType.2: type, constants.%ElementType.2: type) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType 0 [symbolic = %SubscriptType (constants.%SubscriptType.2)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType 1 [symbolic = %ElementType (constants.%ElementType.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %At.type: type = fn_type @At.1, @IndexWith(%SubscriptType, %ElementType) [symbolic = %At.type (constants.%At.type.1)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type.1) = struct_value () [symbolic = %At (constants.%At.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @IndexWith.%.1 (%.4), @IndexWith.%At.type (%At.type.1) [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:   %.3: @IndexWith.%.2 (%.5) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:     .At = imports.%import_ref.3
// CHECK:STDOUT:     witness = (imports.%import_ref.4)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %.7 {
// CHECK:STDOUT:   %At.decl: %At.type.2 = fn_decl @At.2 [template = constants.%At.2] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %self.loc9_9.1: %C = param self, runtime_param0
// CHECK:STDOUT:     %self.loc9_9.2: %C = bind_name self, %self.loc9_9.1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:     %subscript.loc9_21.1: %SubscriptType.1 = param subscript, runtime_param1
// CHECK:STDOUT:     %subscript.loc9_21.2: %SubscriptType.1 = bind_name subscript, %subscript.loc9_21.1
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT:     %return.var: ref %ElementType.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = interface_witness (%At.decl) [template = constants.%.10]
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
// CHECK:STDOUT: generic fn @At.1(constants.%SubscriptType.2: type, constants.%ElementType.2: type, constants.%Self.1: @IndexWith.%.1 (%.4)) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType 0 [symbolic = %SubscriptType (constants.%SubscriptType.2)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType 1 [symbolic = %ElementType (constants.%ElementType.2)]
// CHECK:STDOUT:   %.1: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @At.1.%Self (%Self.2)](%subscript: @At.1.%SubscriptType (%SubscriptType.2)) -> @At.1.%ElementType (%ElementType.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[@impl.%self.loc9_9.2: %C](@impl.%subscript.loc9_21.2: %SubscriptType.1) -> @impl.%return.var: %ElementType.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc10_13.2: init %ElementType.1 = class_init (), @impl.%return.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc10_14: init %ElementType.1 = converted %.loc10_13.1, %.loc10_13.2 [template = constants.%struct.1]
// CHECK:STDOUT:   return %.loc10_14 to @impl.%return.var
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
// CHECK:STDOUT:   %.loc16_25.1: type = interface_type @IndexWith, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) [template = constants.%.7]
// CHECK:STDOUT:   %.loc16_25.2: %.8 = specific_constant imports.%import_ref.3, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) [template = constants.%.9]
// CHECK:STDOUT:   %At.ref: %.8 = name_ref At, %.loc16_25.2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc16_25.3: %At.type.3 = interface_witness_access @impl.%.loc8, element0 [template = constants.%At.2]
// CHECK:STDOUT:   %.loc16_25.4: <bound method> = bound_method %c.ref, %.loc16_25.3
// CHECK:STDOUT:   %.loc16_25.5: ref %ElementType.1 = temporary_storage
// CHECK:STDOUT:   %At.call: init %ElementType.1 = call %.loc16_25.4(%c.ref, %s.ref) to %.loc16_25.5
// CHECK:STDOUT:   %.loc16_25.6: ref %ElementType.1 = temporary %.loc16_25.5, %At.call
// CHECK:STDOUT:   %.loc16_25.7: %ElementType.1 = bind_value %.loc16_25.6
// CHECK:STDOUT:   %x: %ElementType.1 = bind_name x, %.loc16_25.7
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType.2, constants.%ElementType.2) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@IndexWith.%SubscriptType, @IndexWith.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@At.1.%SubscriptType, @At.1.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(constants.%SubscriptType.2, constants.%ElementType.2, constants.%Self.1) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.1
// CHECK:STDOUT:   %ElementType => constants.%ElementType.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %At.type => constants.%At.type.3
// CHECK:STDOUT:   %At => constants.%At.3
// CHECK:STDOUT:   %.2 => constants.%.8
// CHECK:STDOUT:   %.3 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(constants.%SubscriptType.1, constants.%ElementType.1, constants.%C) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.1
// CHECK:STDOUT:   %ElementType => constants.%ElementType.1
// CHECK:STDOUT:   %.1 => constants.%.7
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
// CHECK:STDOUT:   %IndexWith.type: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %IndexWith: %IndexWith.type = struct_value () [template]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType 0 [symbolic]
// CHECK:STDOUT:   %.4: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %Self.1: @IndexWith.%.1 (%.4) = bind_symbolic_name Self 2 [symbolic]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 2 [symbolic]
// CHECK:STDOUT:   %At.type.1: type = fn_type @At.1, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %At.1: %At.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.4, %At.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %.7: type = interface_type @IndexWith, @IndexWith(i32, i32) [template]
// CHECK:STDOUT:   %At.type.2: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.2: %At.type.2 = struct_value () [template]
// CHECK:STDOUT:   %At.type.3: type = fn_type @At.1, @IndexWith(i32, i32) [template]
// CHECK:STDOUT:   %At.3: %At.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %.7, %At.type.3 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%At.2) [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.12: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %tuple: %.3 = tuple_value (%.13, %.14) [template]
// CHECK:STDOUT:   %.15: %.5 = assoc_entity element0, imports.%import_ref.7 [symbolic]
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
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %IndexWith.type = import_ref Core//prelude/operators/index, inst+7, loaded [template = constants.%IndexWith]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/index, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.4: @IndexWith.%.2 (%.5) = import_ref Core//prelude/operators/index, inst+35, loaded [symbolic = @IndexWith.%.3 (constants.%.15)]
// CHECK:STDOUT:   %import_ref.5: @IndexWith.%At.type (%At.type.1) = import_ref Core//prelude/operators/index, inst+27, loaded [symbolic = @IndexWith.%At (constants.%At.1)]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/index, inst+27, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/index, inst+27, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .s = @__global_init.%s
// CHECK:STDOUT:     .e = @__global_init.%e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %int.make_type_32.loc4_7: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc4_12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_15.1: %.2 = tuple_literal (%int.make_type_32.loc4_7, %int.make_type_32.loc4_12)
// CHECK:STDOUT:     %.loc4_15.2: type = value_of_initializer %int.make_type_32.loc4_7 [template = i32]
// CHECK:STDOUT:     %.loc4_15.3: type = converted %int.make_type_32.loc4_7, %.loc4_15.2 [template = i32]
// CHECK:STDOUT:     %.loc4_15.4: type = value_of_initializer %int.make_type_32.loc4_12 [template = i32]
// CHECK:STDOUT:     %.loc4_15.5: type = converted %int.make_type_32.loc4_12, %.loc4_15.4 [template = i32]
// CHECK:STDOUT:     %.loc4_15.6: type = converted %.loc4_15.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type = name_ref IndexWith, imports.%import_ref.2 [template = constants.%IndexWith]
// CHECK:STDOUT:     %int.make_type_32.loc4_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc4_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_34.1: type = value_of_initializer %int.make_type_32.loc4_35 [template = i32]
// CHECK:STDOUT:     %.loc4_34.2: type = converted %int.make_type_32.loc4_35, %.loc4_34.1 [template = i32]
// CHECK:STDOUT:     %.loc4_34.3: type = value_of_initializer %int.make_type_32.loc4_40 [template = i32]
// CHECK:STDOUT:     %.loc4_34.4: type = converted %int.make_type_32.loc4_40, %.loc4_34.3 [template = i32]
// CHECK:STDOUT:     %.loc4_34.5: type = interface_type @IndexWith, @IndexWith(i32, i32) [template = constants.%.7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc10_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc10_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc10_17.1: %.2 = tuple_literal (%int.make_type_32.loc10_9, %int.make_type_32.loc10_14)
// CHECK:STDOUT:   %.loc10_17.2: type = value_of_initializer %int.make_type_32.loc10_9 [template = i32]
// CHECK:STDOUT:   %.loc10_17.3: type = converted %int.make_type_32.loc10_9, %.loc10_17.2 [template = i32]
// CHECK:STDOUT:   %.loc10_17.4: type = value_of_initializer %int.make_type_32.loc10_14 [template = i32]
// CHECK:STDOUT:   %.loc10_17.5: type = converted %int.make_type_32.loc10_14, %.loc10_17.4 [template = i32]
// CHECK:STDOUT:   %.loc10_17.6: type = converted %.loc10_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_32.loc18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_8.1: type = value_of_initializer %int.make_type_32.loc18 [template = i32]
// CHECK:STDOUT:   %.loc18_8.2: type = converted %int.make_type_32.loc18, %.loc18_8.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @IndexWith(constants.%SubscriptType: type, constants.%ElementType: type) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType 1 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %At.type: type = fn_type @At.1, @IndexWith(%SubscriptType, %ElementType) [symbolic = %At.type (constants.%At.type.1)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type.1) = struct_value () [symbolic = %At (constants.%At.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @IndexWith.%.1 (%.4), @IndexWith.%At.type (%At.type.1) [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:   %.3: @IndexWith.%.2 (%.5) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .At = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.3 as %.7 {
// CHECK:STDOUT:   %At.decl: %At.type.2 = fn_decl @At.2 [template = constants.%At.2] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %self.loc5_9.1: %.3 = param self, runtime_param0
// CHECK:STDOUT:     %self.loc5_9.2: %.3 = bind_name self, %self.loc5_9.1
// CHECK:STDOUT:     %int.make_type_32.loc5_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_32.1: type = value_of_initializer %int.make_type_32.loc5_32 [template = i32]
// CHECK:STDOUT:     %.loc5_32.2: type = converted %int.make_type_32.loc5_32, %.loc5_32.1 [template = i32]
// CHECK:STDOUT:     %subscript.loc5_21.1: i32 = param subscript, runtime_param1
// CHECK:STDOUT:     %subscript.loc5_21.2: i32 = bind_name subscript, %subscript.loc5_21.1
// CHECK:STDOUT:     %int.make_type_32.loc5_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_40.1: type = value_of_initializer %int.make_type_32.loc5_40 [template = i32]
// CHECK:STDOUT:     %.loc5_40.2: type = converted %int.make_type_32.loc5_40, %.loc5_40.1 [template = i32]
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4: <witness> = interface_witness (%At.decl) [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = %.loc4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @At.1(constants.%SubscriptType: type, constants.%ElementType: type, constants.%Self.1: @IndexWith.%.1 (%.4)) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType 1 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:   %.1: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @At.1.%Self (%Self.2)](%subscript: @At.1.%SubscriptType (%SubscriptType)) -> @At.1.%ElementType (%ElementType);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[@impl.%self.loc5_9.2: %.3](@impl.%subscript.loc5_21.2: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %.3 = name_ref self, @impl.%self.loc5_9.2
// CHECK:STDOUT:   %.loc6_17: i32 = int_literal 0 [template = constants.%.12]
// CHECK:STDOUT:   %.loc6_16: i32 = tuple_access %self.ref, element0
// CHECK:STDOUT:   return %.loc6_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_22: i32 = int_literal 1 [template = constants.%.13]
// CHECK:STDOUT:   %.loc10_25: i32 = int_literal 5 [template = constants.%.14]
// CHECK:STDOUT:   %.loc10_26: %.3 = tuple_literal (%.loc10_22, %.loc10_25)
// CHECK:STDOUT:   %tuple: %.3 = tuple_value (%.loc10_22, %.loc10_25) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc10_27: %.3 = converted %.loc10_26, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %s: %.3 = bind_name s, %.loc10_27
// CHECK:STDOUT:   %s.ref: %.3 = name_ref s, %s
// CHECK:STDOUT:   %.loc18_16: i32 = int_literal 0 [template = constants.%.12]
// CHECK:STDOUT:   %.loc18_17.1: type = interface_type @IndexWith, @IndexWith(i32, i32) [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_17.2: %.8 = specific_constant imports.%import_ref.4, @IndexWith(i32, i32) [template = constants.%.9]
// CHECK:STDOUT:   %At.ref: %.8 = name_ref At, %.loc18_17.2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc18_17.3: %At.type.3 = interface_witness_access @impl.%.loc4, element0 [template = constants.%At.2]
// CHECK:STDOUT:   %.loc18_17.4: <bound method> = bound_method %s.ref, %.loc18_17.3
// CHECK:STDOUT:   %At.call: init i32 = call %.loc18_17.4(%s.ref, %.loc18_16)
// CHECK:STDOUT:   %.loc18_18.1: i32 = value_of_initializer %At.call
// CHECK:STDOUT:   %.loc18_18.2: i32 = converted %At.call, %.loc18_18.1
// CHECK:STDOUT:   %e: i32 = bind_name e, %.loc18_18.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType, constants.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@IndexWith.%SubscriptType, @IndexWith.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@At.1.%SubscriptType, @At.1.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(constants.%SubscriptType, constants.%ElementType, constants.%Self.1) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(i32, i32) {
// CHECK:STDOUT:   %SubscriptType => i32
// CHECK:STDOUT:   %ElementType => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %At.type => constants.%At.type.3
// CHECK:STDOUT:   %At => constants.%At.3
// CHECK:STDOUT:   %.2 => constants.%.8
// CHECK:STDOUT:   %.3 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(i32, i32, constants.%.3) {
// CHECK:STDOUT:   %SubscriptType => i32
// CHECK:STDOUT:   %ElementType => i32
// CHECK:STDOUT:   %.1 => constants.%.7
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
// CHECK:STDOUT:   %IndexWith.type: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %IndexWith: %IndexWith.type = struct_value () [template]
// CHECK:STDOUT:   %ElementType.2: type = bind_symbolic_name ElementType 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.2: type = bind_symbolic_name SubscriptType 0 [symbolic]
// CHECK:STDOUT:   %.4: type = interface_type @IndexWith, @IndexWith(%SubscriptType.2, %ElementType.2) [symbolic]
// CHECK:STDOUT:   %Self.1: @IndexWith.%.1 (%.4) = bind_symbolic_name Self 2 [symbolic]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 2 [symbolic]
// CHECK:STDOUT:   %At.type.1: type = fn_type @At.1, @IndexWith(%SubscriptType.2, %ElementType.2) [symbolic]
// CHECK:STDOUT:   %At.1: %At.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.4, %At.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %.7: type = interface_type @IndexWith, @IndexWith(%SubscriptType.1, %ElementType.1) [template]
// CHECK:STDOUT:   %At.type.2: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.2: %At.type.2 = struct_value () [template]
// CHECK:STDOUT:   %At.type.3: type = fn_type @At.1, @IndexWith(%SubscriptType.1, %ElementType.1) [template]
// CHECK:STDOUT:   %At.3: %At.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %.7, %At.type.3 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%At.2) [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct.1: %ElementType.1 = struct_value () [template]
// CHECK:STDOUT:   %struct.2: %C = struct_value () [template]
// CHECK:STDOUT:   %.12: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic]
// CHECK:STDOUT:   %.13: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.3: @ImplicitAs.%.1 (%.13) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Self.4: %.13 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.14: type = assoc_entity_type %.13, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.15: %.14 = assoc_entity element0, imports.%import_ref.10 [symbolic]
// CHECK:STDOUT:   %.16: type = interface_type @ImplicitAs, @ImplicitAs(%SubscriptType.1) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%SubscriptType.1) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.17: type = assoc_entity_type %.16, %Convert.type.2 [template]
// CHECK:STDOUT:   %.18: %.17 = assoc_entity element0, imports.%import_ref.10 [template]
// CHECK:STDOUT:   %.19: %.14 = assoc_entity element0, imports.%import_ref.11 [symbolic]
// CHECK:STDOUT:   %.20: %.5 = assoc_entity element0, imports.%import_ref.12 [symbolic]
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
// CHECK:STDOUT:   %import_ref.1: %IndexWith.type = import_ref Core//prelude/operators/index, inst+7, loaded [template = constants.%IndexWith]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/index, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.3: @IndexWith.%.2 (%.5) = import_ref Core//prelude/operators/index, inst+35, loaded [symbolic = @IndexWith.%.3 (constants.%.20)]
// CHECK:STDOUT:   %import_ref.4: @IndexWith.%At.type (%At.type.1) = import_ref Core//prelude/operators/index, inst+27, loaded [symbolic = @IndexWith.%At (constants.%At.1)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/index, inst+27, unloaded
// CHECK:STDOUT:   %import_ref.6: %ImplicitAs.type = import_ref Core//prelude/operators/as, inst+37, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.8: @ImplicitAs.%.2 (%.14) = import_ref Core//prelude/operators/as, inst+59, loaded [symbolic = @ImplicitAs.%.3 (constants.%.19)]
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//prelude/operators/index, inst+27, unloaded
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
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %ElementType.decl: type = class_decl @ElementType [template = constants.%ElementType.1] {}
// CHECK:STDOUT:   %SubscriptType.decl: type = class_decl @SubscriptType [template = constants.%SubscriptType.1] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type = name_ref IndexWith, imports.%import_ref.1 [template = constants.%IndexWith]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, %SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:     %ElementType.ref.loc8: type = name_ref ElementType, %ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT:     %.loc8: type = interface_type @IndexWith, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) [template = constants.%.7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref.loc14: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %ElementType.ref.loc22: type = name_ref ElementType, %ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @IndexWith(constants.%SubscriptType.2: type, constants.%ElementType.2: type) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType 0 [symbolic = %SubscriptType (constants.%SubscriptType.2)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType 1 [symbolic = %ElementType (constants.%ElementType.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %At.type: type = fn_type @At.1, @IndexWith(%SubscriptType, %ElementType) [symbolic = %At.type (constants.%At.type.1)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type.1) = struct_value () [symbolic = %At (constants.%At.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @IndexWith.%.1 (%.4), @IndexWith.%At.type (%At.type.1) [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:   %.3: @IndexWith.%.2 (%.5) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.3 (constants.%.6)]
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
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.13)]
// CHECK:STDOUT:   %Self: %.13 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @ImplicitAs.%.1 (%.13), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.14)]
// CHECK:STDOUT:   %.3: @ImplicitAs.%.2 (%.14) = assoc_entity element0, imports.%import_ref.10 [symbolic = %.3 (constants.%.15)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.7
// CHECK:STDOUT:     .Convert = imports.%import_ref.8
// CHECK:STDOUT:     witness = (imports.%import_ref.9)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %.7 {
// CHECK:STDOUT:   %At.decl: %At.type.2 = fn_decl @At.2 [template = constants.%At.2] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %self.loc9_9.1: %C = param self, runtime_param0
// CHECK:STDOUT:     %self.loc9_9.2: %C = bind_name self, %self.loc9_9.1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:     %subscript.loc9_21.1: %SubscriptType.1 = param subscript, runtime_param1
// CHECK:STDOUT:     %subscript.loc9_21.2: %SubscriptType.1 = bind_name subscript, %subscript.loc9_21.1
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT:     %return.var: ref %ElementType.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = interface_witness (%At.decl) [template = constants.%.10]
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
// CHECK:STDOUT: generic fn @At.1(constants.%SubscriptType.2: type, constants.%ElementType.2: type, constants.%Self.1: @IndexWith.%.1 (%.4)) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType 0 [symbolic = %SubscriptType (constants.%SubscriptType.2)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType 1 [symbolic = %ElementType (constants.%ElementType.2)]
// CHECK:STDOUT:   %.1: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @At.1.%Self (%Self.2)](%subscript: @At.1.%SubscriptType (%SubscriptType.2)) -> @At.1.%ElementType (%ElementType.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[@impl.%self.loc9_9.2: %C](@impl.%subscript.loc9_21.2: %SubscriptType.1) -> @impl.%return.var: %ElementType.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc10_13.2: init %ElementType.1 = class_init (), @impl.%return.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc10_14: init %ElementType.1 = converted %.loc10_13.1, %.loc10_13.2 [template = constants.%struct.1]
// CHECK:STDOUT:   return %.loc10_14 to @impl.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.3: @ImplicitAs.%.1 (%.13)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.13)]
// CHECK:STDOUT:   %Self: %.13 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.4)]() -> @Convert.%Dest (%Dest);
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
// CHECK:STDOUT:   %.loc22_24: i32 = int_literal 0 [template = constants.%.12]
// CHECK:STDOUT:   %.loc22_25.1: type = interface_type @ImplicitAs, @ImplicitAs(constants.%SubscriptType.1) [template = constants.%.16]
// CHECK:STDOUT:   %.loc22_25.2: %.17 = specific_constant imports.%import_ref.8, @ImplicitAs(constants.%SubscriptType.1) [template = constants.%.18]
// CHECK:STDOUT:   %Convert.ref: %.17 = name_ref Convert, %.loc22_25.2 [template = constants.%.18]
// CHECK:STDOUT:   %.loc22_25.3: %SubscriptType.1 = converted %.loc22_24, <error> [template = <error>]
// CHECK:STDOUT:   %.loc22_25.4: type = interface_type @IndexWith, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) [template = constants.%.7]
// CHECK:STDOUT:   %.loc22_25.5: %.8 = specific_constant imports.%import_ref.3, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) [template = constants.%.9]
// CHECK:STDOUT:   %At.ref: %.8 = name_ref At, %.loc22_25.5 [template = constants.%.9]
// CHECK:STDOUT:   %.loc22_25.6: %At.type.3 = interface_witness_access @impl.%.loc8, element0 [template = constants.%At.2]
// CHECK:STDOUT:   %.loc22_25.7: <bound method> = bound_method %c.ref, %.loc22_25.6
// CHECK:STDOUT:   %.loc22_25.8: ref %ElementType.1 = temporary_storage
// CHECK:STDOUT:   %At.call: init %ElementType.1 = call %.loc22_25.7(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc22_25.9: ref %ElementType.1 = temporary_storage
// CHECK:STDOUT:   %.loc22_25.10: ref %ElementType.1 = temporary %.loc22_25.9, %At.call
// CHECK:STDOUT:   %.loc22_25.11: %ElementType.1 = bind_value %.loc22_25.10
// CHECK:STDOUT:   %x: %ElementType.1 = bind_name x, %.loc22_25.11
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType.2, constants.%ElementType.2) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@IndexWith.%SubscriptType, @IndexWith.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@At.1.%SubscriptType, @At.1.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(constants.%SubscriptType.2, constants.%ElementType.2, constants.%Self.1) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.2
// CHECK:STDOUT:   %ElementType => constants.%ElementType.2
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.1
// CHECK:STDOUT:   %ElementType => constants.%ElementType.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %At.type => constants.%At.type.3
// CHECK:STDOUT:   %At => constants.%At.3
// CHECK:STDOUT:   %.2 => constants.%.8
// CHECK:STDOUT:   %.3 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(constants.%SubscriptType.1, constants.%ElementType.1, constants.%C) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType.1
// CHECK:STDOUT:   %ElementType => constants.%ElementType.1
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %Self => constants.%C
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
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%SubscriptType.1) {
// CHECK:STDOUT:   %Dest => constants.%SubscriptType.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.2 => constants.%.17
// CHECK:STDOUT:   %.3 => constants.%.18
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
// CHECK:STDOUT:   %IndexWith.type: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %IndexWith: %IndexWith.type = struct_value () [template]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType 0 [symbolic]
// CHECK:STDOUT:   %.6: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %Self.1: @IndexWith.%.1 (%.6) = bind_symbolic_name Self 2 [symbolic]
// CHECK:STDOUT:   %Self.2: %.6 = bind_symbolic_name Self 2 [symbolic]
// CHECK:STDOUT:   %At.type: type = fn_type @At, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %At: %At.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.6, %At.type [symbolic]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.6 [symbolic]
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
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %IndexWith.type = import_ref Core//prelude/operators/index, inst+7, loaded [template = constants.%IndexWith]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/index, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/index, inst+35, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/index, inst+27, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/index, inst+27, unloaded
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
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc10_8.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc10_8.2: type = converted %int.make_type_32, %.loc10_8.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @IndexWith(constants.%SubscriptType: type, constants.%ElementType: type) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType 1 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self: %.6 = bind_symbolic_name Self 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %At.type: type = fn_type @At, @IndexWith(%SubscriptType, %ElementType) [symbolic = %At.type (constants.%At.type)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type) = struct_value () [symbolic = %At (constants.%At)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @IndexWith.%.1 (%.6), @IndexWith.%At.type (%At.type) [symbolic = %.2 (constants.%.7)]
// CHECK:STDOUT:   %.3: @IndexWith.%.2 (%.7) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.3 (constants.%.8)]
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
// CHECK:STDOUT: generic fn @At(constants.%SubscriptType: type, constants.%ElementType: type, constants.%Self.1: @IndexWith.%.1 (%.6)) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %ElementType: type = bind_symbolic_name ElementType 1 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:   %.1: type = interface_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self: %.6 = bind_symbolic_name Self 2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @At.%Self (%Self.2)](%subscript: @At.%SubscriptType (%SubscriptType)) -> @At.%ElementType (%ElementType);
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
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@IndexWith.%SubscriptType, @IndexWith.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@At.%SubscriptType, @At.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At(constants.%SubscriptType, constants.%ElementType, constants.%Self.1) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
