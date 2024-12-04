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
// CHECK:STDERR: fail_invalid_subscript_type.carbon:[[@LINE+7]]:22: error: cannot implicitly convert from `Core.IntLiteral` to `SubscriptType` [ImplicitAsConversionFailure]
// CHECK:STDERR: let x: ElementType = c[0];
// CHECK:STDERR:                      ^~~~
// CHECK:STDERR: fail_invalid_subscript_type.carbon:[[@LINE+4]]:22: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(SubscriptType)` [MissingImplInMemberAccessNote]
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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %ElementType.1: type = class_type @ElementType [template]
// CHECK:STDOUT:   %SubscriptType.1: type = class_type @SubscriptType [template]
// CHECK:STDOUT:   %IndexWith.type.1: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.1 = struct_value () [template]
// CHECK:STDOUT:   %IndexWith.type.3: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.1, %ElementType.1)> [template]
// CHECK:STDOUT:   %At.type.2: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.2: %At.type.2 = struct_value () [template]
// CHECK:STDOUT:   %At.type.3: type = fn_type @At.1, @IndexWith(%SubscriptType.1, %ElementType.1) [template]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%At.2) [template]
// CHECK:STDOUT:   %ElementType.val: %ElementType.1 = struct_value () [template]
// CHECK:STDOUT:   %SubscriptType.val: %SubscriptType.1 = struct_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IndexWith = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IndexWith.type.1 = import_ref Core//prelude/operators/index, inst+13, loaded [template = constants.%IndexWith.generic]
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
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.1 = name_ref IndexWith, imports.%import_ref.1 [template = constants.%IndexWith.generic]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1)> [template = constants.%IndexWith.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SubscriptType.ref: type = name_ref SubscriptType, %SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %ElementType.ref: type = name_ref ElementType, %ElementType.decl [template = constants.%ElementType.1]
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
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%At.decl) [template = constants.%interface]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ElementType {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ElementType.1
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SubscriptType.1
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %C](%subscript.param_patt: %SubscriptType.1) -> %return.param_patt: %ElementType.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_13.2: init %ElementType.1 = class_init (), %return [template = constants.%ElementType.val]
// CHECK:STDOUT:   %.loc10_14: init %ElementType.1 = converted %.loc10_13.1, %.loc10_13.2 [template = constants.%ElementType.val]
// CHECK:STDOUT:   return %.loc10_14 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc14_25.2: ref %SubscriptType.1 = temporary_storage
// CHECK:STDOUT:   %.loc14_25.3: init %SubscriptType.1 = class_init (), %.loc14_25.2 [template = constants.%SubscriptType.val]
// CHECK:STDOUT:   %.loc14_25.4: ref %SubscriptType.1 = temporary %.loc14_25.2, %.loc14_25.3
// CHECK:STDOUT:   %.loc14_26.1: ref %SubscriptType.1 = converted %.loc14_25.1, %.loc14_25.4
// CHECK:STDOUT:   %.loc14_26.2: %SubscriptType.1 = bind_value %.loc14_26.1
// CHECK:STDOUT:   %s: %SubscriptType.1 = bind_name s, %.loc14_26.2
// CHECK:STDOUT:   %.loc15_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc15_13.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc15_13.3: init %C = class_init (), %.loc15_13.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc15_13.4: ref %C = temporary %.loc15_13.2, %.loc15_13.3
// CHECK:STDOUT:   %.loc15_14.1: ref %C = converted %.loc15_13.1, %.loc15_13.4
// CHECK:STDOUT:   %.loc15_14.2: %C = bind_value %.loc15_14.1
// CHECK:STDOUT:   %c: %C = bind_name c, %.loc15_14.2
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %s.ref: %SubscriptType.1 = name_ref s, %s
// CHECK:STDOUT:   %impl.elem0: %At.type.3 = interface_witness_access constants.%interface, element0 [template = constants.%At.2]
// CHECK:STDOUT:   %At.bound: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %.loc16_25.1: ref %ElementType.1 = temporary_storage
// CHECK:STDOUT:   %At.call: init %ElementType.1 = call %At.bound(%c.ref, %s.ref) to %.loc16_25.1
// CHECK:STDOUT:   %.loc16_25.2: ref %ElementType.1 = temporary %.loc16_25.1, %At.call
// CHECK:STDOUT:   %.loc16_25.3: %ElementType.1 = bind_value %.loc16_25.2
// CHECK:STDOUT:   %x: %ElementType.1 = bind_name x, %.loc16_25.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overloaded_builtin.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %IndexWith.type.1: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.1 = struct_value () [template]
// CHECK:STDOUT:   %IndexWith.type.3: type = facet_type <@IndexWith, @IndexWith(%i32, %i32)> [template]
// CHECK:STDOUT:   %At.type.2: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.2: %At.type.2 = struct_value () [template]
// CHECK:STDOUT:   %At.type.3: type = fn_type @At.1, @IndexWith(%i32, %i32) [template]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness (%At.2) [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.10: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_5.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int_1.2, %int_5.2) [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_0.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .IndexWith = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.7
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %IndexWith.type.1 = import_ref Core//prelude/operators/index, inst+13, loaded [template = constants.%IndexWith.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .s = @__global_init.%s
// CHECK:STDOUT:     .e = @__global_init.%e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %int_32.loc4_7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_7: init type = call constants.%Int(%int_32.loc4_7) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_12: init type = call constants.%Int(%int_32.loc4_12) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_15.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc4_7, %int.make_type_signed.loc4_12)
// CHECK:STDOUT:     %.loc4_15.2: type = value_of_initializer %int.make_type_signed.loc4_7 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_15.3: type = converted %int.make_type_signed.loc4_7, %.loc4_15.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_15.4: type = value_of_initializer %int.make_type_signed.loc4_12 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_15.5: type = converted %int.make_type_signed.loc4_12, %.loc4_15.4 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_15.6: type = converted %.loc4_15.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.1 = name_ref IndexWith, imports.%import_ref.2 [template = constants.%IndexWith.generic]
// CHECK:STDOUT:     %int_32.loc4_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_35: init type = call constants.%Int(%int_32.loc4_35) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_40: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_40: init type = call constants.%Int(%int_32.loc4_40) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_43.1: type = value_of_initializer %int.make_type_signed.loc4_35 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_43.2: type = converted %int.make_type_signed.loc4_35, %.loc4_43.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_43.3: type = value_of_initializer %int.make_type_signed.loc4_40 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_43.4: type = converted %int.make_type_signed.loc4_40, %.loc4_43.3 [template = constants.%i32]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%i32, constants.%i32)> [template = constants.%IndexWith.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc10_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc10_9: init type = call constants.%Int(%int_32.loc10_9) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc10_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc10_14: init type = call constants.%Int(%int_32.loc10_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc10_9, %int.make_type_signed.loc10_14)
// CHECK:STDOUT:   %.loc10_17.2: type = value_of_initializer %int.make_type_signed.loc10_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_17.3: type = converted %int.make_type_signed.loc10_9, %.loc10_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_17.4: type = value_of_initializer %int.make_type_signed.loc10_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_17.5: type = converted %int.make_type_signed.loc10_14, %.loc10_17.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_17.6: type = converted %.loc10_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%int_32.loc11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_8.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_8.2: type = converted %int.make_type_signed.loc11, %.loc11_8.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc4_15.6 as %IndexWith.type {
// CHECK:STDOUT:   %At.decl: %At.type.2 = fn_decl @At.2 [template = constants.%At.2] {
// CHECK:STDOUT:     %self.patt: %tuple.type.2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %tuple.type.2 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %subscript.patt: %i32 = binding_pattern subscript
// CHECK:STDOUT:     %subscript.param_patt: %i32 = value_param_pattern %subscript.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.1.%.loc4_15.6 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %int_32.loc5_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5_32: init type = call constants.%Int(%int_32.loc5_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_32.1: type = value_of_initializer %int.make_type_signed.loc5_32 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_32.2: type = converted %int.make_type_signed.loc5_32, %.loc5_32.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_40: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5_40: init type = call constants.%Int(%int_32.loc5_40) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_40.1: type = value_of_initializer %int.make_type_signed.loc5_40 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_40.2: type = converted %int.make_type_signed.loc5_40, %.loc5_40.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: %tuple.type.2 = value_param runtime_param0
// CHECK:STDOUT:     %self: %tuple.type.2 = bind_name self, %self.param
// CHECK:STDOUT:     %subscript.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %subscript: %i32 = bind_name subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%At.decl) [template = constants.%interface.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %tuple.type.2](%subscript.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %tuple.type.2 = name_ref self, %self
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %tuple.elem0: %i32 = tuple_access %self.ref, element0
// CHECK:STDOUT:   return %tuple.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc10_26.1: %tuple.type.3 = tuple_literal (%int_1, %int_5)
// CHECK:STDOUT:   %impl.elem0.loc10_26.1: %Convert.type.2 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc10_26.1: <bound method> = bound_method %int_1, %impl.elem0.loc10_26.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_26.1: <specific function> = specific_function %Convert.bound.loc10_26.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_26.1: init %i32 = call %Convert.specific_fn.loc10_26.1(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_26.2: %i32 = value_of_initializer %int.convert_checked.loc10_26.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_26.3: %i32 = converted %int_1, %.loc10_26.2 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc10_26.2: %Convert.type.2 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc10_26.2: <bound method> = bound_method %int_5, %impl.elem0.loc10_26.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_26.2: <specific function> = specific_function %Convert.bound.loc10_26.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc10_26.2: init %i32 = call %Convert.specific_fn.loc10_26.2(%int_5) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc10_26.4: %i32 = value_of_initializer %int.convert_checked.loc10_26.2 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc10_26.5: %i32 = converted %int_5, %.loc10_26.4 [template = constants.%int_5.2]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.loc10_26.3, %.loc10_26.5) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc10_27: %tuple.type.2 = converted %.loc10_26.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %s: %tuple.type.2 = bind_name s, %.loc10_27
// CHECK:STDOUT:   %s.ref: %tuple.type.2 = name_ref s, %s
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc11_17.1: %Convert.type.2 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_0, %impl.elem0.loc11_17.1 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %Convert.specific_fn.loc11(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_17.1: %i32 = value_of_initializer %int.convert_checked.loc11 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_17.2: %i32 = converted %int_0, %.loc11_17.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc11_17.2: %At.type.3 = interface_witness_access constants.%interface.1, element0 [template = constants.%At.2]
// CHECK:STDOUT:   %At.bound: <bound method> = bound_method %s.ref, %impl.elem0.loc11_17.2
// CHECK:STDOUT:   %At.call: init %i32 = call %At.bound(%s.ref, %.loc11_17.2)
// CHECK:STDOUT:   %.loc11_18.1: %i32 = value_of_initializer %At.call
// CHECK:STDOUT:   %.loc11_18.2: %i32 = converted %At.call, %.loc11_18.1
// CHECK:STDOUT:   %e: %i32 = bind_name e, %.loc11_18.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_subscript_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %ElementType.1: type = class_type @ElementType [template]
// CHECK:STDOUT:   %SubscriptType.1: type = class_type @SubscriptType [template]
// CHECK:STDOUT:   %IndexWith.type.1: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.1 = struct_value () [template]
// CHECK:STDOUT:   %IndexWith.type.3: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.1, %ElementType.1)> [template]
// CHECK:STDOUT:   %At.type.2: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.2: %At.type.2 = struct_value () [template]
// CHECK:STDOUT:   %At.type.3: type = fn_type @At.1, @IndexWith(%SubscriptType.1, %ElementType.1) [template]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness (%At.2) [template]
// CHECK:STDOUT:   %ElementType.val: %ElementType.1 = struct_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IndexWith = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IndexWith.type.1 = import_ref Core//prelude/operators/index, inst+13, loaded [template = constants.%IndexWith.generic]
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
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.1 = name_ref IndexWith, imports.%import_ref.1 [template = constants.%IndexWith.generic]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [template = constants.%SubscriptType.1]
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [template = constants.%ElementType.1]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%SubscriptType.1, constants.%ElementType.1)> [template = constants.%IndexWith.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %ElementType.ref: type = name_ref ElementType, %ElementType.decl [template = constants.%ElementType.1]
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
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%At.decl) [template = constants.%interface.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ElementType {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ElementType.1
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SubscriptType.1
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %C](%subscript.param_patt: %SubscriptType.1) -> %return.param_patt: %ElementType.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_13.2: init %ElementType.1 = class_init (), %return [template = constants.%ElementType.val]
// CHECK:STDOUT:   %.loc10_14: init %ElementType.1 = converted %.loc10_13.1, %.loc10_13.2 [template = constants.%ElementType.val]
// CHECK:STDOUT:   return %.loc10_14 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc14_13.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc14_13.3: init %C = class_init (), %.loc14_13.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc14_13.4: ref %C = temporary %.loc14_13.2, %.loc14_13.3
// CHECK:STDOUT:   %.loc14_14.1: ref %C = converted %.loc14_13.1, %.loc14_13.4
// CHECK:STDOUT:   %.loc14_14.2: %C = bind_value %.loc14_14.1
// CHECK:STDOUT:   %c: %C = bind_name c, %.loc14_14.2
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc22_25.1: %SubscriptType.1 = converted %int_0, <error> [template = <error>]
// CHECK:STDOUT:   %impl.elem0: %At.type.3 = interface_witness_access constants.%interface.1, element0 [template = constants.%At.2]
// CHECK:STDOUT:   %At.bound: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %.loc22_25.2: ref %ElementType.1 = temporary_storage
// CHECK:STDOUT:   %At.call: init %ElementType.1 = call %At.bound(%c.ref, <error>) to %.loc22_25.2
// CHECK:STDOUT:   %.loc22_25.3: ref %ElementType.1 = temporary %.loc22_25.2, %At.call
// CHECK:STDOUT:   %.loc22_25.4: %ElementType.1 = bind_value %.loc22_25.3
// CHECK:STDOUT:   %x: %ElementType.1 = bind_name x, %.loc22_25.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_index_with_not_implemented.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .IndexWith = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_8.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_8.2: type = converted %int.make_type_signed, %.loc10_8.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc6_13.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc6_13.3: init %C = class_init (), %.loc6_13.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc6_13.4: ref %C = temporary %.loc6_13.2, %.loc6_13.3
// CHECK:STDOUT:   %.loc6_14.1: ref %C = converted %.loc6_13.1, %.loc6_13.4
// CHECK:STDOUT:   %.loc6_14.2: %C = bind_value %.loc6_14.1
// CHECK:STDOUT:   %c: %C = bind_name c, %.loc6_14.2
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %x: %i32 = bind_name x, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
