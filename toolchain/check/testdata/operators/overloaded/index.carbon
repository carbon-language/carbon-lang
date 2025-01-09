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
// CHECK:STDERR: fail_invalid_subscript_type.carbon:[[@LINE+4]]:22: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(SubscriptType)` [MissingImplInMemberAccessNote]
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
// CHECK:STDOUT:   %ElementType.e6b: type = class_type @ElementType [template]
// CHECK:STDOUT:   %SubscriptType.8ee: type = class_type @SubscriptType [template]
// CHECK:STDOUT:   %IndexWith.type.504: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.504 = struct_value () [template]
// CHECK:STDOUT:   %IndexWith.type.2d3: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.8ee, %ElementType.e6b)> [template]
// CHECK:STDOUT:   %At.type.38e: type = fn_type @At.1, @IndexWith(%SubscriptType.8ee, %ElementType.e6b) [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%At.decl) [template]
// CHECK:STDOUT:   %At.type.afe: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.9bf: %At.type.afe = struct_value () [template]
// CHECK:STDOUT:   %ElementType.val: %ElementType.e6b = struct_value () [template]
// CHECK:STDOUT:   %SubscriptType.val: %SubscriptType.8ee = struct_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IndexWith = %import_ref.76d
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.76d: %IndexWith.type.504 = import_ref Core//prelude/operators/index, IndexWith, loaded [template = constants.%IndexWith.generic]
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
// CHECK:STDOUT:   %ElementType.decl: type = class_decl @ElementType [template = constants.%ElementType.e6b] {} {}
// CHECK:STDOUT:   %SubscriptType.decl: type = class_decl @SubscriptType [template = constants.%SubscriptType.8ee] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.504 = name_ref IndexWith, imports.%import_ref.76d [template = constants.%IndexWith.generic]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [template = constants.%SubscriptType.8ee]
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [template = constants.%ElementType.e6b]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%SubscriptType.8ee, constants.%ElementType.e6b)> [template = constants.%IndexWith.type.2d3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%At.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %IndexWith.type {
// CHECK:STDOUT:   %At.decl: %At.type.afe = fn_decl @At.2 [template = constants.%At.9bf] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %subscript.patt: %SubscriptType.8ee = binding_pattern subscript
// CHECK:STDOUT:     %subscript.param_patt: %SubscriptType.8ee = value_param_pattern %subscript.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %ElementType.e6b = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ElementType.e6b = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [template = constants.%ElementType.e6b]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%C.ref [template = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %subscript.param: %SubscriptType.8ee = value_param runtime_param1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [template = constants.%SubscriptType.8ee]
// CHECK:STDOUT:     %subscript: %SubscriptType.8ee = bind_name subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %ElementType.e6b = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %ElementType.e6b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = file.%impl_witness
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
// CHECK:STDOUT:   .Self = constants.%ElementType.e6b
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SubscriptType.8ee
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %C](%subscript.param_patt: %SubscriptType.8ee) -> %return.param_patt: %ElementType.e6b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_13.2: init %ElementType.e6b = class_init (), %return [template = constants.%ElementType.val]
// CHECK:STDOUT:   %.loc10_14: init %ElementType.e6b = converted %.loc10_13.1, %.loc10_13.2 [template = constants.%ElementType.val]
// CHECK:STDOUT:   return %.loc10_14 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc14_25.2: ref %SubscriptType.8ee = temporary_storage
// CHECK:STDOUT:   %.loc14_25.3: init %SubscriptType.8ee = class_init (), %.loc14_25.2 [template = constants.%SubscriptType.val]
// CHECK:STDOUT:   %.loc14_25.4: ref %SubscriptType.8ee = temporary %.loc14_25.2, %.loc14_25.3
// CHECK:STDOUT:   %.loc14_26.1: ref %SubscriptType.8ee = converted %.loc14_25.1, %.loc14_25.4
// CHECK:STDOUT:   %.loc14_26.2: %SubscriptType.8ee = bind_value %.loc14_26.1
// CHECK:STDOUT:   %s: %SubscriptType.8ee = bind_name s, %.loc14_26.2
// CHECK:STDOUT:   %.loc15_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc15_13.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc15_13.3: init %C = class_init (), %.loc15_13.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc15_13.4: ref %C = temporary %.loc15_13.2, %.loc15_13.3
// CHECK:STDOUT:   %.loc15_14.1: ref %C = converted %.loc15_13.1, %.loc15_13.4
// CHECK:STDOUT:   %.loc15_14.2: %C = bind_value %.loc15_14.1
// CHECK:STDOUT:   %c: %C = bind_name c, %.loc15_14.2
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %s.ref: %SubscriptType.8ee = name_ref s, %s
// CHECK:STDOUT:   %impl.elem0: %At.type.38e = impl_witness_access constants.%impl_witness, element0 [template = constants.%At.9bf]
// CHECK:STDOUT:   %At.bound: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %.loc16_25.1: ref %ElementType.e6b = temporary_storage
// CHECK:STDOUT:   %At.call: init %ElementType.e6b = call %At.bound(%c.ref, %s.ref) to %.loc16_25.1
// CHECK:STDOUT:   %.loc16_25.2: ref %ElementType.e6b = temporary %.loc16_25.1, %At.call
// CHECK:STDOUT:   %.loc16_25.3: %ElementType.e6b = bind_value %.loc16_25.2
// CHECK:STDOUT:   %x: %ElementType.e6b = bind_name x, %.loc16_25.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overloaded_builtin.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.471: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %IndexWith.type.504: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.504 = struct_value () [template]
// CHECK:STDOUT:   %IndexWith.type.d22: type = facet_type <@IndexWith, @IndexWith(%i32, %i32)> [template]
// CHECK:STDOUT:   %At.type.740: type = fn_type @At.1, @IndexWith(%i32, %i32) [template]
// CHECK:STDOUT:   %impl_witness.d5f: <witness> = impl_witness (@impl.1.%At.decl) [template]
// CHECK:STDOUT:   %At.type.37e: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.065: %At.type.37e = struct_value () [template]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.afd: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b73: <specific function> = specific_function %Convert.bound.afd, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.b33: <bound method> = bound_method %int_5.64b, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.89d: <specific function> = specific_function %Convert.bound.b33, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.70b: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.471 = tuple_value (%int_1.c60, %int_5.70b) [template]
// CHECK:STDOUT:   %Convert.bound.b41: <bound method> = bound_method %int_0.5c6, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.c35: <specific function> = specific_function %Convert.bound.b41, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.f61: %i32 = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .IndexWith = %import_ref.76d
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.76d: %IndexWith.type.504 = import_ref Core//prelude/operators/index, IndexWith, loaded [template = constants.%IndexWith.generic]
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
// CHECK:STDOUT:     %i32.loc4_7: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_12: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_15.1: %tuple.type.24b = tuple_literal (%i32.loc4_7, %i32.loc4_12)
// CHECK:STDOUT:     %.loc4_15.2: type = converted %.loc4_15.1, constants.%tuple.type.471 [template = constants.%tuple.type.471]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.504 = name_ref IndexWith, imports.%import_ref.76d [template = constants.%IndexWith.generic]
// CHECK:STDOUT:     %int_32.loc4_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_35: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_40: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_40: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%i32, constants.%i32)> [template = constants.%IndexWith.type.d22]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.1.%At.decl) [template = constants.%impl_witness.d5f]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc4_15.2 as %IndexWith.type {
// CHECK:STDOUT:   %At.decl: %At.type.37e = fn_decl @At.2 [template = constants.%At.065] {
// CHECK:STDOUT:     %self.patt: %tuple.type.471 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %tuple.type.471 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %subscript.patt: %i32 = binding_pattern subscript
// CHECK:STDOUT:     %subscript.param_patt: %i32 = value_param_pattern %subscript.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_40: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_40: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %self.param: %tuple.type.471 = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.1.%.loc4_15.2 [template = constants.%tuple.type.471]
// CHECK:STDOUT:     %self: %tuple.type.471 = bind_name self, %self.param
// CHECK:STDOUT:     %subscript.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc5: type = splice_block %i32.loc5_32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc5_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc5_32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %subscript: %i32 = bind_name subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %tuple.type.471](%subscript.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %tuple.type.471 = name_ref self, %self
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c6]
// CHECK:STDOUT:   %tuple.elem0: %i32 = tuple_access %self.ref, element0
// CHECK:STDOUT:   return %tuple.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc10_26.1: %tuple.type.f94 = tuple_literal (%int_1, %int_5)
// CHECK:STDOUT:   %impl.elem0.loc10_26.1: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc10_26.1: <bound method> = bound_method %int_1, %impl.elem0.loc10_26.1 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_26.1: <specific function> = specific_function %Convert.bound.loc10_26.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc10_26.1: init %i32 = call %Convert.specific_fn.loc10_26.1(%int_1) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc10_26.2: %i32 = value_of_initializer %int.convert_checked.loc10_26.1 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc10_26.3: %i32 = converted %int_1, %.loc10_26.2 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %impl.elem0.loc10_26.2: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc10_26.2: <bound method> = bound_method %int_5, %impl.elem0.loc10_26.2 [template = constants.%Convert.bound.b33]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_26.2: <specific function> = specific_function %Convert.bound.loc10_26.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.89d]
// CHECK:STDOUT:   %int.convert_checked.loc10_26.2: init %i32 = call %Convert.specific_fn.loc10_26.2(%int_5) [template = constants.%int_5.70b]
// CHECK:STDOUT:   %.loc10_26.4: %i32 = value_of_initializer %int.convert_checked.loc10_26.2 [template = constants.%int_5.70b]
// CHECK:STDOUT:   %.loc10_26.5: %i32 = converted %int_5, %.loc10_26.4 [template = constants.%int_5.70b]
// CHECK:STDOUT:   %tuple: %tuple.type.471 = tuple_value (%.loc10_26.3, %.loc10_26.5) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc10_27: %tuple.type.471 = converted %.loc10_26.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %s: %tuple.type.471 = bind_name s, %.loc10_27
// CHECK:STDOUT:   %s.ref: %tuple.type.471 = name_ref s, %s
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc11_17.1: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_0, %impl.elem0.loc11_17.1 [template = constants.%Convert.bound.b41]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c35]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %Convert.specific_fn.loc11(%int_0) [template = constants.%int_0.f61]
// CHECK:STDOUT:   %.loc11_17.1: %i32 = value_of_initializer %int.convert_checked.loc11 [template = constants.%int_0.f61]
// CHECK:STDOUT:   %.loc11_17.2: %i32 = converted %int_0, %.loc11_17.1 [template = constants.%int_0.f61]
// CHECK:STDOUT:   %impl.elem0.loc11_17.2: %At.type.740 = impl_witness_access constants.%impl_witness.d5f, element0 [template = constants.%At.065]
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
// CHECK:STDOUT:   %ElementType.e6b: type = class_type @ElementType [template]
// CHECK:STDOUT:   %SubscriptType.8ee: type = class_type @SubscriptType [template]
// CHECK:STDOUT:   %IndexWith.type.504: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.504 = struct_value () [template]
// CHECK:STDOUT:   %IndexWith.type.2d3: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.8ee, %ElementType.e6b)> [template]
// CHECK:STDOUT:   %At.type.38e: type = fn_type @At.1, @IndexWith(%SubscriptType.8ee, %ElementType.e6b) [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%At.decl) [template]
// CHECK:STDOUT:   %At.type.afe: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.9bf: %At.type.afe = struct_value () [template]
// CHECK:STDOUT:   %ElementType.val: %ElementType.e6b = struct_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IndexWith = %import_ref.76d
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.76d: %IndexWith.type.504 = import_ref Core//prelude/operators/index, IndexWith, loaded [template = constants.%IndexWith.generic]
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
// CHECK:STDOUT:   %ElementType.decl: type = class_decl @ElementType [template = constants.%ElementType.e6b] {} {}
// CHECK:STDOUT:   %SubscriptType.decl: type = class_decl @SubscriptType [template = constants.%SubscriptType.8ee] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.504 = name_ref IndexWith, imports.%import_ref.76d [template = constants.%IndexWith.generic]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [template = constants.%SubscriptType.8ee]
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [template = constants.%ElementType.e6b]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%SubscriptType.8ee, constants.%ElementType.e6b)> [template = constants.%IndexWith.type.2d3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%At.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %IndexWith.type {
// CHECK:STDOUT:   %At.decl: %At.type.afe = fn_decl @At.2 [template = constants.%At.9bf] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %subscript.patt: %SubscriptType.8ee = binding_pattern subscript
// CHECK:STDOUT:     %subscript.param_patt: %SubscriptType.8ee = value_param_pattern %subscript.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %ElementType.e6b = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ElementType.e6b = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [template = constants.%ElementType.e6b]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%C.ref [template = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %subscript.param: %SubscriptType.8ee = value_param runtime_param1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [template = constants.%SubscriptType.8ee]
// CHECK:STDOUT:     %subscript: %SubscriptType.8ee = bind_name subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %ElementType.e6b = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %ElementType.e6b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = file.%impl_witness
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
// CHECK:STDOUT:   .Self = constants.%ElementType.e6b
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SubscriptType.8ee
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %C](%subscript.param_patt: %SubscriptType.8ee) -> %return.param_patt: %ElementType.e6b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_13.2: init %ElementType.e6b = class_init (), %return [template = constants.%ElementType.val]
// CHECK:STDOUT:   %.loc10_14: init %ElementType.e6b = converted %.loc10_13.1, %.loc10_13.2 [template = constants.%ElementType.val]
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
// CHECK:STDOUT:   %.loc22_25.1: %SubscriptType.8ee = converted %int_0, <error> [template = <error>]
// CHECK:STDOUT:   %impl.elem0: %At.type.38e = impl_witness_access constants.%impl_witness, element0 [template = constants.%At.9bf]
// CHECK:STDOUT:   %At.bound: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %.loc22_25.2: ref %ElementType.e6b = temporary_storage
// CHECK:STDOUT:   %At.call: init %ElementType.e6b = call %At.bound(%c.ref, <error>) to %.loc22_25.2
// CHECK:STDOUT:   %.loc22_25.3: ref %ElementType.e6b = temporary %.loc22_25.2, %At.call
// CHECK:STDOUT:   %.loc22_25.4: %ElementType.e6b = bind_value %.loc22_25.3
// CHECK:STDOUT:   %x: %ElementType.e6b = bind_name x, %.loc22_25.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_index_with_not_implemented.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .IndexWith = %import_ref.76d
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
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
