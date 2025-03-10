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
// CHECK:STDERR: fail_invalid_subscript_type.carbon:[[@LINE+7]]:22: error: cannot implicitly convert value of type `Core.IntLiteral` to `SubscriptType` [ImplicitAsConversionFailure]
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
// CHECK:STDERR: fail_index_with_not_implemented.carbon:[[@LINE+4]]:14: error: type `C` does not support indexing [TypeNotIndexable]
// CHECK:STDERR: let x: i32 = c[0];
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
let x: i32 = c[0];


// CHECK:STDOUT: --- overloaded_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ElementType.e6b: type = class_type @ElementType [concrete]
// CHECK:STDOUT:   %SubscriptType.8ee: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %IndexWith.type.504: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.504 = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.type.e80: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.8ee, %ElementType.e6b)> [concrete]
// CHECK:STDOUT:   %At.type.b3f: type = fn_type @At.1, @IndexWith(%SubscriptType.8ee, %ElementType.e6b) [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%At.decl) [concrete]
// CHECK:STDOUT:   %At.type.178: type = fn_type @At.2 [concrete]
// CHECK:STDOUT:   %At.d43: %At.type.178 = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet: %IndexWith.type.e80 = facet_value %C, (%impl_witness) [concrete]
// CHECK:STDOUT:   %ElementType.val: %ElementType.e6b = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType.val: %SubscriptType.8ee = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.9e4: type = fn_type_with_self_type %At.type.b3f, %IndexWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.504 = import_ref Core//prelude/operators/index, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .ElementType = %ElementType.decl
// CHECK:STDOUT:     .SubscriptType = %SubscriptType.decl
// CHECK:STDOUT:     .s = %s
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %ElementType.decl: type = class_decl @ElementType [concrete = constants.%ElementType.e6b] {} {}
// CHECK:STDOUT:   %SubscriptType.decl: type = class_decl @SubscriptType [concrete = constants.%SubscriptType.8ee] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.504 = name_ref IndexWith, imports.%Core.IndexWith [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.8ee]
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [concrete = constants.%ElementType.e6b]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%SubscriptType.8ee, constants.%ElementType.e6b)> [concrete = constants.%IndexWith.type.e80]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%At.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %SubscriptType.8ee = binding_pattern s
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SubscriptType.ref: type = name_ref SubscriptType, %SubscriptType.decl [concrete = constants.%SubscriptType.8ee]
// CHECK:STDOUT:   %.loc14_25.1: ref %SubscriptType.8ee = temporary_storage
// CHECK:STDOUT:   %.loc14_25.2: init %SubscriptType.8ee = class_init (), %.loc14_25.1 [concrete = constants.%SubscriptType.val]
// CHECK:STDOUT:   %.loc14_25.3: ref %SubscriptType.8ee = temporary %.loc14_25.1, %.loc14_25.2
// CHECK:STDOUT:   %.loc14_25.4: ref %SubscriptType.8ee = converted @__global_init.%.loc14, %.loc14_25.3
// CHECK:STDOUT:   %s: ref %SubscriptType.8ee = bind_name s, %.loc14_25.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc15_13.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc15_13.2: init %C = class_init (), %.loc15_13.1 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc15_13.3: ref %C = temporary %.loc15_13.1, %.loc15_13.2
// CHECK:STDOUT:   %.loc15_13.4: ref %C = converted @__global_init.%.loc15, %.loc15_13.3
// CHECK:STDOUT:   %c: ref %C = bind_name c, %.loc15_13.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %ElementType.e6b = binding_pattern x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ElementType.ref: type = name_ref ElementType, %ElementType.decl [concrete = constants.%ElementType.e6b]
// CHECK:STDOUT:   %.loc16: ref %ElementType.e6b = temporary @__global_init.%.loc16_25, @__global_init.%At.call
// CHECK:STDOUT:   %x: ref %ElementType.e6b = bind_name x, %.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %IndexWith.type {
// CHECK:STDOUT:   %At.decl: %At.type.178 = fn_decl @At.2 [concrete = constants.%At.d43] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %subscript.patt: %SubscriptType.8ee = binding_pattern subscript
// CHECK:STDOUT:     %subscript.param_patt: %SubscriptType.8ee = value_param_pattern %subscript.patt, call_param1
// CHECK:STDOUT:     %return.patt: %ElementType.e6b = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ElementType.e6b = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [concrete = constants.%ElementType.e6b]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %subscript.param: %SubscriptType.8ee = value_param call_param1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.8ee]
// CHECK:STDOUT:     %subscript: %SubscriptType.8ee = bind_name subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %ElementType.e6b = out_param call_param2
// CHECK:STDOUT:     %return: ref %ElementType.e6b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SubscriptType = <poisoned>
// CHECK:STDOUT:   .ElementType = <poisoned>
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ElementType {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ElementType.e6b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SubscriptType.8ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %C](%subscript.param_patt: %SubscriptType.8ee) -> %return.param_patt: %ElementType.e6b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_13.2: init %ElementType.e6b = class_init (), %return [concrete = constants.%ElementType.val]
// CHECK:STDOUT:   %.loc10_14: init %ElementType.e6b = converted %.loc10_13.1, %.loc10_13.2 [concrete = constants.%ElementType.val]
// CHECK:STDOUT:   return %.loc10_14 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc15: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, file.%c
// CHECK:STDOUT:   %s.ref: ref %SubscriptType.8ee = name_ref s, file.%s
// CHECK:STDOUT:   %.loc16_24: %SubscriptType.8ee = bind_value %s.ref
// CHECK:STDOUT:   %impl.elem0: %.9e4 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%At.d43]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %.loc16_25: ref %ElementType.e6b = temporary_storage
// CHECK:STDOUT:   %.loc16_22: %C = bind_value %c.ref
// CHECK:STDOUT:   %At.call: init %ElementType.e6b = call %bound_method(%.loc16_22, %.loc16_24) to %.loc16_25
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overloaded_builtin.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %IndexWith.type.504: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.504 = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.type.917: type = facet_type <@IndexWith, @IndexWith(%i32, %i32)> [concrete]
// CHECK:STDOUT:   %At.type.d77: type = fn_type @At.1, @IndexWith(%i32, %i32) [concrete]
// CHECK:STDOUT:   %impl_witness.123: <witness> = impl_witness (@impl.18b.%At.decl) [concrete]
// CHECK:STDOUT:   %At.type.9ac: type = fn_type @At.2 [concrete]
// CHECK:STDOUT:   %At.642: %At.type.9ac = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet: %IndexWith.type.917 = facet_value %tuple.type.d07, (%impl_witness.123) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.4e6: <bound method> = bound_method %int_5.64b, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.a25: <bound method> = bound_method %int_5.64b, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_5.0f6) [concrete]
// CHECK:STDOUT:   %Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %.1a5: type = fn_type_with_self_type %At.type.d77, %IndexWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.504 = import_ref Core//prelude/operators/index, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .s = %s
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   impl_decl @impl.18b [concrete] {} {
// CHECK:STDOUT:     %int_32.loc4_7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_15.1: %tuple.type.24b = tuple_literal (%i32.loc4_7, %i32.loc4_12)
// CHECK:STDOUT:     %.loc4_15.2: type = converted %.loc4_15.1, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.504 = name_ref IndexWith, imports.%Core.IndexWith [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:     %int_32.loc4_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%i32, constants.%i32)> [concrete = constants.%IndexWith.type.917]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.18b.%At.decl) [concrete = constants.%impl_witness.123]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %tuple.type.d07 = binding_pattern s
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_17.1: type = splice_block %.loc10_17.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc10_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc10_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10_17.2: %tuple.type.24b = tuple_literal (%i32.loc10_9, %i32.loc10_14)
// CHECK:STDOUT:     %.loc10_17.3: type = converted %.loc10_17.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc10_26.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc10_26.1: <bound method> = bound_method @__global_init.%int_1, %impl.elem0.loc10_26.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc10_26.1: <specific function> = specific_function %impl.elem0.loc10_26.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_26.2: <bound method> = bound_method @__global_init.%int_1, %specific_fn.loc10_26.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc10_26.1: init %i32 = call %bound_method.loc10_26.2(@__global_init.%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_26.1: %i32 = value_of_initializer %int.convert_checked.loc10_26.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_26.2: %i32 = converted @__global_init.%int_1, %.loc10_26.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc10_26.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc10_26.3: <bound method> = bound_method @__global_init.%int_5, %impl.elem0.loc10_26.2 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn.loc10_26.2: <specific function> = specific_function %impl.elem0.loc10_26.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_26.4: <bound method> = bound_method @__global_init.%int_5, %specific_fn.loc10_26.2 [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:   %int.convert_checked.loc10_26.2: init %i32 = call %bound_method.loc10_26.4(@__global_init.%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc10_26.3: %i32 = value_of_initializer %int.convert_checked.loc10_26.2 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc10_26.4: %i32 = converted @__global_init.%int_5, %.loc10_26.3 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %tuple: %tuple.type.d07 = tuple_value (%.loc10_26.2, %.loc10_26.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc10_26.5: %tuple.type.d07 = converted @__global_init.%.loc10, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %s: %tuple.type.d07 = bind_name s, %.loc10_26.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %i32 = binding_pattern e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_8: type = splice_block %i32.loc11 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_17.1: ref %i32 = temporary_storage
// CHECK:STDOUT:   %.loc11_17.2: ref %i32 = temporary %.loc11_17.1, @__global_init.%At.call
// CHECK:STDOUT:   %e: ref %i32 = bind_name e, %.loc11_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.18b: %.loc4_15.2 as %IndexWith.type {
// CHECK:STDOUT:   %At.decl: %At.type.9ac = fn_decl @At.2 [concrete = constants.%At.642] {
// CHECK:STDOUT:     %self.patt: %tuple.type.d07 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %tuple.type.d07 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %subscript.patt: %i32 = binding_pattern subscript
// CHECK:STDOUT:     %subscript.param_patt: %i32 = value_param_pattern %subscript.patt, call_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %tuple.type.d07 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.18b.%.loc4_15.2 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     %self: %tuple.type.d07 = bind_name self, %self.param
// CHECK:STDOUT:     %subscript.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc5: type = splice_block %i32.loc5_32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc5_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc5_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %subscript: %i32 = bind_name subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %tuple.type.d07](%subscript.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %tuple.type.d07 = name_ref self, %self
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %tuple.elem0: %i32 = tuple_access %self.ref, element0
// CHECK:STDOUT:   return %tuple.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc10: %tuple.type.f94 = tuple_literal (%int_1, %int_5)
// CHECK:STDOUT:   %s.ref: %tuple.type.d07 = name_ref s, file.%s
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc11_17.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_17.1: <bound method> = bound_method %int_0, %impl.elem0.loc11_17.1 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc11_17.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_17.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc11_17.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc11_17.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc11_17.2: %i32 = converted %int_0, %.loc11_17.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %impl.elem0.loc11_17.2: %.1a5 = impl_witness_access constants.%impl_witness.123, element0 [concrete = constants.%At.642]
// CHECK:STDOUT:   %bound_method.loc11_17.3: <bound method> = bound_method %s.ref, %impl.elem0.loc11_17.2
// CHECK:STDOUT:   %At.call: init %i32 = call %bound_method.loc11_17.3(%s.ref, %.loc11_17.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_subscript_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ElementType.e6b: type = class_type @ElementType [concrete]
// CHECK:STDOUT:   %SubscriptType.8ee: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %IndexWith.type.504: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.504 = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.type.e80: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.8ee, %ElementType.e6b)> [concrete]
// CHECK:STDOUT:   %At.type.b3f: type = fn_type @At.1, @IndexWith(%SubscriptType.8ee, %ElementType.e6b) [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%At.decl) [concrete]
// CHECK:STDOUT:   %At.type.178: type = fn_type @At.2 [concrete]
// CHECK:STDOUT:   %At.d43: %At.type.178 = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet: %IndexWith.type.e80 = facet_value %C, (%impl_witness) [concrete]
// CHECK:STDOUT:   %ElementType.val: %ElementType.e6b = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %.9e4: type = fn_type_with_self_type %At.type.b3f, %IndexWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.504 = import_ref Core//prelude/operators/index, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .ElementType = %ElementType.decl
// CHECK:STDOUT:     .SubscriptType = %SubscriptType.decl
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %ElementType.decl: type = class_decl @ElementType [concrete = constants.%ElementType.e6b] {} {}
// CHECK:STDOUT:   %SubscriptType.decl: type = class_decl @SubscriptType [concrete = constants.%SubscriptType.8ee] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.504 = name_ref IndexWith, imports.%Core.IndexWith [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.8ee]
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [concrete = constants.%ElementType.e6b]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%SubscriptType.8ee, constants.%ElementType.e6b)> [concrete = constants.%IndexWith.type.e80]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%At.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc14_13.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc14_13.2: init %C = class_init (), %.loc14_13.1 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc14_13.3: ref %C = temporary %.loc14_13.1, %.loc14_13.2
// CHECK:STDOUT:   %.loc14_13.4: ref %C = converted @__global_init.%.loc14, %.loc14_13.3
// CHECK:STDOUT:   %c: ref %C = bind_name c, %.loc14_13.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %ElementType.e6b = binding_pattern x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ElementType.ref: type = name_ref ElementType, %ElementType.decl [concrete = constants.%ElementType.e6b]
// CHECK:STDOUT:   %.loc22: ref %ElementType.e6b = temporary @__global_init.%.loc22_25.2, @__global_init.%At.call
// CHECK:STDOUT:   %x: ref %ElementType.e6b = bind_name x, %.loc22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %IndexWith.type {
// CHECK:STDOUT:   %At.decl: %At.type.178 = fn_decl @At.2 [concrete = constants.%At.d43] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %subscript.patt: %SubscriptType.8ee = binding_pattern subscript
// CHECK:STDOUT:     %subscript.param_patt: %SubscriptType.8ee = value_param_pattern %subscript.patt, call_param1
// CHECK:STDOUT:     %return.patt: %ElementType.e6b = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ElementType.e6b = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [concrete = constants.%ElementType.e6b]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %subscript.param: %SubscriptType.8ee = value_param call_param1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.8ee]
// CHECK:STDOUT:     %subscript: %SubscriptType.8ee = bind_name subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %ElementType.e6b = out_param call_param2
// CHECK:STDOUT:     %return: ref %ElementType.e6b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SubscriptType = <poisoned>
// CHECK:STDOUT:   .ElementType = <poisoned>
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ElementType {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ElementType.e6b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SubscriptType.8ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %C](%subscript.param_patt: %SubscriptType.8ee) -> %return.param_patt: %ElementType.e6b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_13.2: init %ElementType.e6b = class_init (), %return [concrete = constants.%ElementType.val]
// CHECK:STDOUT:   %.loc10_14: init %ElementType.e6b = converted %.loc10_13.1, %.loc10_13.2 [concrete = constants.%ElementType.val]
// CHECK:STDOUT:   return %.loc10_14 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, file.%c
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc22_25.1: %SubscriptType.8ee = converted %int_0, <error> [concrete = <error>]
// CHECK:STDOUT:   %impl.elem0: %.9e4 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%At.d43]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %.loc22_25.2: ref %ElementType.e6b = temporary_storage
// CHECK:STDOUT:   %.loc22_22: %C = bind_value %c.ref
// CHECK:STDOUT:   %At.call: init %ElementType.e6b = call %bound_method(%.loc22_22, <error>) to %.loc22_25.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_index_with_not_implemented.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc6_13.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc6_13.2: init %C = class_init (), %.loc6_13.1 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_13.3: ref %C = temporary %.loc6_13.1, %.loc6_13.2
// CHECK:STDOUT:   %.loc6_13.4: ref %C = converted @__global_init.%.loc6, %.loc6_13.3
// CHECK:STDOUT:   %c: ref %C = bind_name c, %.loc6_13.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %i32 = binding_pattern x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %i32 = bind_name x, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, file.%c
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
