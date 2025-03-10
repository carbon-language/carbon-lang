// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/stringify.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/stringify.carbon

// --- fail_empty_params.carbon

library "[[@TEST_NAME]]";

class NoParams {}
class EmptyParams() {}

var v: NoParams;
// CHECK:STDERR: fail_empty_params.carbon:[[@LINE+7]]:1: error: cannot implicitly convert value of type `NoParams` to `EmptyParams()` [ImplicitAsConversionFailure]
// CHECK:STDERR: var w: EmptyParams() = v;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_empty_params.carbon:[[@LINE+4]]:1: note: type `NoParams` does not implement interface `Core.ImplicitAs(EmptyParams())` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var w: EmptyParams() = v;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var w: EmptyParams() = v;

// --- fail_nested.carbon

library "[[@TEST_NAME]]";

class Outer(T:! type) {
  class Inner(U:! type) {
  }
}

var v: Outer({}*);

// TODO: It would be nice to include the `Outer({}*).` prefix in the name of `Inner`.
// CHECK:STDERR: fail_nested.carbon:[[@LINE+7]]:1: error: cannot implicitly convert value of type `Outer({}*)` to `Inner({.a: i32}*)` [ImplicitAsConversionFailure]
// CHECK:STDERR: var w: Outer({}*).Inner({.a: i32}*) = v;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:1: note: type `Outer({}*)` does not implement interface `Core.ImplicitAs(Inner({.a: i32}*))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var w: Outer({}*).Inner({.a: i32}*) = v;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var w: Outer({}*).Inner({.a: i32}*) = v;

// --- fail_int_value.carbon

library "[[@TEST_NAME]]";

class C(N:! i32) {}

// CHECK:STDERR: fail_int_value.carbon:[[@LINE+7]]:1: error: cannot implicitly convert value of type `()` to `C(123)` [ImplicitAsConversionFailure]
// CHECK:STDERR: var v: C(123) = ();
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_int_value.carbon:[[@LINE+4]]:1: note: type `()` does not implement interface `Core.ImplicitAs(C(123))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var v: C(123) = ();
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
var v: C(123) = ();

// --- fail_class_param.carbon

library "[[@TEST_NAME]]";

class D {
  var a: i32;
  var b: i32;
}

class E(F:! D) {}

// CHECK:STDERR: fail_class_param.carbon:[[@LINE+14]]:8: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
// CHECK:STDERR: var g: E({.a = 1, .b = 2}) = {} as E({.a = 3, .b = 4} as D);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_class_param.carbon:[[@LINE-5]]:9: note: initializing generic parameter `F` declared here [InitializingGenericParam]
// CHECK:STDERR: class E(F:! D) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
// CHECK:STDERR: fail_class_param.carbon:[[@LINE+7]]:36: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
// CHECK:STDERR: var g: E({.a = 1, .b = 2}) = {} as E({.a = 3, .b = 4} as D);
// CHECK:STDERR:                                    ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_class_param.carbon:[[@LINE-12]]:9: note: initializing generic parameter `F` declared here [InitializingGenericParam]
// CHECK:STDERR: class E(F:! D) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
var g: E({.a = 1, .b = 2}) = {} as E({.a = 3, .b = 4} as D);

// CHECK:STDOUT: --- fail_empty_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NoParams: type = class_type @NoParams [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyParams.type: type = generic_class_type @EmptyParams [concrete]
// CHECK:STDOUT:   %EmptyParams.generic: %EmptyParams.type = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyParams: type = class_type @EmptyParams [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NoParams = %NoParams.decl
// CHECK:STDOUT:     .EmptyParams = %EmptyParams.decl
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:     .w = %w
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NoParams.decl: type = class_decl @NoParams [concrete = constants.%NoParams] {} {}
// CHECK:STDOUT:   %EmptyParams.decl: %EmptyParams.type = class_decl @EmptyParams [concrete = constants.%EmptyParams.generic] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %NoParams = binding_pattern v
// CHECK:STDOUT:     %.loc7: %NoParams = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %NoParams = var v
// CHECK:STDOUT:   %NoParams.ref: type = name_ref NoParams, %NoParams.decl [concrete = constants.%NoParams]
// CHECK:STDOUT:   %v: ref %NoParams = bind_name v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %EmptyParams = binding_pattern w
// CHECK:STDOUT:     %.loc15_1: %EmptyParams = var_pattern %w.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %EmptyParams = var w
// CHECK:STDOUT:   %.loc15_20: type = splice_block %EmptyParams [concrete = constants.%EmptyParams] {
// CHECK:STDOUT:     %EmptyParams.ref: %EmptyParams.type = name_ref EmptyParams, %EmptyParams.decl [concrete = constants.%EmptyParams.generic]
// CHECK:STDOUT:     %EmptyParams: type = class_type @EmptyParams [concrete = constants.%EmptyParams]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: ref %EmptyParams = bind_name w, %w.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoParams {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoParams
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @EmptyParams {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%EmptyParams
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: ref %NoParams = name_ref v, file.%v
// CHECK:STDOUT:   %.loc15: %EmptyParams = converted %v.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%w.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.9d6: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.eae: type = generic_class_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.generic.137: %Inner.type.eae = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.c71: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %Outer.614: type = class_type @Outer, @Outer(%ptr.c28) [concrete]
// CHECK:STDOUT:   %Inner.type.5d2: type = generic_class_type @Inner, @Outer(%ptr.c28) [concrete]
// CHECK:STDOUT:   %Inner.generic.8e6: %Inner.type.5d2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %ptr.1bb: type = ptr_type %struct_type.a [concrete]
// CHECK:STDOUT:   %Inner.277: type = class_type @Inner, @Inner(%ptr.c28, %ptr.1bb) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:     .w = %w
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: %Outer.type = class_decl @Outer [concrete = constants.%Outer.generic] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %Outer.614 = binding_pattern v
// CHECK:STDOUT:     %.loc9_1: %Outer.614 = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Outer.614 = var v
// CHECK:STDOUT:   %.loc9_17: type = splice_block %Outer.loc9 [concrete = constants.%Outer.614] {
// CHECK:STDOUT:     %Outer.ref.loc9: %Outer.type = name_ref Outer, %Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %.loc9_15: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_16: type = converted %.loc9_15, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc9: type = ptr_type %empty_struct_type [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %Outer.loc9: type = class_type @Outer, @Outer(constants.%ptr.c28) [concrete = constants.%Outer.614]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Outer.614 = bind_name v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %Inner.277 = binding_pattern w
// CHECK:STDOUT:     %.loc19_1: %Inner.277 = var_pattern %w.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %Inner.277 = var w
// CHECK:STDOUT:   %.loc19_35: type = splice_block %Inner [concrete = constants.%Inner.277] {
// CHECK:STDOUT:     %Outer.ref.loc19: %Outer.type = name_ref Outer, %Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %.loc19_15: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc19_16: type = converted %.loc19_15, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc19_16: type = ptr_type %empty_struct_type [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %Outer.loc19: type = class_type @Outer, @Outer(constants.%ptr.c28) [concrete = constants.%Outer.614]
// CHECK:STDOUT:     %.loc19_18: %Inner.type.5d2 = specific_constant @Outer.%Inner.decl, @Outer(constants.%ptr.c28) [concrete = constants.%Inner.generic.8e6]
// CHECK:STDOUT:     %Inner.ref: %Inner.type.5d2 = name_ref Inner, %.loc19_18 [concrete = constants.%Inner.generic.8e6]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %ptr.loc19_34: type = ptr_type %struct_type.a [concrete = constants.%ptr.1bb]
// CHECK:STDOUT:     %Inner: type = class_type @Inner, @Inner(constants.%ptr.c28, constants.%ptr.1bb) [concrete = constants.%Inner.277]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: ref %Inner.277 = bind_name w, %w.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc4_13.2) [symbolic = %Inner.type (constants.%Inner.type.eae)]
// CHECK:STDOUT:   %Inner.generic: @Outer.%Inner.type (%Inner.type.eae) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.137)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.eae) = class_decl @Inner [symbolic = @Outer.%Inner.generic (constants.%Inner.generic.137)] {
// CHECK:STDOUT:       %U.patt.loc5_15.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc5_15.2 (constants.%U.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.loc5_15.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc5_15.2 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.9d6
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_13.1: type, %U.loc5_15.1: type) {
// CHECK:STDOUT:   %U.loc5_15.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc5_15.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc5_15.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc5_15.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.c71
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: ref %Outer.614 = name_ref v, file.%v
// CHECK:STDOUT:   %.loc19: %Inner.277 = converted %v.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%w.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc5_15.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc5_15.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(%T.loc4_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%ptr.c28) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%ptr.c28
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%ptr.c28
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5d2
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.8e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%ptr.c28, constants.%ptr.1bb) {
// CHECK:STDOUT:   %U.loc5_15.2 => constants.%ptr.1bb
// CHECK:STDOUT:   %U.patt.loc5_15.2 => constants.%ptr.1bb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_int_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.8e2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.506: type = class_type @C, @C(%N.51e) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_123.fff, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_123.fff, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.f7f: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %C.4c3: type = class_type @C, @C(%int_123.f7f) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %N.patt.loc4_9.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_9.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_9.1: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_9.2 (constants.%N.51e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %C.4c3 = binding_pattern v
// CHECK:STDOUT:     %.loc13_1: %C.4c3 = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %C.4c3 = var v
// CHECK:STDOUT:   %.loc13_13.1: type = splice_block %C [concrete = constants.%C.4c3] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:     %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc13_13.1: <bound method> = bound_method %int_123, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_13.2: <bound method> = bound_method %int_123, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc13_13.2(%int_123) [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:     %.loc13_13.2: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:     %.loc13_13.3: %i32 = converted %int_123, %.loc13_13.2 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_123.f7f) [concrete = constants.%C.4c3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %C.4c3 = bind_name v, %v.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc4_9.1: %i32) {
// CHECK:STDOUT:   %N.loc4_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_9.2 (constants.%N.51e)]
// CHECK:STDOUT:   %N.patt.loc4_9.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_9.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.506
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_18: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_1: %C.4c3 = converted %.loc13_18, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%v.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc4_9.2 => constants.%N.51e
// CHECK:STDOUT:   %N.patt.loc4_9.2 => constants.%N.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_123.f7f) {
// CHECK:STDOUT:   %N.loc4_9.2 => constants.%int_123.f7f
// CHECK:STDOUT:   %N.patt.loc4_9.2 => constants.%int_123.f7f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_class_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.501: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.705: <witness> = complete_type_witness %struct_type.a.b.501 [concrete]
// CHECK:STDOUT:   %F: %D = bind_symbolic_name F, 0 [symbolic]
// CHECK:STDOUT:   %F.patt: %D = symbolic_binding_pattern F, 0 [symbolic]
// CHECK:STDOUT:   %E.type: type = generic_class_type @E [concrete]
// CHECK:STDOUT:   %E.generic: %E.type = struct_value () [concrete]
// CHECK:STDOUT:   %E: type = class_type @E, @E(%F) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
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
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %D.val.413: %D = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %D.val.835: %D = struct_value (%int_3.822, %int_4.940) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .g = %g
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %E.decl: %E.type = class_decl @E [concrete = constants.%E.generic] {
// CHECK:STDOUT:     %F.patt.loc9_9.1: %D = symbolic_binding_pattern F, 0 [symbolic = %F.patt.loc9_9.2 (constants.%F.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %F.loc9_9.1: %D = bind_symbolic_name F, 0 [symbolic = %F.loc9_9.2 (constants.%F)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %g.patt: <error> = binding_pattern g
// CHECK:STDOUT:     %.loc25_1: <error> = var_pattern %g.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %g.var: ref <error> = var g
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %E.ref: %E.type = name_ref E, %E.decl [concrete = constants.%E.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc25_25.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:     %impl.elem0.loc25_25.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc25_25.1: <bound method> = bound_method %int_1, %impl.elem0.loc25_25.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:     %specific_fn.loc25_25.1: <specific function> = specific_function %impl.elem0.loc25_25.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc25_25.2: <bound method> = bound_method %int_1, %specific_fn.loc25_25.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:     %int.convert_checked.loc25_25.1: init %i32 = call %bound_method.loc25_25.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc25_25.2: init %i32 = converted %int_1, %int.convert_checked.loc25_25.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc25_25.3: ref %D = temporary_storage
// CHECK:STDOUT:     %.loc25_25.4: ref %i32 = class_element_access %.loc25_25.3, element0
// CHECK:STDOUT:     %.loc25_25.5: init %i32 = initialize_from %.loc25_25.2 to %.loc25_25.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %impl.elem0.loc25_25.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc25_25.3: <bound method> = bound_method %int_2, %impl.elem0.loc25_25.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:     %specific_fn.loc25_25.2: <specific function> = specific_function %impl.elem0.loc25_25.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc25_25.4: <bound method> = bound_method %int_2, %specific_fn.loc25_25.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:     %int.convert_checked.loc25_25.2: init %i32 = call %bound_method.loc25_25.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc25_25.6: init %i32 = converted %int_2, %int.convert_checked.loc25_25.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc25_25.7: ref %i32 = class_element_access %.loc25_25.3, element1
// CHECK:STDOUT:     %.loc25_25.8: init %i32 = initialize_from %.loc25_25.6 to %.loc25_25.7 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc25_25.9: init %D = class_init (%.loc25_25.5, %.loc25_25.8), %.loc25_25.3 [concrete = constants.%D.val.413]
// CHECK:STDOUT:     %.loc25_25.10: ref %D = temporary %.loc25_25.3, %.loc25_25.9
// CHECK:STDOUT:     %.loc25_26.1: ref %D = converted %.loc25_25.1, %.loc25_25.10
// CHECK:STDOUT:     %.loc25_26.2: %D = bind_value %.loc25_26.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %g: <error> = bind_name g, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc5_8: %D.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %D.elem = var_pattern %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc5: ref %D.elem = var <none>
// CHECK:STDOUT:   %.loc6_8: %D.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc6_3: %D.elem = var_pattern %.loc6_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc6: ref %D.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b.501 [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .a = %.loc5_8
// CHECK:STDOUT:   .b = %.loc6_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @E(%F.loc9_9.1: %D) {
// CHECK:STDOUT:   %F.loc9_9.2: %D = bind_symbolic_name F, 0 [symbolic = %F.loc9_9.2 (constants.%F)]
// CHECK:STDOUT:   %F.patt.loc9_9.2: %D = symbolic_binding_pattern F, 0 [symbolic = %F.patt.loc9_9.2 (constants.%F.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%E
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc25_31: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, file.%E.decl [concrete = constants.%E.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %.loc25_53.1: %struct_type.a.b.cfd = struct_literal (%int_3, %int_4)
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %impl.elem0.loc25_53.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc25_53.1: <bound method> = bound_method %int_3, %impl.elem0.loc25_53.1 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc25_53.1: <specific function> = specific_function %impl.elem0.loc25_53.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_53.2: <bound method> = bound_method %int_3, %specific_fn.loc25_53.1 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc25_53.1: init %i32 = call %bound_method.loc25_53.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc25_53.2: init %i32 = converted %int_3, %int.convert_checked.loc25_53.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc25_53.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc25_53.4: ref %i32 = class_element_access %.loc25_53.3, element0
// CHECK:STDOUT:   %.loc25_53.5: init %i32 = initialize_from %.loc25_53.2 to %.loc25_53.4 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %impl.elem0.loc25_53.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc25_53.3: <bound method> = bound_method %int_4, %impl.elem0.loc25_53.2 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc25_53.2: <specific function> = specific_function %impl.elem0.loc25_53.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_53.4: <bound method> = bound_method %int_4, %specific_fn.loc25_53.2 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc25_53.2: init %i32 = call %bound_method.loc25_53.4(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc25_53.6: init %i32 = converted %int_4, %int.convert_checked.loc25_53.2 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc25_53.7: ref %i32 = class_element_access %.loc25_53.3, element1
// CHECK:STDOUT:   %.loc25_53.8: init %i32 = initialize_from %.loc25_53.6 to %.loc25_53.7 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc25_53.9: init %D = class_init (%.loc25_53.5, %.loc25_53.8), %.loc25_53.3 [concrete = constants.%D.val.835]
// CHECK:STDOUT:   %.loc25_53.10: ref %D = temporary %.loc25_53.3, %.loc25_53.9
// CHECK:STDOUT:   %.loc25_55.1: ref %D = converted %.loc25_53.1, %.loc25_53.10
// CHECK:STDOUT:   %.loc25_55.2: %D = bind_value %.loc25_55.1
// CHECK:STDOUT:   assign file.%g.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E(constants.%F) {
// CHECK:STDOUT:   %F.loc9_9.2 => constants.%F
// CHECK:STDOUT:   %F.patt.loc9_9.2 => constants.%F
// CHECK:STDOUT: }
// CHECK:STDOUT:
