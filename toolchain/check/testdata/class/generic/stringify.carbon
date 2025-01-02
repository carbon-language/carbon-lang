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
// CHECK:STDERR: fail_empty_params.carbon:[[@LINE+7]]:1: error: cannot implicitly convert from `NoParams` to `EmptyParams()` [ImplicitAsConversionFailure]
// CHECK:STDERR: var w: EmptyParams() = v;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_empty_params.carbon:[[@LINE+4]]:1: note: type `NoParams` does not implement interface `Core.ImplicitAs(EmptyParams())` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var w: EmptyParams() = v;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDERR: fail_nested.carbon:[[@LINE+7]]:1: error: cannot implicitly convert from `Outer({}*)` to `Inner({.a: i32}*)` [ImplicitAsConversionFailure]
// CHECK:STDERR: var w: Outer({}*).Inner({.a: i32}*) = v;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:1: note: type `Outer({}*)` does not implement interface `Core.ImplicitAs(Inner({.a: i32}*))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var w: Outer({}*).Inner({.a: i32}*) = v;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var w: Outer({}*).Inner({.a: i32}*) = v;

// --- fail_int_value.carbon

library "[[@TEST_NAME]]";

class C(N:! i32) {}

// CHECK:STDERR: fail_int_value.carbon:[[@LINE+7]]:1: error: cannot implicitly convert from `()` to `C(123)` [ImplicitAsConversionFailure]
// CHECK:STDERR: var v: C(123) = ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_int_value.carbon:[[@LINE+4]]:1: note: type `()` does not implement interface `Core.ImplicitAs(C(123))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var v: C(123) = ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var v: C(123) = ();

// --- fail_class_param.carbon

library "[[@TEST_NAME]]";

class D {
  var a: i32;
  var b: i32;
}

class E(F:! D) {}

// CHECK:STDERR: fail_class_param.carbon:[[@LINE+13]]:8: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
// CHECK:STDERR: var g: E({.a = 1, .b = 2}) = {} as E({.a = 3, .b = 4} as D);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_class_param.carbon:[[@LINE-5]]:9: note: initializing generic parameter `F` declared here [InitializingGenericParam]
// CHECK:STDERR: class E(F:! D) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
// CHECK:STDERR: fail_class_param.carbon:[[@LINE+6]]:36: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
// CHECK:STDERR: var g: E({.a = 1, .b = 2}) = {} as E({.a = 3, .b = 4} as D);
// CHECK:STDERR:                                    ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_class_param.carbon:[[@LINE-12]]:9: note: initializing generic parameter `F` declared here [InitializingGenericParam]
// CHECK:STDERR: class E(F:! D) {}
// CHECK:STDERR:         ^
var g: E({.a = 1, .b = 2}) = {} as E({.a = 3, .b = 4} as D);

// CHECK:STDOUT: --- fail_empty_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NoParams: type = class_type @NoParams [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %EmptyParams.type: type = generic_class_type @EmptyParams [template]
// CHECK:STDOUT:   %EmptyParams.generic: %EmptyParams.type = struct_value () [template]
// CHECK:STDOUT:   %EmptyParams: type = class_type @EmptyParams [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NoParams = %NoParams.decl
// CHECK:STDOUT:     .EmptyParams = %EmptyParams.decl
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:     .w = %w
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NoParams.decl: type = class_decl @NoParams [template = constants.%NoParams] {} {}
// CHECK:STDOUT:   %EmptyParams.decl: %EmptyParams.type = class_decl @EmptyParams [template = constants.%EmptyParams.generic] {} {}
// CHECK:STDOUT:   %v.var: ref %NoParams = var v
// CHECK:STDOUT:   %v: ref %NoParams = bind_name v, %v.var
// CHECK:STDOUT:   %w.var: ref %EmptyParams = var w
// CHECK:STDOUT:   %w: ref %EmptyParams = bind_name w, %w.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoParams {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoParams
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @EmptyParams {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%EmptyParams
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: ref %NoParams = name_ref v, file.%v
// CHECK:STDOUT:   %.loc15: %EmptyParams = converted %v.ref, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%w.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [template]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [template]
// CHECK:STDOUT:   %Outer.1: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.1: type = generic_class_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.generic.1: %Inner.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.1: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %empty_struct_type [template]
// CHECK:STDOUT:   %Outer.2: type = class_type @Outer, @Outer(%ptr.1) [template]
// CHECK:STDOUT:   %Inner.type.2: type = generic_class_type @Inner, @Outer(%ptr.1) [template]
// CHECK:STDOUT:   %Inner.generic.2: %Inner.type.2 = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %struct_type.a [template]
// CHECK:STDOUT:   %Inner.2: type = class_type @Inner, @Inner(%ptr.1, %ptr.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:     .w = %w
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: %Outer.type = class_decl @Outer [template = constants.%Outer.generic] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Outer.2 = var v
// CHECK:STDOUT:   %v: ref %Outer.2 = bind_name v, %v.var
// CHECK:STDOUT:   %w.var: ref %Inner.2 = var w
// CHECK:STDOUT:   %w: ref %Inner.2 = bind_name w, %w.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc4_13.2) [symbolic = %Inner.type (constants.%Inner.type.1)]
// CHECK:STDOUT:   %Inner.generic: @Outer.%Inner.type (%Inner.type.1) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.1) = class_decl @Inner [symbolic = @Outer.%Inner.generic (constants.%Inner.generic.1)] {
// CHECK:STDOUT:       %U.patt.loc5_15.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc5_15.2 (constants.%U.patt)]
// CHECK:STDOUT:       %U.param_patt: type = value_param_pattern %U.patt.loc5_15.1, runtime_param<invalid> [symbolic = %U.patt.loc5_15.2 (constants.%U.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %U.loc5_15.1: type = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc5_15.2 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.1
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     complete_type_witness = %complete_type
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
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: ref %Outer.2 = name_ref v, file.%v
// CHECK:STDOUT:   %.loc19: %Inner.2 = converted %v.ref, <error> [template = <error>]
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
// CHECK:STDOUT: specific @Outer(constants.%ptr.1) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%ptr.1
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%ptr.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.2
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%ptr.1, constants.%ptr.2) {
// CHECK:STDOUT:   %U.loc5_15.2 => constants.%ptr.2
// CHECK:STDOUT:   %U.patt.loc5_15.2 => constants.%ptr.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_int_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %N.2: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.1: type = class_type @C, @C(%N.2) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %int_123.2: %i32 = int_value 123 [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%int_123.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %N.patt.loc4_9.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc4_9.1, runtime_param<invalid> [symbolic = %N.patt.loc4_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_9.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc4_9.2 (constants.%N.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %C.2 = var v
// CHECK:STDOUT:   %v: ref %C.2 = bind_name v, %v.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc4_9.1: %i32) {
// CHECK:STDOUT:   %N.loc4_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_9.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc4_9.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_18: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_19: %C.2 = converted %.loc13_18, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%v.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.2) {
// CHECK:STDOUT:   %N.loc4_9.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc4_9.2 => constants.%N.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_123.2) {
// CHECK:STDOUT:   %N.loc4_9.2 => constants.%int_123.2
// CHECK:STDOUT:   %N.patt.loc4_9.2 => constants.%int_123.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_class_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [template]
// CHECK:STDOUT:   %struct_type.a.b.1: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.a.b.1 [template]
// CHECK:STDOUT:   %F: %D = bind_symbolic_name F, 0 [symbolic]
// CHECK:STDOUT:   %F.patt: %D = symbolic_binding_pattern F, 0 [symbolic]
// CHECK:STDOUT:   %E.type: type = generic_class_type @E [template]
// CHECK:STDOUT:   %E.generic: %E.type = struct_value () [template]
// CHECK:STDOUT:   %E: type = class_type @E, @E(%F) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %struct_type.a.b.2: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.19: <witness> = impl_witness (imports.%import_ref.14), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_4.1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_3.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_4.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4.2: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %D.val.2: %D = struct_value (%int_3.2, %int_4.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .g = %g
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %E.decl: %E.type = class_decl @E [template = constants.%E.generic] {
// CHECK:STDOUT:     %F.patt.loc9_9.1: %D = symbolic_binding_pattern F, 0 [symbolic = %F.patt.loc9_9.2 (constants.%F.patt)]
// CHECK:STDOUT:     %F.param_patt: %D = value_param_pattern %F.patt.loc9_9.1, runtime_param<invalid> [symbolic = %F.patt.loc9_9.2 (constants.%F.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %F.param: %D = value_param runtime_param<invalid>
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %F.loc9_9.1: %D = bind_symbolic_name F, 0, %F.param [symbolic = %F.loc9_9.2 (constants.%F)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %g.var: ref <error> = var g
// CHECK:STDOUT:   %g: ref <error> = bind_name g, %g.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc5: %D.elem = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc6: %D.elem = field_decl b, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b.1 [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .a = %.loc5
// CHECK:STDOUT:   .b = %.loc6
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @E(%F.loc9_9.1: %D) {
// CHECK:STDOUT:   %F.loc9_9.2: %D = bind_symbolic_name F, 0 [symbolic = %F.loc9_9.2 (constants.%F)]
// CHECK:STDOUT:   %F.patt.loc9_9.2: %D = symbolic_binding_pattern F, 0 [symbolic = %F.patt.loc9_9.2 (constants.%F.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.4]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%E
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_31: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, file.%E.decl [template = constants.%E.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4.1]
// CHECK:STDOUT:   %.loc24_53.1: %struct_type.a.b.2 = struct_literal (%int_3, %int_4)
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %impl.elem0.loc24_53.1: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc24_53.1: <bound method> = bound_method %int_3, %impl.elem0.loc24_53.1 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc24_53.1: <specific function> = specific_function %Convert.bound.loc24_53.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc24_53.1: init %i32 = call %Convert.specific_fn.loc24_53.1(%int_3) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc24_53.2: init %i32 = converted %int_3, %int.convert_checked.loc24_53.1 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc24_53.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc24_53.4: ref %i32 = class_element_access %.loc24_53.3, element0
// CHECK:STDOUT:   %.loc24_53.5: init %i32 = initialize_from %.loc24_53.2 to %.loc24_53.4 [template = constants.%int_3.2]
// CHECK:STDOUT:   %impl.elem0.loc24_53.2: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc24_53.2: <bound method> = bound_method %int_4, %impl.elem0.loc24_53.2 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc24_53.2: <specific function> = specific_function %Convert.bound.loc24_53.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc24_53.2: init %i32 = call %Convert.specific_fn.loc24_53.2(%int_4) [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc24_53.6: init %i32 = converted %int_4, %int.convert_checked.loc24_53.2 [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc24_53.7: ref %i32 = class_element_access %.loc24_53.3, element1
// CHECK:STDOUT:   %.loc24_53.8: init %i32 = initialize_from %.loc24_53.6 to %.loc24_53.7 [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc24_53.9: init %D = class_init (%.loc24_53.5, %.loc24_53.8), %.loc24_53.3 [template = constants.%D.val.2]
// CHECK:STDOUT:   %.loc24_53.10: ref %D = temporary %.loc24_53.3, %.loc24_53.9
// CHECK:STDOUT:   %.loc24_55.1: ref %D = converted %.loc24_53.1, %.loc24_53.10
// CHECK:STDOUT:   %.loc24_55.2: %D = bind_value %.loc24_55.1
// CHECK:STDOUT:   assign file.%g.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E(constants.%F) {
// CHECK:STDOUT:   %F.loc9_9.2 => constants.%F
// CHECK:STDOUT:   %F.patt.loc9_9.2 => constants.%F
// CHECK:STDOUT: }
// CHECK:STDOUT:
