// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDERR: fail_empty_params.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `NoParams` to `EmptyParams()` [ConversionFailure]
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
// CHECK:STDERR: fail_nested.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `Outer({}*)` to `Inner({.a: i32}*)` [ConversionFailure]
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

// CHECK:STDERR: fail_int_value.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `()` to `C(123)` [ConversionFailure]
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.007: <witness> = impl_witness @NoParams.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.4c4: type = ptr_type %NoParams [concrete]
// CHECK:STDOUT:   %pattern_type.105: type = pattern_type %ptr.4c4 [concrete]
// CHECK:STDOUT:   %NoParams.as.Destroy.impl.Op.type: type = fn_type @NoParams.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %NoParams.as.Destroy.impl.Op: %NoParams.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %EmptyParams.type: type = generic_class_type @EmptyParams [concrete]
// CHECK:STDOUT:   %EmptyParams.generic: %EmptyParams.type = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyParams: type = class_type @EmptyParams [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ab8: <witness> = impl_witness @EmptyParams.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.3ae: type = ptr_type %EmptyParams [concrete]
// CHECK:STDOUT:   %pattern_type.73b: type = pattern_type %ptr.3ae [concrete]
// CHECK:STDOUT:   %EmptyParams.as.Destroy.impl.Op.type: type = fn_type @EmptyParams.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %EmptyParams.as.Destroy.impl.Op: %EmptyParams.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.32d: type = pattern_type %NoParams [concrete]
// CHECK:STDOUT:   %pattern_type.71c: type = pattern_type %EmptyParams [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %v.patt: %pattern_type.32d = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.32d = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %NoParams = var %v.var_patt [concrete]
// CHECK:STDOUT:   %NoParams.ref: type = name_ref NoParams, %NoParams.decl [concrete = constants.%NoParams]
// CHECK:STDOUT:   %v: ref %NoParams = bind_name v, %v.var [concrete = %v.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.71c = binding_pattern w [concrete]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.71c = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %EmptyParams = var %w.var_patt [concrete]
// CHECK:STDOUT:   %.loc15: type = splice_block %EmptyParams [concrete = constants.%EmptyParams] {
// CHECK:STDOUT:     %EmptyParams.ref: %EmptyParams.type = name_ref EmptyParams, %EmptyParams.decl [concrete = constants.%EmptyParams.generic]
// CHECK:STDOUT:     %EmptyParams: type = class_type @EmptyParams [concrete = constants.%EmptyParams]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: ref %EmptyParams = bind_name w, %w.var [concrete = %w.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @NoParams.as.Destroy.impl: @NoParams.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %NoParams.as.Destroy.impl.Op.decl: %NoParams.as.Destroy.impl.Op.type = fn_decl @NoParams.as.Destroy.impl.Op [concrete = constants.%NoParams.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.105 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.105 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.4c4 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoParams [concrete = constants.%NoParams]
// CHECK:STDOUT:     %self: %ptr.4c4 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %NoParams.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @NoParams.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @EmptyParams.as.Destroy.impl: @EmptyParams.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %EmptyParams.as.Destroy.impl.Op.decl: %EmptyParams.as.Destroy.impl.Op.type = fn_decl @EmptyParams.as.Destroy.impl.Op [concrete = constants.%EmptyParams.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.73b = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.73b = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc5: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.3ae = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%EmptyParams [concrete = constants.%EmptyParams]
// CHECK:STDOUT:     %self: %ptr.3ae = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %EmptyParams.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @EmptyParams.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoParams {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%NoParams [concrete = constants.%NoParams]
// CHECK:STDOUT:   impl_decl @NoParams.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@NoParams.as.Destroy.impl.%NoParams.as.Destroy.impl.Op.decl), @NoParams.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.007]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoParams
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @EmptyParams {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%EmptyParams [concrete = constants.%EmptyParams]
// CHECK:STDOUT:   impl_decl @EmptyParams.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@EmptyParams.as.Destroy.impl.%EmptyParams.as.Destroy.impl.Op.decl), @EmptyParams.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.ab8]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%EmptyParams
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoParams.as.Destroy.impl.Op(%self.param: %ptr.4c4) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @EmptyParams.as.Destroy.impl.Op(%self.param: %ptr.3ae) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: ref %NoParams = name_ref v, file.%v [concrete = file.%v.var]
// CHECK:STDOUT:   %.loc15: %EmptyParams = converted %v.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%w.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.9d6: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.eae: type = generic_class_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.generic.137: %Inner.type.eae = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.c71: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.7f3: <witness> = impl_witness @Inner.%Destroy.impl_witness_table, @Inner.as.Destroy.impl(%T, %U) [symbolic]
// CHECK:STDOUT:   %ptr.276: type = ptr_type %Inner.c71 [symbolic]
// CHECK:STDOUT:   %pattern_type.01f: type = pattern_type %ptr.276 [symbolic]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.type: type = fn_type @Inner.as.Destroy.impl.Op, @Inner.as.Destroy.impl(%T, %U) [symbolic]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op: %Inner.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.52f: <witness> = impl_witness @Outer.%Destroy.impl_witness_table, @Outer.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.6ff: type = ptr_type %Outer.9d6 [symbolic]
// CHECK:STDOUT:   %pattern_type.07e: type = pattern_type %ptr.6ff [symbolic]
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.type: type = fn_type @Outer.as.Destroy.impl.Op, @Outer.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op: %Outer.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %Outer.614: type = class_type @Outer, @Outer(%ptr.c28) [concrete]
// CHECK:STDOUT:   %Inner.type.5d2: type = generic_class_type @Inner, @Outer(%ptr.c28) [concrete]
// CHECK:STDOUT:   %Inner.generic.8e6: %Inner.type.5d2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.828: type = pattern_type %Outer.614 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %ptr.1bb: type = ptr_type %struct_type.a [concrete]
// CHECK:STDOUT:   %Inner.277: type = class_type @Inner, @Inner(%ptr.c28, %ptr.1bb) [concrete]
// CHECK:STDOUT:   %pattern_type.5fa: type = pattern_type %Inner.277 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.828 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.828 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Outer.614 = var %v.var_patt [concrete]
// CHECK:STDOUT:   %.loc9_17: type = splice_block %Outer.loc9 [concrete = constants.%Outer.614] {
// CHECK:STDOUT:     %Outer.ref.loc9: %Outer.type = name_ref Outer, %Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %.loc9_15: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_16: type = converted %.loc9_15, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc9: type = ptr_type %.loc9_16 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %Outer.loc9: type = class_type @Outer, @Outer(constants.%ptr.c28) [concrete = constants.%Outer.614]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Outer.614 = bind_name v, %v.var [concrete = %v.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.5fa = binding_pattern w [concrete]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.5fa = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %Inner.277 = var %w.var_patt [concrete]
// CHECK:STDOUT:   %.loc19_35: type = splice_block %Inner [concrete = constants.%Inner.277] {
// CHECK:STDOUT:     %Outer.ref.loc19: %Outer.type = name_ref Outer, %Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %.loc19_15: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc19_16: type = converted %.loc19_15, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc19_16: type = ptr_type %.loc19_16 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %Outer.loc19: type = class_type @Outer, @Outer(constants.%ptr.c28) [concrete = constants.%Outer.614]
// CHECK:STDOUT:     %.loc19_18: %Inner.type.5d2 = specific_constant @Outer.%Inner.decl, @Outer(constants.%ptr.c28) [concrete = constants.%Inner.generic.8e6]
// CHECK:STDOUT:     %Inner.ref: %Inner.type.5d2 = name_ref Inner, %.loc19_18 [concrete = constants.%Inner.generic.8e6]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %ptr.loc19_34: type = ptr_type %struct_type.a [concrete = constants.%ptr.1bb]
// CHECK:STDOUT:     %Inner: type = class_type @Inner, @Inner(constants.%ptr.c28, constants.%ptr.1bb) [concrete = constants.%Inner.277]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: ref %Inner.277 = bind_name w, %w.var [concrete = %w.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Inner.as.Destroy.impl(@Outer.%T.loc4_13.2: type, @Inner.%U.loc5_15.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T, %U) [symbolic = %Inner (constants.%Inner.c71)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Inner.%Destroy.impl_witness_table, @Inner.as.Destroy.impl(%T, %U) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.7f3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.type: type = fn_type @Inner.as.Destroy.impl.Op, @Inner.as.Destroy.impl(%T, %U) [symbolic = %Inner.as.Destroy.impl.Op.type (constants.%Inner.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op: @Inner.as.Destroy.impl.%Inner.as.Destroy.impl.Op.type (%Inner.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Inner.as.Destroy.impl.Op (constants.%Inner.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Inner.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Inner.as.Destroy.impl.Op.decl: @Inner.as.Destroy.impl.%Inner.as.Destroy.impl.Op.type (%Inner.as.Destroy.impl.Op.type) = fn_decl @Inner.as.Destroy.impl.Op [symbolic = @Inner.as.Destroy.impl.%Inner.as.Destroy.impl.Op (constants.%Inner.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Inner.as.Destroy.impl.Op.%pattern_type (%pattern_type.01f) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Inner.as.Destroy.impl.Op.%pattern_type (%pattern_type.01f) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc5_25.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Inner.as.Destroy.impl.Op.%ptr (%ptr.276) = value_param call_param0
// CHECK:STDOUT:       %.loc5_25.2: type = splice_block %Self.ref [symbolic = %Inner (constants.%Inner.c71)] {
// CHECK:STDOUT:         %.loc5_25.3: type = specific_constant constants.%Inner.c71, @Inner(constants.%T, constants.%U) [symbolic = %Inner (constants.%Inner.c71)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_25.3 [symbolic = %Inner (constants.%Inner.c71)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Inner.as.Destroy.impl.Op.%ptr (%ptr.276) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Inner.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Inner.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Outer.as.Destroy.impl(@Outer.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(%T) [symbolic = %Outer (constants.%Outer.9d6)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Outer.%Destroy.impl_witness_table, @Outer.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.52f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.type: type = fn_type @Outer.as.Destroy.impl.Op, @Outer.as.Destroy.impl(%T) [symbolic = %Outer.as.Destroy.impl.Op.type (constants.%Outer.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op: @Outer.as.Destroy.impl.%Outer.as.Destroy.impl.Op.type (%Outer.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Outer.as.Destroy.impl.Op (constants.%Outer.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Outer.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Outer.as.Destroy.impl.Op.decl: @Outer.as.Destroy.impl.%Outer.as.Destroy.impl.Op.type (%Outer.as.Destroy.impl.Op.type) = fn_decl @Outer.as.Destroy.impl.Op [symbolic = @Outer.as.Destroy.impl.%Outer.as.Destroy.impl.Op (constants.%Outer.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Outer.as.Destroy.impl.Op.%pattern_type (%pattern_type.07e) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Outer.as.Destroy.impl.Op.%pattern_type (%pattern_type.07e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Outer.as.Destroy.impl.Op.%ptr (%ptr.6ff) = value_param call_param0
// CHECK:STDOUT:       %.loc4_23.2: type = splice_block %Self.ref [symbolic = %Outer (constants.%Outer.9d6)] {
// CHECK:STDOUT:         %.loc4_23.3: type = specific_constant constants.%Outer.9d6, @Outer(constants.%T) [symbolic = %Outer (constants.%Outer.9d6)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_23.3 [symbolic = %Outer (constants.%Outer.9d6)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Outer.as.Destroy.impl.Op.%ptr (%ptr.6ff) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Outer.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Outer.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc4_13.1) [symbolic = %Inner.type (constants.%Inner.type.eae)]
// CHECK:STDOUT:   %Inner.generic: @Outer.%Inner.type (%Inner.type.eae) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.137)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.eae) = class_decl @Inner [symbolic = @Outer.%Inner.generic (constants.%Inner.generic.137)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc5_15.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc5_15.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Outer.9d6 [symbolic = @Outer.as.Destroy.impl.%Outer (constants.%Outer.9d6)]
// CHECK:STDOUT:     impl_decl @Outer.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Outer.as.Destroy.impl.%Outer.as.Destroy.impl.Op.decl), @Outer.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Outer.as.Destroy.impl(constants.%T) [symbolic = @Outer.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.52f)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.9d6
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_13.2: type, %U.loc5_15.2: type) {
// CHECK:STDOUT:   %U.loc5_15.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc5_15.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Inner.c71 [symbolic = @Inner.as.Destroy.impl.%Inner (constants.%Inner.c71)]
// CHECK:STDOUT:     impl_decl @Inner.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Inner.as.Destroy.impl.%Inner.as.Destroy.impl.Op.decl), @Inner.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Inner.as.Destroy.impl(constants.%T, constants.%U) [symbolic = @Inner.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.7f3)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.c71
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.as.Destroy.impl.Op(@Outer.%T.loc4_13.2: type, @Inner.%U.loc5_15.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T, %U) [symbolic = %Inner (constants.%Inner.c71)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Inner [symbolic = %ptr (constants.%ptr.276)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.01f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Inner.as.Destroy.impl.Op.%ptr (%ptr.276)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.as.Destroy.impl.Op(@Outer.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(%T) [symbolic = %Outer (constants.%Outer.9d6)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Outer [symbolic = %ptr (constants.%ptr.6ff)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.07e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Outer.as.Destroy.impl.Op.%ptr (%ptr.6ff)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: ref %Outer.614 = name_ref v, file.%v [concrete = file.%v.var]
// CHECK:STDOUT:   %.loc19: %Inner.277 = converted %v.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%w.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc5_15.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.as.Destroy.impl(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Inner => constants.%Inner.c71
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.7f3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.as.Destroy.impl.Op(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Inner => constants.%Inner.c71
// CHECK:STDOUT:   %ptr => constants.%ptr.276
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.01f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Outer => constants.%Outer.9d6
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.52f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Outer => constants.%Outer.9d6
// CHECK:STDOUT:   %ptr => constants.%ptr.6ff
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.07e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%ptr.c28) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%ptr.c28
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5d2
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.8e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%ptr.c28, constants.%ptr.1bb) {
// CHECK:STDOUT:   %U.loc5_15.1 => constants.%ptr.1bb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_int_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.506: type = class_type @C, @C(%N.51e) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%N.51e) [symbolic]
// CHECK:STDOUT:   %ptr.128: type = ptr_type %C.506 [symbolic]
// CHECK:STDOUT:   %pattern_type.d64: type = pattern_type %ptr.128 [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%N.51e) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.f7f: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %C.4c3: type = class_type @C, @C(%int_123.f7f) [concrete]
// CHECK:STDOUT:   %pattern_type.1b4: type = pattern_type %C.4c3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_9.1 (constants.%N.51e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.1b4 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.1b4 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %C.4c3 = var %v.var_patt [concrete]
// CHECK:STDOUT:   %.loc13_13.1: type = splice_block %C [concrete = constants.%C.4c3] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:     %impl.elem0: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:     %bound_method.loc13_13.1: <bound method> = bound_method %int_123, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_13.2: <bound method> = bound_method %int_123, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_13.2(%int_123) [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:     %.loc13_13.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:     %.loc13_13.3: %i32 = converted %int_123, %.loc13_13.2 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_123.f7f) [concrete = constants.%C.4c3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %C.4c3 = bind_name v, %v.var [concrete = %v.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(@C.%N.loc4_9.2: %i32) {
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.51e)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%N) [symbolic = %C (constants.%C.506)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%N) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%N) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.decl: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = fn_decl @C.as.Destroy.impl.Op [symbolic = @C.as.Destroy.impl.%C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.d64) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.d64) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_18.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.128) = value_param call_param0
// CHECK:STDOUT:       %.loc4_18.2: type = splice_block %Self.ref [symbolic = %C (constants.%C.506)] {
// CHECK:STDOUT:         %.loc4_18.3: type = specific_constant constants.%C.506, @C(constants.%N.51e) [symbolic = %C (constants.%C.506)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_18.3 [symbolic = %C (constants.%C.506)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Destroy.impl.Op.%ptr (%ptr.128) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @C.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc4_9.2: %i32) {
// CHECK:STDOUT:   %N.loc4_9.1: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_9.1 (constants.%N.51e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C.506 [symbolic = @C.as.Destroy.impl.%C (constants.%C.506)]
// CHECK:STDOUT:     impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @C.as.Destroy.impl(constants.%N.51e) [symbolic = @C.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.506
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(@C.%N.loc4_9.2: %i32) {
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.51e)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%N) [symbolic = %C (constants.%C.506)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.128)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.d64)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.128)) = "no_op";
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
// CHECK:STDOUT:   %N.loc4_9.1 => constants.%N.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%N.51e) {
// CHECK:STDOUT:   %N => constants.%N.51e
// CHECK:STDOUT:   %C => constants.%C.506
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%N.51e) {
// CHECK:STDOUT:   %N => constants.%N.51e
// CHECK:STDOUT:   %C => constants.%C.506
// CHECK:STDOUT:   %ptr => constants.%ptr.128
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_123.f7f) {
// CHECK:STDOUT:   %N.loc4_9.1 => constants.%int_123.f7f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_class_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.cd5: <witness> = impl_witness @D.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.a94: type = pattern_type %ptr.19c [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: %D.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.501: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.705: <witness> = complete_type_witness %struct_type.a.b.501 [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %F: %D = bind_symbolic_name F, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %E.type: type = generic_class_type @E [concrete]
// CHECK:STDOUT:   %E.generic: %E.type = struct_value () [concrete]
// CHECK:STDOUT:   %E: type = class_type @E, @E(%F) [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.520: <witness> = impl_witness @E.%Destroy.impl_witness_table, @E.as.Destroy.impl(%F) [symbolic]
// CHECK:STDOUT:   %ptr.4e0: type = ptr_type %E [symbolic]
// CHECK:STDOUT:   %pattern_type.72c: type = pattern_type %ptr.4e0 [symbolic]
// CHECK:STDOUT:   %E.as.Destroy.impl.Op.type: type = fn_type @E.as.Destroy.impl.Op, @E.as.Destroy.impl(%F) [symbolic]
// CHECK:STDOUT:   %E.as.Destroy.impl.Op: %E.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c11: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.8bd: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %D.val.413: %D = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.f36: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5bb: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.9cd: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %D.val.835: %D = struct_value (%int_3.822, %int_4.940) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %F.patt: %pattern_type.510 = symbolic_binding_pattern F, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9: type = splice_block %D.ref [concrete = constants.%D] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.loc9_9.2: %D = bind_symbolic_name F, 0 [symbolic = %F.loc9_9.1 (constants.%F)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %g.patt: <error> = binding_pattern g [concrete]
// CHECK:STDOUT:     %g.var_patt: <error> = var_pattern %g.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %g.var: ref <error> = var %g.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %E.ref: %E.type = name_ref E, %E.decl [concrete = constants.%E.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc25_25.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:     %impl.elem0.loc25_25.1: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:     %bound_method.loc25_25.1: <bound method> = bound_method %int_1, %impl.elem0.loc25_25.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:     %specific_fn.loc25_25.1: <specific function> = specific_function %impl.elem0.loc25_25.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc25_25.2: <bound method> = bound_method %int_1, %specific_fn.loc25_25.1 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25_25.1: init %i32 = call %bound_method.loc25_25.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc25_25.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25_25.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc25_25.3: ref %D = temporary_storage
// CHECK:STDOUT:     %.loc25_25.4: ref %i32 = class_element_access %.loc25_25.3, element0
// CHECK:STDOUT:     %.loc25_25.5: init %i32 = initialize_from %.loc25_25.2 to %.loc25_25.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %impl.elem0.loc25_25.2: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:     %bound_method.loc25_25.3: <bound method> = bound_method %int_2, %impl.elem0.loc25_25.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:     %specific_fn.loc25_25.2: <specific function> = specific_function %impl.elem0.loc25_25.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc25_25.4: <bound method> = bound_method %int_2, %specific_fn.loc25_25.2 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25_25.2: init %i32 = call %bound_method.loc25_25.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc25_25.6: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25_25.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc25_25.7: ref %i32 = class_element_access %.loc25_25.3, element1
// CHECK:STDOUT:     %.loc25_25.8: init %i32 = initialize_from %.loc25_25.6 to %.loc25_25.7 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc25_25.9: init %D = class_init (%.loc25_25.5, %.loc25_25.8), %.loc25_25.3 [concrete = constants.%D.val.413]
// CHECK:STDOUT:     %.loc25_25.10: ref %D = temporary %.loc25_25.3, %.loc25_25.9
// CHECK:STDOUT:     %.loc25_26.1: ref %D = converted %.loc25_25.1, %.loc25_25.10
// CHECK:STDOUT:     %.loc25_26.2: %D = bind_value %.loc25_26.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %g: <error> = bind_name g, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Destroy.impl: @D.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.decl: %D.as.Destroy.impl.Op.type = fn_decl @D.as.Destroy.impl.Op [concrete = constants.%D.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a94 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a94 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.19c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %self: %ptr.19c = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %D.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @D.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @E.as.Destroy.impl(@E.%F.loc9_9.2: %D) {
// CHECK:STDOUT:   %F: %D = bind_symbolic_name F, 0 [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %E: type = class_type @E, @E(%F) [symbolic = %E (constants.%E)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @E.%Destroy.impl_witness_table, @E.as.Destroy.impl(%F) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.520)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %E.as.Destroy.impl.Op.type: type = fn_type @E.as.Destroy.impl.Op, @E.as.Destroy.impl(%F) [symbolic = %E.as.Destroy.impl.Op.type (constants.%E.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %E.as.Destroy.impl.Op: @E.as.Destroy.impl.%E.as.Destroy.impl.Op.type (%E.as.Destroy.impl.Op.type) = struct_value () [symbolic = %E.as.Destroy.impl.Op (constants.%E.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @E.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %E.as.Destroy.impl.Op.decl: @E.as.Destroy.impl.%E.as.Destroy.impl.Op.type (%E.as.Destroy.impl.Op.type) = fn_decl @E.as.Destroy.impl.Op [symbolic = @E.as.Destroy.impl.%E.as.Destroy.impl.Op (constants.%E.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @E.as.Destroy.impl.Op.%pattern_type (%pattern_type.72c) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @E.as.Destroy.impl.Op.%pattern_type (%pattern_type.72c) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc9_16.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @E.as.Destroy.impl.Op.%ptr (%ptr.4e0) = value_param call_param0
// CHECK:STDOUT:       %.loc9_16.2: type = splice_block %Self.ref [symbolic = %E (constants.%E)] {
// CHECK:STDOUT:         %.loc9_16.3: type = specific_constant constants.%E, @E(constants.%F) [symbolic = %E (constants.%E)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc9_16.3 [symbolic = %E (constants.%E)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @E.as.Destroy.impl.Op.%ptr (%ptr.4e0) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %E.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @E.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %D.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc6: %D.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:   impl_decl @D.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@D.as.Destroy.impl.%D.as.Destroy.impl.Op.decl), @D.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.cd5]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b.501 [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .a = %.loc5
// CHECK:STDOUT:   .b = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @E(%F.loc9_9.2: %D) {
// CHECK:STDOUT:   %F.loc9_9.1: %D = bind_symbolic_name F, 0 [symbolic = %F.loc9_9.1 (constants.%F)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [symbolic = @E.as.Destroy.impl.%E (constants.%E)]
// CHECK:STDOUT:     impl_decl @E.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@E.as.Destroy.impl.%E.as.Destroy.impl.Op.decl), @E.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @E.as.Destroy.impl(constants.%F) [symbolic = @E.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.520)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%E
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Destroy.impl.Op(%self.param: %ptr.19c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @E.as.Destroy.impl.Op(@E.%F.loc9_9.2: %D) {
// CHECK:STDOUT:   %F: %D = bind_symbolic_name F, 0 [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %E: type = class_type @E, @E(%F) [symbolic = %E (constants.%E)]
// CHECK:STDOUT:   %ptr: type = ptr_type %E [symbolic = %ptr (constants.%ptr.4e0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.72c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @E.as.Destroy.impl.Op.%ptr (%ptr.4e0)) = "no_op";
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
// CHECK:STDOUT:   %impl.elem0.loc25_53.1: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc25_53.1: <bound method> = bound_method %int_3, %impl.elem0.loc25_53.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595]
// CHECK:STDOUT:   %specific_fn.loc25_53.1: <specific function> = specific_function %impl.elem0.loc25_53.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_53.2: <bound method> = bound_method %int_3, %specific_fn.loc25_53.1 [concrete = constants.%bound_method.f36]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25_53.1: init %i32 = call %bound_method.loc25_53.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc25_53.2: init %i32 = converted %int_3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25_53.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc25_53.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc25_53.4: ref %i32 = class_element_access %.loc25_53.3, element0
// CHECK:STDOUT:   %.loc25_53.5: init %i32 = initialize_from %.loc25_53.2 to %.loc25_53.4 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %impl.elem0.loc25_53.2: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc25_53.3: <bound method> = bound_method %int_4, %impl.elem0.loc25_53.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5bb]
// CHECK:STDOUT:   %specific_fn.loc25_53.2: <specific function> = specific_function %impl.elem0.loc25_53.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_53.4: <bound method> = bound_method %int_4, %specific_fn.loc25_53.2 [concrete = constants.%bound_method.9cd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25_53.2: init %i32 = call %bound_method.loc25_53.4(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc25_53.6: init %i32 = converted %int_4, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25_53.2 [concrete = constants.%int_4.940]
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
// CHECK:STDOUT:   %F.loc9_9.1 => constants.%F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E.as.Destroy.impl(constants.%F) {
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %E => constants.%E
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.520
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E.as.Destroy.impl.Op(constants.%F) {
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %E => constants.%E
// CHECK:STDOUT:   %ptr => constants.%ptr.4e0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.72c
// CHECK:STDOUT: }
// CHECK:STDOUT:
