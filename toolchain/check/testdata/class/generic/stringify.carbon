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
// CHECK:STDERR: fail_empty_params.carbon:[[@LINE+4]]:1: note: type `NoParams` does not implement interface `ImplicitAs(EmptyParams())` [MissingImplInMemberAccessNote]
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
// CHECK:STDERR: fail_nested.carbon:[[@LINE+6]]:1: error: cannot implicitly convert from `Outer({}*)` to `Inner({.a: i32}*)` [ImplicitAsConversionFailure]
// CHECK:STDERR: var w: Outer({}*).Inner({.a: i32}*) = v;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_nested.carbon:[[@LINE+3]]:1: note: type `Outer({}*)` does not implement interface `ImplicitAs(Inner({.a: i32}*))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var w: Outer({}*).Inner({.a: i32}*) = v;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
var w: Outer({}*).Inner({.a: i32}*) = v;

// CHECK:STDOUT: --- fail_empty_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NoParams: type = class_type @NoParams [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %EmptyParams.type: type = generic_class_type @EmptyParams [template]
// CHECK:STDOUT:   %EmptyParams.1: %EmptyParams.type = struct_value () [template]
// CHECK:STDOUT:   %EmptyParams.2: type = class_type @EmptyParams [template]
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
// CHECK:STDOUT:   %EmptyParams.decl: %EmptyParams.type = class_decl @EmptyParams [template = constants.%EmptyParams.1] {} {}
// CHECK:STDOUT:   %NoParams.ref: type = name_ref NoParams, %NoParams.decl [template = constants.%NoParams]
// CHECK:STDOUT:   %v.var: ref %NoParams = var v
// CHECK:STDOUT:   %v: ref %NoParams = bind_name v, %v.var
// CHECK:STDOUT:   %EmptyParams.ref: %EmptyParams.type = name_ref EmptyParams, %EmptyParams.decl [template = constants.%EmptyParams.1]
// CHECK:STDOUT:   %EmptyParams: type = class_type @EmptyParams [template = constants.%EmptyParams.2]
// CHECK:STDOUT:   %w.var: ref %EmptyParams.2 = var w
// CHECK:STDOUT:   %w: ref %EmptyParams.2 = bind_name w, %w.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoParams {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoParams
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @EmptyParams {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%EmptyParams.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: ref %NoParams = name_ref v, file.%v
// CHECK:STDOUT:   %.loc15: %EmptyParams.2 = converted %v.ref, <error> [template = <error>]
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
// CHECK:STDOUT:   %Outer.1: %Outer.type = struct_value () [template]
// CHECK:STDOUT:   %Outer.2: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.1: type = generic_class_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.1: %Inner.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.2: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %Outer.3: type = class_type @Outer, @Outer(%.3) [template]
// CHECK:STDOUT:   %Inner.type.2: type = generic_class_type @Inner, @Outer(%.3) [template]
// CHECK:STDOUT:   %Inner.3: %Inner.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.4 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %Inner.4: type = class_type @Inner, @Inner(%.3, %.6) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
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
// CHECK:STDOUT:   %Outer.decl: %Outer.type = class_decl @Outer [template = constants.%Outer.1] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Outer.ref.loc9: %Outer.type = name_ref Outer, %Outer.decl [template = constants.%Outer.1]
// CHECK:STDOUT:   %.loc9_15: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc9_16.1: type = converted %.loc9_15, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_16.2: type = ptr_type %.1 [template = constants.%.3]
// CHECK:STDOUT:   %Outer.loc9: type = class_type @Outer, @Outer(constants.%.3) [template = constants.%Outer.3]
// CHECK:STDOUT:   %v.var: ref %Outer.3 = var v
// CHECK:STDOUT:   %v: ref %Outer.3 = bind_name v, %v.var
// CHECK:STDOUT:   %Outer.ref.loc18: %Outer.type = name_ref Outer, %Outer.decl [template = constants.%Outer.1]
// CHECK:STDOUT:   %.loc18_15: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc18_16.1: type = converted %.loc18_15, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc18_16.2: type = ptr_type %.1 [template = constants.%.3]
// CHECK:STDOUT:   %Outer.loc18: type = class_type @Outer, @Outer(constants.%.3) [template = constants.%Outer.3]
// CHECK:STDOUT:   %.loc18_18: %Inner.type.2 = specific_constant @Outer.%Inner.decl, @Outer(constants.%.3) [template = constants.%Inner.3]
// CHECK:STDOUT:   %Inner.ref: %Inner.type.2 = name_ref Inner, %.loc18_18 [template = constants.%Inner.3]
// CHECK:STDOUT:   %.loc18_30.1: Core.IntLiteral = int_value 32 [template = constants.%.4]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc18_30.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_30.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_30.3: type = converted %int.make_type_signed, %.loc18_30.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_33: type = struct_type {.a: %i32} [template = constants.%.5]
// CHECK:STDOUT:   %.loc18_34: type = ptr_type %.5 [template = constants.%.6]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(constants.%.3, constants.%.6) [template = constants.%Inner.4]
// CHECK:STDOUT:   %w.var: ref %Inner.4 = var w
// CHECK:STDOUT:   %w: ref %Inner.4 = bind_name w, %w.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc4_13.2) [symbolic = %Inner.type (constants.%Inner.type.1)]
// CHECK:STDOUT:   %Inner: @Outer.%Inner.type (%Inner.type.1) = struct_value () [symbolic = %Inner (constants.%Inner.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.1) = class_decl @Inner [symbolic = @Outer.%Inner (constants.%Inner.1)] {
// CHECK:STDOUT:       %U.patt.loc5_15.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc5_15.2 (constants.%U.patt)]
// CHECK:STDOUT:       %U.param_patt: type = value_param_pattern %U.patt.loc5_15.1, runtime_param<invalid> [symbolic = %U.patt.loc5_15.2 (constants.%U.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %U.loc5_15.1: type = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc5_15.2 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.2
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
// CHECK:STDOUT:     %.loc6: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: ref %Outer.3 = name_ref v, file.%v
// CHECK:STDOUT:   %.loc18: %Inner.4 = converted %v.ref, <error> [template = <error>]
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
// CHECK:STDOUT: specific @Outer(%T.loc4_13.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%.3) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%.3
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.2
// CHECK:STDOUT:   %Inner => constants.%Inner.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%.3, constants.%.6) {
// CHECK:STDOUT:   %U.loc5_15.2 => constants.%.6
// CHECK:STDOUT:   %U.patt.loc5_15.2 => constants.%.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
