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
// CHECK:STDERR: fail_empty_params.carbon:[[@LINE+4]]:1: note: type `NoParams` does not implement interface `Core.ImplicitAs(EmptyParams())` [MissingImplInMemberAccessInContext]
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
// CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:1: note: type `Outer({}*)` does not implement interface `Core.ImplicitAs(Inner({.a: i32}*))` [MissingImplInMemberAccessInContext]
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
// CHECK:STDERR: fail_int_value.carbon:[[@LINE+4]]:1: note: type `()` does not implement interface `Core.ImplicitAs(C(123))` [MissingImplInMemberAccessInContext]
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

// CHECK:STDERR: fail_class_param.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `E({.a = 3, .b = 4})` to `E({.a = 1, .b = 2})` [ConversionFailure]
// CHECK:STDERR: var g: E({.a = 1, .b = 2}) = {} as E({.a = 3, .b = 4} as D);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_class_param.carbon:[[@LINE+4]]:1: note: type `E({.a = 3, .b = 4})` does not implement interface `Core.ImplicitAs(E({.a = 1, .b = 2}))` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var g: E({.a = 1, .b = 2}) = {} as E({.a = 3, .b = 4} as D);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %pattern_type.89c: type = pattern_type %NoParams [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.d5a: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.9f4: %T.as.DefaultOrUnformed.impl.Op.type.d5a = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.59f: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.f82, @T.as.DefaultOrUnformed.impl(%NoParams) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.a4d: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%NoParams) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.363: %T.as.DefaultOrUnformed.impl.Op.type.a4d = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %NoParams, (%DefaultOrUnformed.impl_witness.59f) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.363, @T.as.DefaultOrUnformed.impl.Op(%NoParams) [concrete]
// CHECK:STDOUT:   %pattern_type.7a5: type = pattern_type %EmptyParams [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.b30: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.d5a) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.9f4)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.f82 = impl_witness_table (%Core.import_ref.b30), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %v.patt: %pattern_type.89c = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.89c = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %NoParams = var %v.var_patt [concrete]
// CHECK:STDOUT:   %NoParams.ref: type = name_ref NoParams, %NoParams.decl [concrete = constants.%NoParams]
// CHECK:STDOUT:   %v: ref %NoParams = ref_binding v, %v.var [concrete = %v.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.7a5 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.7a5 = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %EmptyParams = var %w.var_patt [concrete]
// CHECK:STDOUT:   %.loc15: type = splice_block %EmptyParams [concrete = constants.%EmptyParams] {
// CHECK:STDOUT:     %EmptyParams.ref: %EmptyParams.type = name_ref EmptyParams, %EmptyParams.decl [concrete = constants.%EmptyParams.generic]
// CHECK:STDOUT:     %EmptyParams: type = class_type @EmptyParams [concrete = constants.%EmptyParams]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: ref %EmptyParams = ref_binding w, %w.var [concrete = %w.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoParams {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoParams
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @EmptyParams {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%EmptyParams
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%NoParams, (constants.%DefaultOrUnformed.impl_witness.59f) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc7_16.1: %DefaultOrUnformed.type = converted constants.%NoParams, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_16.1 [concrete = constants.%NoParams]
// CHECK:STDOUT:   %.loc7_16.2: type = converted %.loc7_16.1, %as_type [concrete = constants.%NoParams]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.363, @T.as.DefaultOrUnformed.impl.Op(constants.%NoParams) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %.loc7_1: ref %NoParams = splice_block file.%v.var [concrete = file.%v.var] {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %NoParams to %.loc7_1 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign file.%v.var, %T.as.DefaultOrUnformed.impl.Op.call
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
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.22f: type = class_type @Outer, @Outer(%T.67d) [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.4c4: type = generic_class_type @Inner, @Outer(%T.67d) [symbolic]
// CHECK:STDOUT:   %Inner.generic.9fc: %Inner.type.4c4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.64e: type = class_type @Inner, @Inner(%T.67d, %U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %Outer.b4e: type = class_type @Outer, @Outer(%ptr.c28) [concrete]
// CHECK:STDOUT:   %Inner.type.d89: type = generic_class_type @Inner, @Outer(%ptr.c28) [concrete]
// CHECK:STDOUT:   %Inner.generic.67c: %Inner.type.d89 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.b59: type = pattern_type %Outer.b4e [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.d5a: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.9f4: %T.as.DefaultOrUnformed.impl.Op.type.d5a = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.434: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.f82, @T.as.DefaultOrUnformed.impl(%Outer.b4e) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.85f: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%Outer.b4e) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.c18: %T.as.DefaultOrUnformed.impl.Op.type.85f = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %Outer.b4e, (%DefaultOrUnformed.impl_witness.434) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.c18, @T.as.DefaultOrUnformed.impl.Op(%Outer.b4e) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %ptr.452: type = ptr_type %struct_type.a [concrete]
// CHECK:STDOUT:   %Inner.4e2: type = class_type @Inner, @Inner(%ptr.c28, %ptr.452) [concrete]
// CHECK:STDOUT:   %pattern_type.6a0: type = pattern_type %Inner.4e2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.b30: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.d5a) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.9f4)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.f82 = impl_witness_table (%Core.import_ref.b30), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.b59 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.b59 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Outer.b4e = var %v.var_patt [concrete]
// CHECK:STDOUT:   %.loc9_17: type = splice_block %Outer.loc9 [concrete = constants.%Outer.b4e] {
// CHECK:STDOUT:     %Outer.ref.loc9: %Outer.type = name_ref Outer, %Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %.loc9_15: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc9_16: type = converted %.loc9_15, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc9: type = ptr_type %.loc9_16 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %Outer.loc9: type = class_type @Outer, @Outer(constants.%ptr.c28) [concrete = constants.%Outer.b4e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Outer.b4e = ref_binding v, %v.var [concrete = %v.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.6a0 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.6a0 = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %Inner.4e2 = var %w.var_patt [concrete]
// CHECK:STDOUT:   %.loc19_35: type = splice_block %Inner [concrete = constants.%Inner.4e2] {
// CHECK:STDOUT:     %Outer.ref.loc19: %Outer.type = name_ref Outer, %Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %.loc19_15: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc19_16: type = converted %.loc19_15, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc19_16: type = ptr_type %.loc19_16 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %Outer.loc19: type = class_type @Outer, @Outer(constants.%ptr.c28) [concrete = constants.%Outer.b4e]
// CHECK:STDOUT:     %.loc19_18: %Inner.type.d89 = specific_constant @Outer.%Inner.decl, @Outer(constants.%ptr.c28) [concrete = constants.%Inner.generic.67c]
// CHECK:STDOUT:     %Inner.ref: %Inner.type.d89 = name_ref Inner, %.loc19_18 [concrete = constants.%Inner.generic.67c]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %ptr.loc19_34: type = ptr_type %struct_type.a [concrete = constants.%ptr.452]
// CHECK:STDOUT:     %Inner: type = class_type @Inner, @Inner(constants.%ptr.c28, constants.%ptr.452) [concrete = constants.%Inner.4e2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: ref %Inner.4e2 = ref_binding w, %w.var [concrete = %w.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc4_14.1) [symbolic = %Inner.type (constants.%Inner.type.4c4)]
// CHECK:STDOUT:   %Inner.generic: @Outer.%Inner.type (%Inner.type.4c4) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.9fc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.4c4) = class_decl @Inner [symbolic = @Outer.%Inner.generic (constants.%Inner.generic.9fc)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_19.1: type = splice_block %.loc5_19.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc5_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc5_16.2: type = symbolic_binding U, 1 [symbolic = %U.loc5_16.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.22f
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_14.2: type, %U.loc5_16.2: type) {
// CHECK:STDOUT:   %U.loc5_16.1: type = symbolic_binding U, 1 [symbolic = %U.loc5_16.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.64e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Outer.b4e, (constants.%DefaultOrUnformed.impl_witness.434) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc9_18.1: %DefaultOrUnformed.type = converted constants.%Outer.b4e, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc9_18.1 [concrete = constants.%Outer.b4e]
// CHECK:STDOUT:   %.loc9_18.2: type = converted %.loc9_18.1, %as_type [concrete = constants.%Outer.b4e]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.c18, @T.as.DefaultOrUnformed.impl.Op(constants.%Outer.b4e) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %.loc9_1: ref %Outer.b4e = splice_block file.%v.var [concrete = file.%v.var] {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %Outer.b4e to %.loc9_1 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign file.%v.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %v.ref: ref %Outer.b4e = name_ref v, file.%v [concrete = file.%v.var]
// CHECK:STDOUT:   %.loc19: %Inner.4e2 = converted %v.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%w.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.67d, constants.%U) {
// CHECK:STDOUT:   %U.loc5_16.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%ptr.c28) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%ptr.c28
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d89
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.67c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%ptr.c28, constants.%ptr.452) {
// CHECK:STDOUT:   %U.loc5_16.1 => constants.%ptr.452
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_int_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.afc: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.77d: type = class_type @C, @C(%N.afc) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.5a3: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %C.fec: type = class_type @C, @C(%int_123.5a3) [concrete]
// CHECK:STDOUT:   %pattern_type.70b: type = pattern_type %C.fec [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.c6b = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_10.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_10.1 (constants.%N.afc)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.70b = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.70b = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %C.fec = var %v.var_patt [concrete]
// CHECK:STDOUT:   %.loc13_13.1: type = splice_block %C [concrete = constants.%C.fec] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:     %impl.elem0: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:     %bound_method.loc13_13.1: <bound method> = bound_method %int_123, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_13.2: <bound method> = bound_method %int_123, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_13.2(%int_123) [concrete = constants.%int_123.5a3]
// CHECK:STDOUT:     %.loc13_13.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_123.5a3]
// CHECK:STDOUT:     %.loc13_13.3: %i32 = converted %int_123, %.loc13_13.2 [concrete = constants.%int_123.5a3]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_123.5a3) [concrete = constants.%C.fec]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %C.fec = ref_binding v, %v.var [concrete = %v.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc4_10.2: %i32) {
// CHECK:STDOUT:   %N.loc4_10.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_10.1 (constants.%N.afc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.77d
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_18: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_1: %C.fec = converted %.loc13_18, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%v.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.afc) {
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%N.afc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_123.5a3) {
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_123.5a3
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
// CHECK:STDOUT:   %struct_type.a.b.602: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.f58: <witness> = complete_type_witness %struct_type.a.b.602 [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.b7e: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %F: %D = symbolic_binding F, 0 [symbolic]
// CHECK:STDOUT:   %E.type: type = generic_class_type @E [concrete]
// CHECK:STDOUT:   %E.generic: %E.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.0b4: type = class_type @E, @E(%F) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.4aa: %struct_type.a.b.cfd = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.33d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d5e: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.cc3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.4ff: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %D.val.e9d: %D = struct_value (%int_1.d5e, %int_2.4ff) [concrete]
// CHECK:STDOUT:   %.35b: ref %D = temporary invalid, %D.val.e9d [concrete]
// CHECK:STDOUT:   %E.231: type = class_type @E, @E(%D.val.e9d) [concrete]
// CHECK:STDOUT:   %pattern_type.f8c: type = pattern_type %E.231 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %struct.cb7: %struct_type.a.b.cfd = struct_value (%int_3.1ba, %int_4.0c1) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a7a: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.353: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.49a: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.30c: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.bef: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.a7d: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %D.val.2d7: %D = struct_value (%int_3.49a, %int_4.a7d) [concrete]
// CHECK:STDOUT:   %.7f3: ref %D = temporary invalid, %D.val.2d7 [concrete]
// CHECK:STDOUT:   %E.f0f: type = class_type @E, @E(%D.val.2d7) [concrete]
// CHECK:STDOUT:   %E.val: %E.f0f = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %F.patt: %pattern_type.b7e = symbolic_binding_pattern F, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9: type = splice_block %D.ref [concrete = constants.%D] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.loc9_10.2: %D = symbolic_binding F, 0 [symbolic = %F.loc9_10.1 (constants.%F)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %g.patt: %pattern_type.f8c = ref_binding_pattern g [concrete]
// CHECK:STDOUT:     %g.var_patt: %pattern_type.f8c = var_pattern %g.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %g.var: ref %E.231 = var %g.var_patt [concrete]
// CHECK:STDOUT:   %.loc18_26.1: type = splice_block %E [concrete = constants.%E.231] {
// CHECK:STDOUT:     %E.ref: %E.type = name_ref E, %E.decl [concrete = constants.%E.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc18_25.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2) [concrete = constants.%struct.4aa]
// CHECK:STDOUT:     %impl.elem0.loc18_25.1: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:     %bound_method.loc18_25.1: <bound method> = bound_method %int_1, %impl.elem0.loc18_25.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d]
// CHECK:STDOUT:     %specific_fn.loc18_25.1: <specific function> = specific_function %impl.elem0.loc18_25.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc18_25.2: <bound method> = bound_method %int_1, %specific_fn.loc18_25.1 [concrete = constants.%bound_method.33d]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_25.1: init %i32 = call %bound_method.loc18_25.2(%int_1) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:     %.loc18_25.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_25.1 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:     %.loc18_25.3: ref %D = temporary_storage
// CHECK:STDOUT:     %.loc18_25.4: ref %i32 = class_element_access %.loc18_25.3, element0
// CHECK:STDOUT:     %.loc18_25.5: init %i32 to %.loc18_25.4 = in_place_init %.loc18_25.2 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:     %impl.elem0.loc18_25.2: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:     %bound_method.loc18_25.3: <bound method> = bound_method %int_2, %impl.elem0.loc18_25.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e]
// CHECK:STDOUT:     %specific_fn.loc18_25.2: <specific function> = specific_function %impl.elem0.loc18_25.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc18_25.4: <bound method> = bound_method %int_2, %specific_fn.loc18_25.2 [concrete = constants.%bound_method.cc3]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_25.2: init %i32 = call %bound_method.loc18_25.4(%int_2) [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:     %.loc18_25.6: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_25.2 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:     %.loc18_25.7: ref %i32 = class_element_access %.loc18_25.3, element1
// CHECK:STDOUT:     %.loc18_25.8: init %i32 to %.loc18_25.7 = in_place_init %.loc18_25.6 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:     %.loc18_25.9: init %D to %.loc18_25.3 = class_init (%.loc18_25.5, %.loc18_25.8) [concrete = constants.%D.val.e9d]
// CHECK:STDOUT:     %.loc18_26.2: init %D = converted %.loc18_25.1, %.loc18_25.9 [concrete = constants.%D.val.e9d]
// CHECK:STDOUT:     %.loc18_26.3: ref %D = temporary %.loc18_25.3, %.loc18_26.2 [concrete = constants.%.35b]
// CHECK:STDOUT:     %.loc18_26.4: %D = acquire_value %.loc18_26.3 [concrete = constants.%D.val.e9d]
// CHECK:STDOUT:     %E: type = class_type @E, @E(constants.%D.val.e9d) [concrete = constants.%E.231]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %g: ref %E.231 = ref_binding g, %g.var [concrete = %g.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc5: %D.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %.loc6: %D.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b.602 [concrete = constants.%complete_type.f58]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .a = %.loc5
// CHECK:STDOUT:   .b = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @E(%F.loc9_10.2: %D) {
// CHECK:STDOUT:   %F.loc9_10.1: %D = symbolic_binding F, 0 [symbolic = %F.loc9_10.1 (constants.%F)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%E.0b4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_31.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, file.%E.decl [concrete = constants.%E.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %.loc18_53.1: %struct_type.a.b.cfd = struct_literal (%int_3, %int_4) [concrete = constants.%struct.cb7]
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %impl.elem0.loc18_53.1: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc18_53.1: <bound method> = bound_method %int_3, %impl.elem0.loc18_53.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a7a]
// CHECK:STDOUT:   %specific_fn.loc18_53.1: <specific function> = specific_function %impl.elem0.loc18_53.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_53.2: <bound method> = bound_method %int_3, %specific_fn.loc18_53.1 [concrete = constants.%bound_method.353]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_53.1: init %i32 = call %bound_method.loc18_53.2(%int_3) [concrete = constants.%int_3.49a]
// CHECK:STDOUT:   %.loc18_53.2: init %i32 = converted %int_3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_53.1 [concrete = constants.%int_3.49a]
// CHECK:STDOUT:   %.loc18_53.3: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc18_53.4: ref %i32 = class_element_access %.loc18_53.3, element0
// CHECK:STDOUT:   %.loc18_53.5: init %i32 to %.loc18_53.4 = in_place_init %.loc18_53.2 [concrete = constants.%int_3.49a]
// CHECK:STDOUT:   %impl.elem0.loc18_53.2: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc18_53.3: <bound method> = bound_method %int_4, %impl.elem0.loc18_53.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.30c]
// CHECK:STDOUT:   %specific_fn.loc18_53.2: <specific function> = specific_function %impl.elem0.loc18_53.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_53.4: <bound method> = bound_method %int_4, %specific_fn.loc18_53.2 [concrete = constants.%bound_method.bef]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_53.2: init %i32 = call %bound_method.loc18_53.4(%int_4) [concrete = constants.%int_4.a7d]
// CHECK:STDOUT:   %.loc18_53.6: init %i32 = converted %int_4, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_53.2 [concrete = constants.%int_4.a7d]
// CHECK:STDOUT:   %.loc18_53.7: ref %i32 = class_element_access %.loc18_53.3, element1
// CHECK:STDOUT:   %.loc18_53.8: init %i32 to %.loc18_53.7 = in_place_init %.loc18_53.6 [concrete = constants.%int_4.a7d]
// CHECK:STDOUT:   %.loc18_53.9: init %D to %.loc18_53.3 = class_init (%.loc18_53.5, %.loc18_53.8) [concrete = constants.%D.val.2d7]
// CHECK:STDOUT:   %.loc18_55.1: init %D = converted %.loc18_53.1, %.loc18_53.9 [concrete = constants.%D.val.2d7]
// CHECK:STDOUT:   %.loc18_55.2: ref %D = temporary %.loc18_53.3, %.loc18_55.1 [concrete = constants.%.7f3]
// CHECK:STDOUT:   %.loc18_55.3: %D = acquire_value %.loc18_55.2 [concrete = constants.%D.val.2d7]
// CHECK:STDOUT:   %E: type = class_type @E, @E(constants.%D.val.2d7) [concrete = constants.%E.f0f]
// CHECK:STDOUT:   %.loc18_31.2: ref %E.f0f = temporary_storage
// CHECK:STDOUT:   %.loc18_31.3: init %E.f0f to %.loc18_31.2 = class_init () [concrete = constants.%E.val]
// CHECK:STDOUT:   %.loc18_33: init %E.f0f = converted %.loc18_31.1, %.loc18_31.3 [concrete = constants.%E.val]
// CHECK:STDOUT:   %.loc18_1: %E.231 = converted %.loc18_33, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%g.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E(constants.%F) {
// CHECK:STDOUT:   %F.loc9_10.1 => constants.%F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E(constants.%D.val.e9d) {
// CHECK:STDOUT:   %F.loc9_10.1 => constants.%D.val.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E(constants.%D.val.2d7) {
// CHECK:STDOUT:   %F.loc9_10.1 => constants.%D.val.2d7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
