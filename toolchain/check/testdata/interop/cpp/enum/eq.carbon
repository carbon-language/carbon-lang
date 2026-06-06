// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/enum/eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/enum/eq.carbon

// --- enum.h

enum Unscoped { A, B };
enum class Scoped { X, Y };

// --- overloaded.h

enum CustomEnum { X, Y };
auto operator==(CustomEnum lhs, CustomEnum rhs) -> bool;
auto operator!=(CustomEnum lhs, CustomEnum rhs) -> bool;

enum HeteroUnscoped { A, B };
enum class HeteroScoped { X, Y };
auto operator==(HeteroScoped lhs, HeteroUnscoped rhs) -> bool;
auto operator!=(HeteroScoped lhs, HeteroUnscoped rhs) -> bool;

// --- eq.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum.h";

//@dump-sem-ir-begin
fn CompareGeneric[U:! type, T:! Core.EqWith(U)](x: T, y: U) -> bool {
  return x == y;
}
fn CompareUnscoped(x: Cpp.Unscoped, y: Cpp.Unscoped) -> bool {
  return x == y;
}

fn CompareScoped(x: Cpp.Scoped, y: Cpp.Scoped) -> bool {
  return x == y;
}

fn CallCompareGeneric(x: Cpp.Unscoped, y: Cpp.Unscoped) -> bool {
  return CompareGeneric(x, y);
}
//@dump-sem-ir-end

// --- overloaded_op.carbon

library "[[@TEST_NAME]]";

import Cpp library "overloaded.h";

fn CompareGeneric[U:! type, T:! Core.EqWith(U)](x: T, y: U) -> bool {
  return x == y;
}

//@dump-sem-ir-begin
fn CompareCustom(x: Cpp.CustomEnum, y: Cpp.CustomEnum) -> bool {
  return x == y;
}

fn CompareHetero(x: Cpp.HeteroScoped, y: Cpp.HeteroUnscoped) -> bool {
  return CompareGeneric(x, y);
}
//@dump-sem-ir-end

// --- fail_hetero.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum.h";

fn CompareGeneric[U:! type, T:! Core.EqWith(U)](x: T, y: U) -> bool {
  return x == y;
}

fn CompareHeteroFail(x: Cpp.Scoped, y: Cpp.Unscoped) -> bool {
  // CHECK:STDERR: fail_hetero.carbon:[[@LINE+7]]:10: error: cannot convert type `Cpp.Scoped` into type implementing `Core.EqWith(Cpp.Unscoped)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   return CompareGeneric(x, y);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_hetero.carbon:[[@LINE-8]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CompareGeneric[U:! type, T:! Core.EqWith(U)](x: T, y: U) -> bool {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return CompareGeneric(x, y);
}

// CHECK:STDOUT: --- eq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %EqWith.type.f77: type = generic_interface_type @EqWith [concrete]
// CHECK:STDOUT:   %EqWith.generic: %EqWith.type.f77 = struct_value () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %EqWith.type.98ea86.1: type = facet_type <@EqWith, @EqWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.d94ae0.1: %EqWith.type.98ea86.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.85b122.1: type = assoc_entity_type @EqWith, @EqWith(%Other) [symbolic]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %.f34: Core.Form = init_form bool [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b82ec7.1: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Other, %Self.d94ae0.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.ae3c3e.1: %EqWith.WithSelf.Equal.type.b82ec7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.98ea86.2: type = facet_type <@EqWith, @EqWith(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.3d4: type = pattern_type %EqWith.type.98ea86.2 [symbolic]
// CHECK:STDOUT:   %T: %EqWith.type.98ea86.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.85b122.2: type = assoc_entity_type @EqWith, @EqWith(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.eee873.2: %EqWith.assoc_type.85b122.2 = assoc_entity element0, imports.%Core.import_ref.5ba [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.e6374a.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %CompareGeneric.type: type = fn_type @CompareGeneric [concrete]
// CHECK:STDOUT:   %CompareGeneric: %CompareGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.7f1: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.fb5: <witness> = require_complete_type %EqWith.type.98ea86.2 [symbolic]
// CHECK:STDOUT:   %assoc0.d00: %EqWith.assoc_type.85b122.1 = assoc_entity element0, imports.%Core.import_ref.2e5 [symbolic]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @EqWith, @EqWith(%U) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b82ec7.3: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.74c: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.b82ec7.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.74c = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @EqWith.WithSelf.Equal(%U, %T) [symbolic]
// CHECK:STDOUT:   %Unscoped: type = class_type @Unscoped [concrete]
// CHECK:STDOUT:   %pattern_type.812: type = pattern_type %Unscoped [concrete]
// CHECK:STDOUT:   %CompareUnscoped.type: type = fn_type @CompareUnscoped [concrete]
// CHECK:STDOUT:   %CompareUnscoped: %CompareUnscoped.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %u32.builtin: type = int_type unsigned, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.8a9: <witness> = complete_type_witness %u32.builtin [concrete]
// CHECK:STDOUT:   %Equal.type.eace26.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.8c4faf.1: %Equal.type.eace26.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Scoped: type = class_type @Scoped [concrete]
// CHECK:STDOUT:   %pattern_type.5f2: type = pattern_type %Scoped [concrete]
// CHECK:STDOUT:   %CompareScoped.type: type = fn_type @CompareScoped [concrete]
// CHECK:STDOUT:   %CompareScoped: %CompareScoped.type = struct_value () [concrete]
// CHECK:STDOUT:   %Equal.type.eace26.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.8c4faf.2: %Equal.type.eace26.2 = struct_value () [concrete]
// CHECK:STDOUT:   %CallCompareGeneric.type: type = fn_type @CallCompareGeneric [concrete]
// CHECK:STDOUT:   %CallCompareGeneric: %CallCompareGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.type.ed9: type = facet_type <@EqWith, @EqWith(%Unscoped)> [concrete]
// CHECK:STDOUT:   %Equal.type.eace26.3: type = fn_type @Equal.3 [concrete]
// CHECK:STDOUT:   %Equal.8c4faf.3: %Equal.type.eace26.3 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.b47: type = assoc_entity_type @EqWith, @EqWith(%Unscoped) [concrete]
// CHECK:STDOUT:   %assoc0.9e5: %EqWith.assoc_type.b47 = assoc_entity element0, imports.%Core.import_ref.5ba [concrete]
// CHECK:STDOUT:   %custom_witness.ce6: <witness> = custom_witness (%Equal.8c4faf.3, %NotEqual), @EqWith, @EqWith(%Unscoped) [concrete]
// CHECK:STDOUT:   %EqWith.facet.500: %EqWith.type.ed9 = facet_value %Unscoped, (%custom_witness.ce6) [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %EqWith.type.ed9 [concrete]
// CHECK:STDOUT:   %CompareGeneric.specific_fn: <specific function> = specific_function %CompareGeneric, @CompareGeneric(%Unscoped, %EqWith.facet.500) [concrete]
// CHECK:STDOUT:   %complete_type.0e2: <witness> = complete_type_witness %EqWith.type.ed9 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.d93: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Unscoped, %EqWith.facet.500) [concrete]
// CHECK:STDOUT:   %.43c: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.d93, %EqWith.facet.500 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .EqWith = %Core.EqWith
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Unscoped = %Unscoped.decl
// CHECK:STDOUT:     .Scoped = %Scoped.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.EqWith: %EqWith.type.f77 = import_ref Core//prelude/operators/comparison, EqWith, loaded [concrete = constants.%EqWith.generic]
// CHECK:STDOUT:   %Core.import_ref.725: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.85b122.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc0 (constants.%assoc0.d00)]
// CHECK:STDOUT:   %Core.import_ref.5ba: @EqWith.WithSelf.%EqWith.WithSelf.Equal.type (%EqWith.WithSelf.Equal.type.b82ec7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.Equal (constants.%EqWith.WithSelf.Equal.ae3c3e.1)]
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.import_ref.2e5 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Unscoped.decl: type = class_decl @Unscoped [concrete = constants.%Unscoped] {} {}
// CHECK:STDOUT:   %Scoped.decl: type = class_decl @Scoped [concrete = constants.%Scoped] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Equal.decl.c9e403.1: %Equal.type.eace26.1 = fn_decl @Equal.1 [concrete = constants.%Equal.8c4faf.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Equal.decl.c9e403.2: %Equal.type.eace26.2 = fn_decl @Equal.2 [concrete = constants.%Equal.8c4faf.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %CompareGeneric.decl: %CompareGeneric.type = fn_decl @CompareGeneric [concrete = constants.%CompareGeneric] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %T.patt: @CompareGeneric.%pattern_type.loc7_30 (%pattern_type.3d4) = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     %x.param_patt: @CompareGeneric.%pattern_type.loc7_50 (%pattern_type.e6374a.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @CompareGeneric.%pattern_type.loc7_50 (%pattern_type.e6374a.2) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %y.param_patt: @CompareGeneric.%pattern_type.loc7_56 (%pattern_type.51d1c4.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %y.patt: @CompareGeneric.%pattern_type.loc7_56 (%pattern_type.51d1c4.2) = at_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc7_64.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_64.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc7_64.2: Core.Form = init_form %.loc7_64.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %.loc7_23.1: type = splice_block %.loc7_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc7_20: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_20.2: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.1 (constants.%U)]
// CHECK:STDOUT:     %.loc7_46: type = splice_block %EqWith.type.loc7_46.2 [symbolic = %EqWith.type.loc7_46.1 (constants.%EqWith.type.98ea86.2)] {
// CHECK:STDOUT:       %.Self.loc7_30: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %EqWith.ref: %EqWith.type.f77 = name_ref EqWith, imports.%Core.EqWith [concrete = constants.%EqWith.generic]
// CHECK:STDOUT:       %U.ref.loc7_45: type = name_ref U, %U.loc7_20.2 [symbolic = %U.loc7_20.1 (constants.%U)]
// CHECK:STDOUT:       %EqWith.type.loc7_46.2: type = facet_type <@EqWith, @EqWith(constants.%U)> [symbolic = %EqWith.type.loc7_46.1 (constants.%EqWith.type.98ea86.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_30.2: @CompareGeneric.%EqWith.type.loc7_46.1 (%EqWith.type.98ea86.2) = symbolic_binding T, 1 [symbolic = %T.loc7_30.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @CompareGeneric.%T.as_type.loc7_52.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_52.1: type = splice_block %.loc7_52.2 [symbolic = %T.as_type.loc7_52.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: @CompareGeneric.%EqWith.type.loc7_46.1 (%EqWith.type.98ea86.2) = name_ref T, %T.loc7_30.2 [symbolic = %T.loc7_30.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc7_52.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_52.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_52.2: type = converted %T.ref, %T.as_type.loc7_52.2 [symbolic = %T.as_type.loc7_52.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @CompareGeneric.%T.as_type.loc7_52.1 (%T.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %y.param: @CompareGeneric.%U.loc7_20.1 (%U) = value_param call_param1
// CHECK:STDOUT:     %U.ref.loc7_58: type = name_ref U, %U.loc7_20.2 [symbolic = %U.loc7_20.1 (constants.%U)]
// CHECK:STDOUT:     %y: @CompareGeneric.%U.loc7_20.1 (%U) = value_binding y, %y.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CompareUnscoped.decl: %CompareUnscoped.type = fn_decl @CompareUnscoped [concrete = constants.%CompareUnscoped] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.812 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.812 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.812 = value_param_pattern [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.812 = at_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc10_57.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_57.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc10_57.2: Core.Form = init_form %.loc10_57.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %x.param: %Unscoped = value_param call_param0
// CHECK:STDOUT:     %.loc10_26: type = splice_block %Unscoped.ref.loc10_26 [concrete = constants.%Unscoped] {
// CHECK:STDOUT:       %Cpp.ref.loc10_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Unscoped.ref.loc10_26: type = name_ref Unscoped, imports.%Unscoped.decl [concrete = constants.%Unscoped]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %Unscoped = value_binding x, %x.param
// CHECK:STDOUT:     %y.param: %Unscoped = value_param call_param1
// CHECK:STDOUT:     %.loc10_43: type = splice_block %Unscoped.ref.loc10_43 [concrete = constants.%Unscoped] {
// CHECK:STDOUT:       %Cpp.ref.loc10_40: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Unscoped.ref.loc10_43: type = name_ref Unscoped, imports.%Unscoped.decl [concrete = constants.%Unscoped]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: %Unscoped = value_binding y, %y.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CompareScoped.decl: %CompareScoped.type = fn_decl @CompareScoped [concrete = constants.%CompareScoped] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.5f2 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.5f2 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.5f2 = value_param_pattern [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.5f2 = at_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc14_51.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_51.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc14_51.2: Core.Form = init_form %.loc14_51.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %x.param: %Scoped = value_param call_param0
// CHECK:STDOUT:     %.loc14_24: type = splice_block %Scoped.ref.loc14_24 [concrete = constants.%Scoped] {
// CHECK:STDOUT:       %Cpp.ref.loc14_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Scoped.ref.loc14_24: type = name_ref Scoped, imports.%Scoped.decl [concrete = constants.%Scoped]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %Scoped = value_binding x, %x.param
// CHECK:STDOUT:     %y.param: %Scoped = value_param call_param1
// CHECK:STDOUT:     %.loc14_39: type = splice_block %Scoped.ref.loc14_39 [concrete = constants.%Scoped] {
// CHECK:STDOUT:       %Cpp.ref.loc14_36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Scoped.ref.loc14_39: type = name_ref Scoped, imports.%Scoped.decl [concrete = constants.%Scoped]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: %Scoped = value_binding y, %y.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallCompareGeneric.decl: %CallCompareGeneric.type = fn_decl @CallCompareGeneric [concrete = constants.%CallCompareGeneric] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.812 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.812 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.812 = value_param_pattern [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.812 = at_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc18_60.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_60.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc18_60.2: Core.Form = init_form %.loc18_60.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %x.param: %Unscoped = value_param call_param0
// CHECK:STDOUT:     %.loc18_29: type = splice_block %Unscoped.ref.loc18_29 [concrete = constants.%Unscoped] {
// CHECK:STDOUT:       %Cpp.ref.loc18_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Unscoped.ref.loc18_29: type = name_ref Unscoped, imports.%Unscoped.decl [concrete = constants.%Unscoped]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %Unscoped = value_binding x, %x.param
// CHECK:STDOUT:     %y.param: %Unscoped = value_param call_param1
// CHECK:STDOUT:     %.loc18_46: type = splice_block %Unscoped.ref.loc18_46 [concrete = constants.%Unscoped] {
// CHECK:STDOUT:       %Cpp.ref.loc18_43: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Unscoped.ref.loc18_46: type = name_ref Unscoped, imports.%Unscoped.decl [concrete = constants.%Unscoped]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: %Unscoped = value_binding y, %y.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompareGeneric(%U.loc7_20.2: type, %T.loc7_30.2: @CompareGeneric.%EqWith.type.loc7_46.1 (%EqWith.type.98ea86.2)) {
// CHECK:STDOUT:   %U.loc7_20.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.1 (constants.%U)]
// CHECK:STDOUT:   %EqWith.type.loc7_46.1: type = facet_type <@EqWith, @EqWith(%U.loc7_20.1)> [symbolic = %EqWith.type.loc7_46.1 (constants.%EqWith.type.98ea86.2)]
// CHECK:STDOUT:   %T.loc7_30.1: @CompareGeneric.%EqWith.type.loc7_46.1 (%EqWith.type.98ea86.2) = symbolic_binding T, 1 [symbolic = %T.loc7_30.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type.loc7_30: type = pattern_type %EqWith.type.loc7_46.1 [symbolic = %pattern_type.loc7_30 (constants.%pattern_type.3d4)]
// CHECK:STDOUT:   %T.as_type.loc7_52.1: type = facet_access_type %T.loc7_30.1 [symbolic = %T.as_type.loc7_52.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc7_50: type = pattern_type %T.as_type.loc7_52.1 [symbolic = %pattern_type.loc7_50 (constants.%pattern_type.e6374a.2)]
// CHECK:STDOUT:   %pattern_type.loc7_56: type = pattern_type %U.loc7_20.1 [symbolic = %pattern_type.loc7_56 (constants.%pattern_type.51d1c4.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_50: <witness> = require_complete_type %T.as_type.loc7_52.1 [symbolic = %require_complete.loc7_50 (constants.%require_complete.7f1)]
// CHECK:STDOUT:   %require_complete.loc7_56: <witness> = require_complete_type %U.loc7_20.1 [symbolic = %require_complete.loc7_56 (constants.%require_complete.944)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %EqWith.type.loc7_46.1 [symbolic = %require_complete.loc8 (constants.%require_complete.fb5)]
// CHECK:STDOUT:   %EqWith.assoc_type: type = assoc_entity_type @EqWith, @EqWith(%U.loc7_20.1) [symbolic = %EqWith.assoc_type (constants.%EqWith.assoc_type.85b122.2)]
// CHECK:STDOUT:   %assoc0: @CompareGeneric.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = assoc_entity element0, imports.%Core.import_ref.5ba [symbolic = %assoc0 (constants.%assoc0.eee873.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U.loc7_20.1, %T.loc7_30.1) [symbolic = %EqWith.WithSelf.Equal.type (constants.%EqWith.WithSelf.Equal.type.b82ec7.3)]
// CHECK:STDOUT:   %.loc8_12.2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type, %T.loc7_30.1 [symbolic = %.loc8_12.2 (constants.%.74c)]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc7_30.1, @EqWith, @EqWith(%U.loc7_20.1) [symbolic = %EqWith.lookup_impl_witness (constants.%EqWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc8_12.2: @CompareGeneric.%.loc8_12.2 (%.74c) = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_12.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_12.2: <specific function> = specific_impl_function %impl.elem0.loc8_12.2, @EqWith.WithSelf.Equal(%U.loc7_20.1, %T.loc7_30.1) [symbolic = %specific_impl_fn.loc8_12.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CompareGeneric.%T.as_type.loc7_52.1 (%T.as_type), %y.param: @CompareGeneric.%U.loc7_20.1 (%U)) -> out %return.param: bool {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @CompareGeneric.%T.as_type.loc7_52.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @CompareGeneric.%U.loc7_20.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc8: type = facet_type <@EqWith, @EqWith(constants.%U)> [symbolic = %EqWith.type.loc7_46.1 (constants.%EqWith.type.98ea86.2)]
// CHECK:STDOUT:     %.loc8_12.1: @CompareGeneric.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = specific_constant imports.%Core.import_ref.725, @EqWith.WithSelf(constants.%U, constants.%Self.d94ae0.1) [symbolic = %assoc0 (constants.%assoc0.eee873.2)]
// CHECK:STDOUT:     %Equal.ref: @CompareGeneric.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = name_ref Equal, %.loc8_12.1 [symbolic = %assoc0 (constants.%assoc0.eee873.2)]
// CHECK:STDOUT:     %impl.elem0.loc8_12.1: @CompareGeneric.%.loc8_12.2 (%.74c) = impl_witness_access constants.%EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_12.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc8_12.1: <bound method> = bound_method %x.ref, %impl.elem0.loc8_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc8_12.1: <specific function> = specific_impl_function %impl.elem0.loc8_12.1, @EqWith.WithSelf.Equal(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc8_12.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc8_12.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc8_12.1
// CHECK:STDOUT:     %EqWith.WithSelf.Equal.call: init bool = call %bound_method.loc8_12.2(%x.ref, %y.ref)
// CHECK:STDOUT:     return %EqWith.WithSelf.Equal.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompareUnscoped(%x.param: %Unscoped, %y.param: %Unscoped) -> out %return.param: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Unscoped = name_ref x, %x
// CHECK:STDOUT:   %y.ref: %Unscoped = name_ref y, %y
// CHECK:STDOUT:   %Equal.call: init bool = call generated.%Equal.decl.c9e403.1(%x.ref, %y.ref)
// CHECK:STDOUT:   return %Equal.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompareScoped(%x.param: %Scoped, %y.param: %Scoped) -> out %return.param: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Scoped = name_ref x, %x
// CHECK:STDOUT:   %y.ref: %Scoped = name_ref y, %y
// CHECK:STDOUT:   %Equal.call: init bool = call generated.%Equal.decl.c9e403.2(%x.ref, %y.ref)
// CHECK:STDOUT:   return %Equal.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallCompareGeneric(%x.param: %Unscoped, %y.param: %Unscoped) -> out %return.param: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CompareGeneric.ref: %CompareGeneric.type = name_ref CompareGeneric, file.%CompareGeneric.decl [concrete = constants.%CompareGeneric]
// CHECK:STDOUT:   %x.ref: %Unscoped = name_ref x, %x
// CHECK:STDOUT:   %y.ref: %Unscoped = name_ref y, %y
// CHECK:STDOUT:   %EqWith.facet: %EqWith.type.ed9 = facet_value constants.%Unscoped, (constants.%custom_witness.ce6) [concrete = constants.%EqWith.facet.500]
// CHECK:STDOUT:   %.loc19: %EqWith.type.ed9 = converted constants.%Unscoped, %EqWith.facet [concrete = constants.%EqWith.facet.500]
// CHECK:STDOUT:   %CompareGeneric.specific_fn: <specific function> = specific_function %CompareGeneric.ref, @CompareGeneric(constants.%Unscoped, constants.%EqWith.facet.500) [concrete = constants.%CompareGeneric.specific_fn]
// CHECK:STDOUT:   %CompareGeneric.call: init bool = call %CompareGeneric.specific_fn(%x.ref, %y.ref)
// CHECK:STDOUT:   return %CompareGeneric.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompareGeneric(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc7_20.1 => constants.%U
// CHECK:STDOUT:   %EqWith.type.loc7_46.1 => constants.%EqWith.type.98ea86.2
// CHECK:STDOUT:   %T.loc7_30.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc7_30 => constants.%pattern_type.3d4
// CHECK:STDOUT:   %T.as_type.loc7_52.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc7_50 => constants.%pattern_type.e6374a.2
// CHECK:STDOUT:   %pattern_type.loc7_56 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompareGeneric(constants.%Unscoped, constants.%EqWith.facet.500) {
// CHECK:STDOUT:   %U.loc7_20.1 => constants.%Unscoped
// CHECK:STDOUT:   %EqWith.type.loc7_46.1 => constants.%EqWith.type.ed9
// CHECK:STDOUT:   %T.loc7_30.1 => constants.%EqWith.facet.500
// CHECK:STDOUT:   %pattern_type.loc7_30 => constants.%pattern_type.fe8
// CHECK:STDOUT:   %T.as_type.loc7_52.1 => constants.%Unscoped
// CHECK:STDOUT:   %pattern_type.loc7_50 => constants.%pattern_type.812
// CHECK:STDOUT:   %pattern_type.loc7_56 => constants.%pattern_type.812
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_50 => constants.%complete_type.8a9
// CHECK:STDOUT:   %require_complete.loc7_56 => constants.%complete_type.8a9
// CHECK:STDOUT:   %require_complete.loc8 => constants.%complete_type.0e2
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.b47
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9e5
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.d93
// CHECK:STDOUT:   %.loc8_12.2 => constants.%.43c
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.ce6
// CHECK:STDOUT:   %impl.elem0.loc8_12.2 => constants.%Equal.8c4faf.3
// CHECK:STDOUT:   %specific_impl_fn.loc8_12.2 => constants.%Equal.8c4faf.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overloaded_op.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %.f34: Core.Form = init_form bool [concrete]
// CHECK:STDOUT:   %CompareGeneric.type: type = fn_type @CompareGeneric [concrete]
// CHECK:STDOUT:   %CompareGeneric: %CompareGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %CustomEnum: type = class_type @CustomEnum [concrete]
// CHECK:STDOUT:   %pattern_type.031: type = pattern_type %CustomEnum [concrete]
// CHECK:STDOUT:   %CompareCustom.type: type = fn_type @CompareCustom [concrete]
// CHECK:STDOUT:   %CompareCustom: %CompareCustom.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.type.d58437.1: type = fn_type @operator_EqualEqual__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.5da7db.1: %operator_EqualEqual__carbon_thunk.type.d58437.1 = struct_value () [concrete]
// CHECK:STDOUT:   %HeteroScoped: type = class_type @HeteroScoped [concrete]
// CHECK:STDOUT:   %pattern_type.55e: type = pattern_type %HeteroScoped [concrete]
// CHECK:STDOUT:   %HeteroUnscoped: type = class_type @HeteroUnscoped [concrete]
// CHECK:STDOUT:   %pattern_type.dac: type = pattern_type %HeteroUnscoped [concrete]
// CHECK:STDOUT:   %CompareHetero.type: type = fn_type @CompareHetero [concrete]
// CHECK:STDOUT:   %CompareHetero: %CompareHetero.type = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.type.553: type = facet_type <@EqWith, @EqWith(%HeteroUnscoped)> [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8de: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%HeteroUnscoped) [concrete]
// CHECK:STDOUT:   %EqWith.facet.356: %EqWith.type.553 = facet_value %HeteroScoped, (%custom_witness.8de) [concrete]
// CHECK:STDOUT:   %CompareGeneric.specific_fn: <specific function> = specific_function %CompareGeneric, @CompareGeneric(%HeteroUnscoped, %EqWith.facet.356) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .CustomEnum = %CustomEnum.decl
// CHECK:STDOUT:     .HeteroScoped = %HeteroScoped.decl
// CHECK:STDOUT:     .HeteroUnscoped = %HeteroUnscoped.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CustomEnum.decl: type = class_decl @CustomEnum [concrete = constants.%CustomEnum] {} {}
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.decl.88153b.1: %operator_EqualEqual__carbon_thunk.type.d58437.1 = fn_decl @operator_EqualEqual__carbon_thunk.1 [concrete = constants.%operator_EqualEqual__carbon_thunk.5da7db.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HeteroScoped.decl: type = class_decl @HeteroScoped [concrete = constants.%HeteroScoped] {} {}
// CHECK:STDOUT:   %HeteroUnscoped.decl: type = class_decl @HeteroUnscoped [concrete = constants.%HeteroUnscoped] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %CompareCustom.decl: %CompareCustom.type = fn_decl @CompareCustom [concrete = constants.%CompareCustom] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.031 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.031 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.031 = value_param_pattern [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.031 = at_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc11_59.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_59.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc11_59.2: Core.Form = init_form %.loc11_59.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %x.param: %CustomEnum = value_param call_param0
// CHECK:STDOUT:     %.loc11_24: type = splice_block %CustomEnum.ref.loc11_24 [concrete = constants.%CustomEnum] {
// CHECK:STDOUT:       %Cpp.ref.loc11_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %CustomEnum.ref.loc11_24: type = name_ref CustomEnum, imports.%CustomEnum.decl [concrete = constants.%CustomEnum]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %CustomEnum = value_binding x, %x.param
// CHECK:STDOUT:     %y.param: %CustomEnum = value_param call_param1
// CHECK:STDOUT:     %.loc11_43: type = splice_block %CustomEnum.ref.loc11_43 [concrete = constants.%CustomEnum] {
// CHECK:STDOUT:       %Cpp.ref.loc11_40: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %CustomEnum.ref.loc11_43: type = name_ref CustomEnum, imports.%CustomEnum.decl [concrete = constants.%CustomEnum]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: %CustomEnum = value_binding y, %y.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CompareHetero.decl: %CompareHetero.type = fn_decl @CompareHetero [concrete = constants.%CompareHetero] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.55e = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.55e = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.dac = value_param_pattern [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.dac = at_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc15_65.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_65.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc15_65.2: Core.Form = init_form %.loc15_65.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %x.param: %HeteroScoped = value_param call_param0
// CHECK:STDOUT:     %.loc15_24: type = splice_block %HeteroScoped.ref [concrete = constants.%HeteroScoped] {
// CHECK:STDOUT:       %Cpp.ref.loc15_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %HeteroScoped.ref: type = name_ref HeteroScoped, imports.%HeteroScoped.decl [concrete = constants.%HeteroScoped]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %HeteroScoped = value_binding x, %x.param
// CHECK:STDOUT:     %y.param: %HeteroUnscoped = value_param call_param1
// CHECK:STDOUT:     %.loc15_45: type = splice_block %HeteroUnscoped.ref [concrete = constants.%HeteroUnscoped] {
// CHECK:STDOUT:       %Cpp.ref.loc15_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %HeteroUnscoped.ref: type = name_ref HeteroUnscoped, imports.%HeteroUnscoped.decl [concrete = constants.%HeteroUnscoped]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: %HeteroUnscoped = value_binding y, %y.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompareCustom(%x.param: %CustomEnum, %y.param: %CustomEnum) -> out %return.param: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %CustomEnum = name_ref x, %x
// CHECK:STDOUT:   %y.ref: %CustomEnum = name_ref y, %y
// CHECK:STDOUT:   %.loc12_12.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc12_12.1
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_EqualEqual__carbon_thunk.decl.88153b.1(%x.ref, %y.ref, %addr)
// CHECK:STDOUT:   %.loc12_12.2: init bool to %.loc12_12.1 = mark_in_place_init %operator_EqualEqual__carbon_thunk.call
// CHECK:STDOUT:   return %.loc12_12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompareHetero(%x.param: %HeteroScoped, %y.param: %HeteroUnscoped) -> out %return.param: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CompareGeneric.ref: %CompareGeneric.type = name_ref CompareGeneric, file.%CompareGeneric.decl [concrete = constants.%CompareGeneric]
// CHECK:STDOUT:   %x.ref: %HeteroScoped = name_ref x, %x
// CHECK:STDOUT:   %y.ref: %HeteroUnscoped = name_ref y, %y
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %EqWith.facet: %EqWith.type.553 = facet_value constants.%HeteroScoped, (constants.%custom_witness.8de) [concrete = constants.%EqWith.facet.356]
// CHECK:STDOUT:   %.loc16: %EqWith.type.553 = converted constants.%HeteroScoped, %EqWith.facet [concrete = constants.%EqWith.facet.356]
// CHECK:STDOUT:   %CompareGeneric.specific_fn: <specific function> = specific_function %CompareGeneric.ref, @CompareGeneric(constants.%HeteroUnscoped, constants.%EqWith.facet.356) [concrete = constants.%CompareGeneric.specific_fn]
// CHECK:STDOUT:   %CompareGeneric.call: init bool = call %CompareGeneric.specific_fn(%x.ref, %y.ref)
// CHECK:STDOUT:   return %CompareGeneric.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
