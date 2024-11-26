// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/tuple.carbon

// --- tuple_type.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}

fn F[T:! type, U:! type](pair: (T, U)) -> U { return F(pair); }

fn G(pair: (C, D)) -> D {
  return F(pair);
}

// --- tuple_value.carbon

library "[[@TEST_NAME]]";

class HasPair(Pair:! (i32, i32)) {}

fn F[A:! i32, B:! i32](h: HasPair((A, B))) -> i32 { return B; }

fn G(h: HasPair((1, 2))) -> i32 {
  return F(h);
}

// --- fail_inconsistent.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}

fn F[T:! type](pair: (T, T)) -> T;

fn G(pair: (C, D)) -> D {
  // CHECK:STDERR: fail_inconsistent.carbon:[[@LINE+6]]:10: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR:   return F(pair);
  // CHECK:STDERR:          ^~
  // CHECK:STDERR: fail_inconsistent.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[T:! type](pair: (T, T)) -> T;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return F(pair);
}

// CHECK:STDOUT: --- tuple_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: <specific function> = specific_function %F, @F(%T, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%C, %D) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.7: <specific function> = specific_function %F, @F(%C, %D) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc7_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc7_6.1, runtime_param<invalid> [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc7_16.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc7_16.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc7_16.1, runtime_param<invalid> [symbolic = %U.patt.loc7_16.2 (constants.%U.patt)]
// CHECK:STDOUT:     %pair.patt: @F.%tuple.type (%tuple.type.2) = binding_pattern pair
// CHECK:STDOUT:     %pair.param_patt: @F.%tuple.type (%tuple.type.2) = value_param_pattern %pair.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%U.loc7_16.2 (%U) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%U.loc7_16.2 (%U) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc7_36: type = name_ref U, %U.loc7_16.1 [symbolic = %U.loc7_16.2 (constants.%U)]
// CHECK:STDOUT:     %.loc7_37.1: %tuple.type.1 = tuple_literal (%T.ref, %U.ref.loc7_36)
// CHECK:STDOUT:     %.loc7_37.2: type = converted %.loc7_37.1, constants.%tuple.type.2 [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:     %U.ref.loc7_43: type = name_ref U, %U.loc7_16.1 [symbolic = %U.loc7_16.2 (constants.%U)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc7_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc7_16.1: type = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc7_16.2 (constants.%U)]
// CHECK:STDOUT:     %pair.param: @F.%tuple.type (%tuple.type.2) = value_param runtime_param0
// CHECK:STDOUT:     %pair: @F.%tuple.type (%tuple.type.2) = bind_name pair, %pair.param
// CHECK:STDOUT:     %return.param: ref @F.%U.loc7_16.2 (%U) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.%U.loc7_16.2 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %pair.patt: %tuple.type.3 = binding_pattern pair
// CHECK:STDOUT:     %pair.param_patt: %tuple.type.3 = value_param_pattern %pair.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %D.ref.loc9_16: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %.loc9_17.1: %tuple.type.1 = tuple_literal (%C.ref, %D.ref.loc9_16)
// CHECK:STDOUT:     %.loc9_17.2: type = converted %.loc9_17.1, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:     %D.ref.loc9_23: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %pair.param: %tuple.type.3 = value_param runtime_param0
// CHECK:STDOUT:     %pair: %tuple.type.3 = bind_name pair, %pair.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
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
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.1: type, %U.loc7_16.1: type) {
// CHECK:STDOUT:   %T.loc7_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %U.loc7_16.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc7_16.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc7_16.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc7_16.2 (constants.%U.patt)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@F.%T.loc7_6.2 (%T), @F.%U.loc7_16.2 (%U)) [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_54.2: <specific function> = specific_function constants.%F, @F(%T.loc7_6.2, %U.loc7_16.2) [symbolic = %.loc7_54.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type, %U.param_patt: type](%pair.param_patt: @F.%tuple.type (%tuple.type.2)) -> @F.%U.loc7_16.2 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %pair.ref: @F.%tuple.type (%tuple.type.2) = name_ref pair, %pair
// CHECK:STDOUT:     %.loc7_54.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [symbolic = %.loc7_54.2 (constants.%.4)]
// CHECK:STDOUT:     %F.call: init @F.%U.loc7_16.2 (%U) = call %.loc7_54.1(%pair.ref)
// CHECK:STDOUT:     %.loc7_61.1: @F.%U.loc7_16.2 (%U) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc7_61.2: @F.%U.loc7_16.2 (%U) = converted %F.call, %.loc7_61.1
// CHECK:STDOUT:     return %.loc7_61.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%pair.param_patt: %tuple.type.3) -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %pair.ref: %tuple.type.3 = name_ref pair, %pair
// CHECK:STDOUT:   %.loc10: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%D) [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_20.2: ref %D = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %D = call %.loc10(%pair.ref) to %.loc9_20.2
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %U.loc7_16.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc7_16.2 => constants.%U
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_54.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc7_6.2, %U.loc7_16.2) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %U.loc7_16.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc7_16.2 => constants.%U
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%D) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc7_6.2 => constants.%C
// CHECK:STDOUT:   %U.loc7_16.2 => constants.%D
// CHECK:STDOUT:   %U.patt.loc7_16.2 => constants.%D
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_54.2 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %Pair: %tuple.type.2 = bind_symbolic_name Pair, 0 [symbolic]
// CHECK:STDOUT:   %Pair.patt: %tuple.type.2 = symbolic_binding_pattern Pair, 0 [symbolic]
// CHECK:STDOUT:   %HasPair.type: type = generic_class_type @HasPair [template]
// CHECK:STDOUT:   %HasPair.1: %HasPair.type = struct_value () [template]
// CHECK:STDOUT:   %HasPair.2: type = class_type @HasPair, @HasPair(%Pair) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %A: %i32 = bind_symbolic_name A, 0 [symbolic]
// CHECK:STDOUT:   %A.patt: %i32 = symbolic_binding_pattern A, 0 [symbolic]
// CHECK:STDOUT:   %B: %i32 = bind_symbolic_name B, 1 [symbolic]
// CHECK:STDOUT:   %B.patt: %i32 = symbolic_binding_pattern B, 1 [symbolic]
// CHECK:STDOUT:   %tuple.1: %tuple.type.2 = tuple_value (%A, %B) [symbolic]
// CHECK:STDOUT:   %HasPair.3: type = class_type @HasPair, @HasPair(%tuple.1) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.31: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.7, %Convert.14 [template]
// CHECK:STDOUT:   %.36: <specific function> = specific_function %.35, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.37: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple.2: %tuple.type.2 = tuple_value (%.34, %.37) [template]
// CHECK:STDOUT:   %HasPair.4: type = class_type @HasPair, @HasPair(%tuple.2) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.38: <specific function> = specific_function %F, @F(%.34, %.37) [template]
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
// CHECK:STDOUT:     .HasPair = %HasPair.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasPair.decl: %HasPair.type = class_decl @HasPair [template = constants.%HasPair.1] {
// CHECK:STDOUT:     %Pair.patt.loc4_15.1: %tuple.type.2 = symbolic_binding_pattern Pair, 0 [symbolic = %Pair.patt.loc4_15.2 (constants.%Pair.patt)]
// CHECK:STDOUT:     %Pair.param_patt: %tuple.type.2 = value_param_pattern %Pair.patt.loc4_15.1, runtime_param<invalid> [symbolic = %Pair.patt.loc4_15.2 (constants.%Pair.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_23: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_23: init type = call constants.%Int(%.loc4_23) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_28: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_28: init type = call constants.%Int(%.loc4_28) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_31.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc4_23, %int.make_type_signed.loc4_28)
// CHECK:STDOUT:     %.loc4_31.2: type = value_of_initializer %int.make_type_signed.loc4_23 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_31.3: type = converted %int.make_type_signed.loc4_23, %.loc4_31.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_31.4: type = value_of_initializer %int.make_type_signed.loc4_28 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_31.5: type = converted %int.make_type_signed.loc4_28, %.loc4_31.4 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_31.6: type = converted %.loc4_31.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %Pair.param: %tuple.type.2 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %Pair.loc4_15.1: %tuple.type.2 = bind_symbolic_name Pair, 0, %Pair.param [symbolic = %Pair.loc4_15.2 (constants.%Pair)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %A.patt.loc6_6.1: %i32 = symbolic_binding_pattern A, 0 [symbolic = %A.patt.loc6_6.2 (constants.%A.patt)]
// CHECK:STDOUT:     %A.param_patt: %i32 = value_param_pattern %A.patt.loc6_6.1, runtime_param<invalid> [symbolic = %A.patt.loc6_6.2 (constants.%A.patt)]
// CHECK:STDOUT:     %B.patt.loc6_15.1: %i32 = symbolic_binding_pattern B, 1 [symbolic = %B.patt.loc6_15.2 (constants.%B.patt)]
// CHECK:STDOUT:     %B.param_patt: %i32 = value_param_pattern %B.patt.loc6_15.1, runtime_param<invalid> [symbolic = %B.patt.loc6_15.2 (constants.%B.patt)]
// CHECK:STDOUT:     %h.patt: @F.%HasPair.loc6_34.2 (%HasPair.3) = binding_pattern h
// CHECK:STDOUT:     %h.param_patt: @F.%HasPair.loc6_34.2 (%HasPair.3) = value_param_pattern %h.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc6_10: init type = call constants.%Int(%.loc6_10.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_10.2: type = value_of_initializer %int.make_type_signed.loc6_10 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_10.3: type = converted %int.make_type_signed.loc6_10, %.loc6_10.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc6_19: init type = call constants.%Int(%.loc6_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_19.2: type = value_of_initializer %int.make_type_signed.loc6_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_19.3: type = converted %int.make_type_signed.loc6_19, %.loc6_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %HasPair.ref: %HasPair.type = name_ref HasPair, file.%HasPair.decl [template = constants.%HasPair.1]
// CHECK:STDOUT:     %A.ref: %i32 = name_ref A, %A.loc6_6.1 [symbolic = %A.loc6_6.2 (constants.%A)]
// CHECK:STDOUT:     %B.ref.loc6_39: %i32 = name_ref B, %B.loc6_15.1 [symbolic = %B.loc6_15.2 (constants.%B)]
// CHECK:STDOUT:     %.loc6_40: %tuple.type.2 = tuple_literal (%A.ref, %B.ref.loc6_39)
// CHECK:STDOUT:     %tuple.loc6_40.1: %tuple.type.2 = tuple_value (%A.ref, %B.ref.loc6_39) [symbolic = %tuple.loc6_40.2 (constants.%tuple.1)]
// CHECK:STDOUT:     %.loc6_34: %tuple.type.2 = converted %.loc6_40, %tuple.loc6_40.1 [symbolic = %tuple.loc6_40.2 (constants.%tuple.1)]
// CHECK:STDOUT:     %HasPair.loc6_34.1: type = class_type @HasPair, @HasPair(constants.%tuple.1) [symbolic = %HasPair.loc6_34.2 (constants.%HasPair.3)]
// CHECK:STDOUT:     %.loc6_47.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc6_47: init type = call constants.%Int(%.loc6_47.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_47.2: type = value_of_initializer %int.make_type_signed.loc6_47 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_47.3: type = converted %int.make_type_signed.loc6_47, %.loc6_47.2 [template = constants.%i32]
// CHECK:STDOUT:     %A.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %A.loc6_6.1: %i32 = bind_symbolic_name A, 0, %A.param [symbolic = %A.loc6_6.2 (constants.%A)]
// CHECK:STDOUT:     %B.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %B.loc6_15.1: %i32 = bind_symbolic_name B, 1, %B.param [symbolic = %B.loc6_15.2 (constants.%B)]
// CHECK:STDOUT:     %h.param: @F.%HasPair.loc6_34.2 (%HasPair.3) = value_param runtime_param0
// CHECK:STDOUT:     %h: @F.%HasPair.loc6_34.2 (%HasPair.3) = bind_name h, %h.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %h.patt: %HasPair.4 = binding_pattern h
// CHECK:STDOUT:     %h.param_patt: %HasPair.4 = value_param_pattern %h.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %HasPair.ref: %HasPair.type = name_ref HasPair, file.%HasPair.decl [template = constants.%HasPair.1]
// CHECK:STDOUT:     %.loc8_18: Core.IntLiteral = int_value 1 [template = constants.%.6]
// CHECK:STDOUT:     %.loc8_21: Core.IntLiteral = int_value 2 [template = constants.%.7]
// CHECK:STDOUT:     %.loc8_22.1: %tuple.type.3 = tuple_literal (%.loc8_18, %.loc8_21)
// CHECK:STDOUT:     %.loc8_22.2: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %.loc8_22.3: <bound method> = bound_method %.loc8_18, %.loc8_22.2 [template = constants.%.32]
// CHECK:STDOUT:     %.loc8_22.4: <specific function> = specific_function %.loc8_22.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:     %int.convert_checked.loc8_22.1: init %i32 = call %.loc8_22.4(%.loc8_18) [template = constants.%.34]
// CHECK:STDOUT:     %.loc8_22.5: %i32 = value_of_initializer %int.convert_checked.loc8_22.1 [template = constants.%.34]
// CHECK:STDOUT:     %.loc8_22.6: %i32 = converted %.loc8_18, %.loc8_22.5 [template = constants.%.34]
// CHECK:STDOUT:     %.loc8_22.7: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %.loc8_22.8: <bound method> = bound_method %.loc8_21, %.loc8_22.7 [template = constants.%.35]
// CHECK:STDOUT:     %.loc8_22.9: <specific function> = specific_function %.loc8_22.8, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:     %int.convert_checked.loc8_22.2: init %i32 = call %.loc8_22.9(%.loc8_21) [template = constants.%.37]
// CHECK:STDOUT:     %.loc8_22.10: %i32 = value_of_initializer %int.convert_checked.loc8_22.2 [template = constants.%.37]
// CHECK:STDOUT:     %.loc8_22.11: %i32 = converted %.loc8_21, %.loc8_22.10 [template = constants.%.37]
// CHECK:STDOUT:     %tuple: %tuple.type.2 = tuple_value (%.loc8_22.6, %.loc8_22.11) [template = constants.%tuple.2]
// CHECK:STDOUT:     %.loc8_16: %tuple.type.2 = converted %.loc8_22.1, %tuple [template = constants.%tuple.2]
// CHECK:STDOUT:     %HasPair: type = class_type @HasPair, @HasPair(constants.%tuple.2) [template = constants.%HasPair.4]
// CHECK:STDOUT:     %.loc8_29.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc8_29.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_29.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_29.3: type = converted %int.make_type_signed, %.loc8_29.2 [template = constants.%i32]
// CHECK:STDOUT:     %h.param: %HasPair.4 = value_param runtime_param0
// CHECK:STDOUT:     %h: %HasPair.4 = bind_name h, %h.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HasPair(%Pair.loc4_15.1: %tuple.type.2) {
// CHECK:STDOUT:   %Pair.loc4_15.2: %tuple.type.2 = bind_symbolic_name Pair, 0 [symbolic = %Pair.loc4_15.2 (constants.%Pair)]
// CHECK:STDOUT:   %Pair.patt.loc4_15.2: %tuple.type.2 = symbolic_binding_pattern Pair, 0 [symbolic = %Pair.patt.loc4_15.2 (constants.%Pair.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4_35: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HasPair.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%A.loc6_6.1: %i32, %B.loc6_15.1: %i32) {
// CHECK:STDOUT:   %A.loc6_6.2: %i32 = bind_symbolic_name A, 0 [symbolic = %A.loc6_6.2 (constants.%A)]
// CHECK:STDOUT:   %A.patt.loc6_6.2: %i32 = symbolic_binding_pattern A, 0 [symbolic = %A.patt.loc6_6.2 (constants.%A.patt)]
// CHECK:STDOUT:   %B.loc6_15.2: %i32 = bind_symbolic_name B, 1 [symbolic = %B.loc6_15.2 (constants.%B)]
// CHECK:STDOUT:   %B.patt.loc6_15.2: %i32 = symbolic_binding_pattern B, 1 [symbolic = %B.patt.loc6_15.2 (constants.%B.patt)]
// CHECK:STDOUT:   %tuple.loc6_40.2: %tuple.type.2 = tuple_value (%A.loc6_6.2, %B.loc6_15.2) [symbolic = %tuple.loc6_40.2 (constants.%tuple.1)]
// CHECK:STDOUT:   %HasPair.loc6_34.2: type = class_type @HasPair, @HasPair(%tuple.loc6_40.2) [symbolic = %HasPair.loc6_34.2 (constants.%HasPair.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%A.param_patt: %i32, %B.param_patt: %i32](%h.param_patt: @F.%HasPair.loc6_34.2 (%HasPair.3)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %B.ref.loc6_60: %i32 = name_ref B, %B.loc6_15.1 [symbolic = %B.loc6_15.2 (constants.%B)]
// CHECK:STDOUT:     return %B.ref.loc6_60
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%h.param_patt: %HasPair.4) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %h.ref: %HasPair.4 = name_ref h, %h
// CHECK:STDOUT:   %.loc9_10: <specific function> = specific_function %F.ref, @F(constants.%.34, constants.%.37) [template = constants.%.38]
// CHECK:STDOUT:   %F.call: init %i32 = call %.loc9_10(%h.ref)
// CHECK:STDOUT:   %.loc9_14.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc9_14.2: %i32 = converted %F.call, %.loc9_14.1
// CHECK:STDOUT:   return %.loc9_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%Pair) {
// CHECK:STDOUT:   %Pair.loc4_15.2 => constants.%Pair
// CHECK:STDOUT:   %Pair.patt.loc4_15.2 => constants.%Pair
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%tuple.1) {
// CHECK:STDOUT:   %Pair.loc4_15.2 => constants.%tuple.1
// CHECK:STDOUT:   %Pair.patt.loc4_15.2 => constants.%tuple.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(@F.%tuple.loc6_40.2) {
// CHECK:STDOUT:   %Pair.loc4_15.2 => constants.%tuple.1
// CHECK:STDOUT:   %Pair.patt.loc4_15.2 => constants.%tuple.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.loc6_6.2 => constants.%A
// CHECK:STDOUT:   %A.patt.loc6_6.2 => constants.%A
// CHECK:STDOUT:   %B.loc6_15.2 => constants.%B
// CHECK:STDOUT:   %B.patt.loc6_15.2 => constants.%B
// CHECK:STDOUT:   %tuple.loc6_40.2 => constants.%tuple.1
// CHECK:STDOUT:   %HasPair.loc6_34.2 => constants.%HasPair.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasPair(constants.%tuple.2) {
// CHECK:STDOUT:   %Pair.loc4_15.2 => constants.%tuple.2
// CHECK:STDOUT:   %Pair.patt.loc4_15.2 => constants.%tuple.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%.34, constants.%.37) {
// CHECK:STDOUT:   %A.loc6_6.2 => constants.%.34
// CHECK:STDOUT:   %A.patt.loc6_6.2 => constants.%.34
// CHECK:STDOUT:   %B.loc6_15.2 => constants.%.37
// CHECK:STDOUT:   %B.patt.loc6_15.2 => constants.%.37
// CHECK:STDOUT:   %tuple.loc6_40.2 => constants.%tuple.2
// CHECK:STDOUT:   %HasPair.loc6_34.2 => constants.%HasPair.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inconsistent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%T, %T) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%C, %D) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc7_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc7_6.1, runtime_param<invalid> [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %pair.patt: @F.%tuple.type (%tuple.type.2) = binding_pattern pair
// CHECK:STDOUT:     %pair.param_patt: @F.%tuple.type (%tuple.type.2) = value_param_pattern %pair.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc7_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc7_6.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc7_23: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc7_26: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc7_27.1: %tuple.type.1 = tuple_literal (%T.ref.loc7_23, %T.ref.loc7_26)
// CHECK:STDOUT:     %.loc7_27.2: type = converted %.loc7_27.1, constants.%tuple.type.2 [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:     %T.ref.loc7_33: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc7_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %pair.param: @F.%tuple.type (%tuple.type.2) = value_param runtime_param0
// CHECK:STDOUT:     %pair: @F.%tuple.type (%tuple.type.2) = bind_name pair, %pair.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc7_6.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc7_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %pair.patt: %tuple.type.3 = binding_pattern pair
// CHECK:STDOUT:     %pair.param_patt: %tuple.type.3 = value_param_pattern %pair.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %D.ref.loc9_16: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %.loc9_17.1: %tuple.type.1 = tuple_literal (%C.ref, %D.ref.loc9_16)
// CHECK:STDOUT:     %.loc9_17.2: type = converted %.loc9_17.1, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:     %D.ref.loc9_23: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %pair.param: %tuple.type.3 = value_param runtime_param0
// CHECK:STDOUT:     %pair: %tuple.type.3 = bind_name pair, %pair.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
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
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.1: type) {
// CHECK:STDOUT:   %T.loc7_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@F.%T.loc7_6.2 (%T), @F.%T.loc7_6.2 (%T)) [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%pair.param_patt: @F.%tuple.type (%tuple.type.2)) -> @F.%T.loc7_6.2 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%pair.param_patt: %tuple.type.3) -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %pair.ref: %tuple.type.3 = name_ref pair, %pair
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
