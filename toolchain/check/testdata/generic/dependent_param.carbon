// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/dependent_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/dependent_param.carbon

// --- nested_class.carbon

library "[[@TEST_NAME]]";

class Outer(T:! type) {
  class Inner(U:! T) {
    fn Get() -> T { return U; }
  }
}

var n: i32 = Outer(i32).Inner(42).Get();

// CHECK:STDOUT: --- nested_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.9d6: type = class_type @Outer, @Outer(%T.8b3) [symbolic]
// CHECK:STDOUT:   %U.f22: %T.8b3 = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %Inner.type.eae: type = generic_class_type @Inner, @Outer(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Inner.generic.137: %Inner.type.eae = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.879: type = class_type @Inner, @Inner(%T.8b3, %U.f22) [symbolic]
// CHECK:STDOUT:   %Get.type.889: type = fn_type @Get, @Inner(%T.8b3, %U.f22) [symbolic]
// CHECK:STDOUT:   %Get.118: %Get.type.889 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Outer.545: type = class_type @Outer, @Outer(%i32) [concrete]
// CHECK:STDOUT:   %Inner.type.bac: type = generic_class_type @Inner, @Outer(%i32) [concrete]
// CHECK:STDOUT:   %Inner.generic.ba0: %Inner.type.bac = struct_value () [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.3, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.3, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_42.20e, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_42.20e, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.c68: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %Inner.025: type = class_type @Inner, @Inner(%i32, %int_42.c68) [concrete]
// CHECK:STDOUT:   %Get.type.512: type = fn_type @Get, @Inner(%i32, %int_42.c68) [concrete]
// CHECK:STDOUT:   %Get.18c: %Get.type.512 = struct_value () [concrete]
// CHECK:STDOUT:   %Get.specific_fn: <specific function> = specific_function %Get.18c, @Get(%i32, %int_42.c68) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: %Outer.type = class_decl @Outer [concrete = constants.%Outer.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.7ce = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i32 = var %n.var_patt [concrete]
// CHECK:STDOUT:   %.loc10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i32 = bind_name n, %n.var [concrete = %n.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc4_13.2) [symbolic = %Inner.type (constants.%Inner.type.eae)]
// CHECK:STDOUT:   %Inner.generic: @Outer.%Inner.type (%Inner.type.eae) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.137)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.eae) = class_decl @Inner [symbolic = @Outer.%Inner.generic (constants.%Inner.generic.137)] {
// CHECK:STDOUT:       %U.patt: @Inner.%pattern_type (%pattern_type.7dcd0a.1) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_13.1 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:       %U.loc5_15.1: @Inner.%T (%T.8b3) = bind_symbolic_name U, 1 [symbolic = %U.loc5_15.2 (constants.%U.f22)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.9d6
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_13.1: type, %U.loc5_15.1: @Inner.%T (%T.8b3)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %U.loc5_15.2: @Inner.%T (%T.8b3) = bind_symbolic_name U, 1 [symbolic = %U.loc5_15.2 (constants.%U.f22)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type: type = fn_type @Get, @Inner(%T, %U.loc5_15.2) [symbolic = %Get.type (constants.%Get.type.889)]
// CHECK:STDOUT:   %Get: @Inner.%Get.type (%Get.type.889) = struct_value () [symbolic = %Get (constants.%Get.118)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Get.decl: @Inner.%Get.type (%Get.type.889) = fn_decl @Get [symbolic = @Inner.%Get (constants.%Get.118)] {
// CHECK:STDOUT:       %return.patt: @Get.%pattern_type (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Get.%pattern_type (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_13.1 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:       %return.param: ref @Get.%T (%T.8b3) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Get.%T (%T.8b3) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.879
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Get = %Get.decl
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Get(@Outer.%T.loc4_13.1: type, @Inner.%U.loc5_15.1: @Inner.%T (%T.8b3)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %U: @Get.%T (%T.8b3) = bind_symbolic_name U, 1 [symbolic = %U (constants.%U.f22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Get.%T (%T.8b3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref: @Get.%T (%T.8b3) = name_ref U, @Inner.%U.loc5_15.1 [symbolic = %U (constants.%U.f22)]
// CHECK:STDOUT:     return %U.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.545]
// CHECK:STDOUT:   %.loc10_24: %Inner.type.bac = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.generic.ba0]
// CHECK:STDOUT:   %Inner.ref: %Inner.type.bac = name_ref Inner, %.loc10_24 [concrete = constants.%Inner.generic.ba0]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc10_33.1: <bound method> = bound_method constants.%int_42.20e, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_33.2: <bound method> = bound_method constants.%int_42.20e, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc10_33.2(constants.%int_42.20e) [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc10_33.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc10_33.2: %i32 = converted constants.%int_42.20e, %.loc10_33.1 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(constants.%i32, constants.%int_42.c68) [concrete = constants.%Inner.025]
// CHECK:STDOUT:   %.loc10_34: %Get.type.512 = specific_constant @Inner.%Get.decl, @Inner(constants.%i32, constants.%int_42.c68) [concrete = constants.%Get.18c]
// CHECK:STDOUT:   %Get.ref: %Get.type.512 = name_ref Get, %.loc10_34 [concrete = constants.%Get.18c]
// CHECK:STDOUT:   %Get.specific_fn: <specific function> = specific_function %Get.ref, @Get(constants.%i32, constants.%int_42.c68) [concrete = constants.%Get.specific_fn]
// CHECK:STDOUT:   %Get.call: init %i32 = call %Get.specific_fn()
// CHECK:STDOUT:   assign file.%n.var, %Get.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T.8b3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.eae
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.8b3, constants.%U.f22) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %U.loc5_15.2 => constants.%U.f22
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type => constants.%Get.type.889
// CHECK:STDOUT:   %Get => constants.%Get.118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%T.8b3, constants.%U.f22) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.bac
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.ba0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%i32, constants.%int_42.c68) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %U.loc5_15.2 => constants.%int_42.c68
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type => constants.%Get.type.512
// CHECK:STDOUT:   %Get => constants.%Get.18c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%i32, constants.%int_42.c68) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %U => constants.%int_42.c68
// CHECK:STDOUT: }
// CHECK:STDOUT:
