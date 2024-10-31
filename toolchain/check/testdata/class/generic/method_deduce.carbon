// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/method_deduce.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/method_deduce.carbon
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:11:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as Add {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:14:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as AddAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:21:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as Inc {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:28:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:31:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as BitAndAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:38:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as BitComplement {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:42:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as BitOr {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:45:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as BitOrAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:52:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as BitXor {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:55:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as BitXorAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:62:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as Div {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:65:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as DivAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:72:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as Eq {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:77:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as LeftShift {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:80:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as LeftShiftAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:87:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as Mod {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:90:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as ModAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:97:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as Mul {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:100:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as MulAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:107:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as Negate {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:111:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as Ordered {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:119:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as RightShift {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:122:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as RightShiftAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:129:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as Sub {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:132:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as SubAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/d45151c2a6e33b065e794da6da9cf423/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon:138:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl i32 as Dec {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~

class A {}
class B {}

class Class(T:! type) {
  fn Get(U:! type) -> (T, U) { return Get(U); }
  fn GetNoDeduce(x: T, U:! type) -> (T, U) { return GetNoDeduce(x, U); }
}

fn CallGenericMethod(c: Class(A)) -> (A, B) {
  return c.Get(B);
}

fn CallGenericMethodWithNonDeducedParam(c: Class(A)) -> (A, B) {
  return c.GetNoDeduce({}, B);
}

// CHECK:STDOUT: --- method_deduce.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt.1: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %.4: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %Get.type.1: type = fn_type @Get, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Get.1: %Get.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %U.patt.2: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %GetNoDeduce.type.1: type = fn_type @GetNoDeduce, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetNoDeduce.1: %GetNoDeduce.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [symbolic]
// CHECK:STDOUT:   %.7: <specific function> = specific_function %Get.1, @Get(%T, %U) [symbolic]
// CHECK:STDOUT:   %.8: <specific function> = specific_function %GetNoDeduce.1, @GetNoDeduce(%T, %U) [symbolic]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(%A) [template]
// CHECK:STDOUT:   %.9: type = tuple_type (%A, %B) [template]
// CHECK:STDOUT:   %CallGenericMethod.type: type = fn_type @CallGenericMethod [template]
// CHECK:STDOUT:   %CallGenericMethod: %CallGenericMethod.type = struct_value () [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.11: type = tuple_type (%.10, %.10) [template]
// CHECK:STDOUT:   %.12: type = ptr_type %.11 [template]
// CHECK:STDOUT:   %Get.type.2: type = fn_type @Get, @Class(%A) [template]
// CHECK:STDOUT:   %Get.2: %Get.type.2 = struct_value () [template]
// CHECK:STDOUT:   %GetNoDeduce.type.2: type = fn_type @GetNoDeduce, @Class(%A) [template]
// CHECK:STDOUT:   %GetNoDeduce.2: %GetNoDeduce.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.13: <specific function> = specific_function %Get.2, @Get(%A, %B) [template]
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam.type: type = fn_type @CallGenericMethodWithNonDeducedParam [template]
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam: %CallGenericMethodWithNonDeducedParam.type = struct_value () [template]
// CHECK:STDOUT:   %.14: <specific function> = specific_function %GetNoDeduce.2, @GetNoDeduce(%A, %B) [template]
// CHECK:STDOUT:   %struct: %A = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/index
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .CallGenericMethod = %CallGenericMethod.decl
// CHECK:STDOUT:     .CallGenericMethodWithNonDeducedParam = %CallGenericMethodWithNonDeducedParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.patt.loc117_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc117_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc117_13.1, runtime_param<invalid> [symbolic = %T.patt.loc117_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc117_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc117_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethod.decl: %CallGenericMethod.type = fn_decl @CallGenericMethod [template = constants.%CallGenericMethod] {
// CHECK:STDOUT:     %c.patt: %Class.3 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %Class.3 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.9 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.9 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %A.ref.loc122_31: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%A) [template = constants.%Class.3]
// CHECK:STDOUT:     %A.ref.loc122_39: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %B.ref.loc122: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc122_43.1: %.4 = tuple_literal (%A.ref.loc122_39, %B.ref.loc122)
// CHECK:STDOUT:     %.loc122_43.2: type = converted %.loc122_43.1, constants.%.9 [template = constants.%.9]
// CHECK:STDOUT:     %c.param: %Class.3 = value_param runtime_param0
// CHECK:STDOUT:     %c: %Class.3 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %.9 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.9 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericMethodWithNonDeducedParam.decl: %CallGenericMethodWithNonDeducedParam.type = fn_decl @CallGenericMethodWithNonDeducedParam [template = constants.%CallGenericMethodWithNonDeducedParam] {
// CHECK:STDOUT:     %c.patt: %Class.3 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %Class.3 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.9 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.9 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %A.ref.loc126_50: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%A) [template = constants.%Class.3]
// CHECK:STDOUT:     %A.ref.loc126_58: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %B.ref.loc126: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc126_62.1: %.4 = tuple_literal (%A.ref.loc126_58, %B.ref.loc126)
// CHECK:STDOUT:     %.loc126_62.2: type = converted %.loc126_62.1, constants.%.9 [template = constants.%.9]
// CHECK:STDOUT:     %c.param: %Class.3 = value_param runtime_param0
// CHECK:STDOUT:     %c: %Class.3 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %.9 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.9 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc114: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %.loc115: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc117_13.1: type) {
// CHECK:STDOUT:   %T.loc117_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc117_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc117_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc117_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type: type = fn_type @Get, @Class(%T.loc117_13.2) [symbolic = %Get.type (constants.%Get.type.1)]
// CHECK:STDOUT:   %Get: @Class.%Get.type (%Get.type.1) = struct_value () [symbolic = %Get (constants.%Get.1)]
// CHECK:STDOUT:   %GetNoDeduce.type: type = fn_type @GetNoDeduce, @Class(%T.loc117_13.2) [symbolic = %GetNoDeduce.type (constants.%GetNoDeduce.type.1)]
// CHECK:STDOUT:   %GetNoDeduce: @Class.%GetNoDeduce.type (%GetNoDeduce.type.1) = struct_value () [symbolic = %GetNoDeduce (constants.%GetNoDeduce.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Get.decl: @Class.%Get.type (%Get.type.1) = fn_decl @Get [symbolic = @Class.%Get (constants.%Get.1)] {
// CHECK:STDOUT:       %U.patt.loc118_10.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc118_10.1 (constants.%U.patt.1)]
// CHECK:STDOUT:       %U.param_patt: type = value_param_pattern %U.patt.loc118_10.2, runtime_param<invalid> [symbolic = %U.patt.loc118_10.1 (constants.%U.patt.1)]
// CHECK:STDOUT:       %return.patt: @Get.%.loc118_28.1 (%.5) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Get.%.loc118_28.1 (%.5) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc117_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc118_27: type = name_ref U, %U.loc118_10.2 [symbolic = %U.loc118_10.1 (constants.%U)]
// CHECK:STDOUT:       %.loc118_28.2: %.4 = tuple_literal (%T.ref, %U.ref.loc118_27)
// CHECK:STDOUT:       %.loc118_28.3: type = converted %.loc118_28.2, constants.%.5 [symbolic = %.loc118_28.1 (constants.%.5)]
// CHECK:STDOUT:       %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %U.loc118_10.2: type = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc118_10.1 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @Get.%.loc118_28.1 (%.5) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @Get.%.loc118_28.1 (%.5) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GetNoDeduce.decl: @Class.%GetNoDeduce.type (%GetNoDeduce.type.1) = fn_decl @GetNoDeduce [symbolic = @Class.%GetNoDeduce (constants.%GetNoDeduce.1)] {
// CHECK:STDOUT:       %x.patt: @GetNoDeduce.%T (%T) = binding_pattern x
// CHECK:STDOUT:       %x.param_patt: @GetNoDeduce.%T (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:       %U.patt.loc119_24.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc119_24.1 (constants.%U.patt.2)]
// CHECK:STDOUT:       %U.param_patt: type = value_param_pattern %U.patt.loc119_24.2, runtime_param<invalid> [symbolic = %U.patt.loc119_24.1 (constants.%U.patt.2)]
// CHECK:STDOUT:       %return.patt: @GetNoDeduce.%.loc119_42.1 (%.5) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @GetNoDeduce.%.loc119_42.1 (%.5) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc119_21: type = name_ref T, @Class.%T.loc117_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %T.ref.loc119_38: type = name_ref T, @Class.%T.loc117_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc119_41: type = name_ref U, %U.loc119_24.2 [symbolic = %U.loc119_24.1 (constants.%U)]
// CHECK:STDOUT:       %.loc119_42.2: %.4 = tuple_literal (%T.ref.loc119_38, %U.ref.loc119_41)
// CHECK:STDOUT:       %.loc119_42.3: type = converted %.loc119_42.2, constants.%.5 [symbolic = %.loc119_42.1 (constants.%.5)]
// CHECK:STDOUT:       %x.param: @GetNoDeduce.%T (%T) = value_param runtime_param0
// CHECK:STDOUT:       %x: @GetNoDeduce.%T (%T) = bind_name x, %x.param
// CHECK:STDOUT:       %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %U.loc119_24.2: type = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc119_24.1 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @GetNoDeduce.%.loc119_42.1 (%.5) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @GetNoDeduce.%.loc119_42.1 (%.5) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc120: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .Get = %Get.decl
// CHECK:STDOUT:     .GetNoDeduce = %GetNoDeduce.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Get(@Class.%T.loc117_13.1: type, %U.loc118_10.2: type) {
// CHECK:STDOUT:   %U.loc118_10.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc118_10.1 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc118_10.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc118_10.1 (constants.%U.patt.1)]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.loc118_28.1: type = tuple_type (@Get.%T (%T), @Get.%U.loc118_10.1 (%U)) [symbolic = %.loc118_28.1 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type: type = fn_type @Get, @Class(%T) [symbolic = %Get.type (constants.%Get.type.1)]
// CHECK:STDOUT:   %Get: @Get.%Get.type (%Get.type.1) = struct_value () [symbolic = %Get (constants.%Get.1)]
// CHECK:STDOUT:   %.loc118_39.3: <specific function> = specific_function %Get, @Get(%T, %U.loc118_10.1) [symbolic = %.loc118_39.3 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param_patt: type) -> %return: @Get.%.loc118_28.1 (%.5) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc118_39.1: @Get.%Get.type (%Get.type.1) = specific_constant @Class.%Get.decl, @Class(constants.%T) [symbolic = %Get (constants.%Get.1)]
// CHECK:STDOUT:     %Get.ref: @Get.%Get.type (%Get.type.1) = name_ref Get, %.loc118_39.1 [symbolic = %Get (constants.%Get.1)]
// CHECK:STDOUT:     %U.ref.loc118_43: type = name_ref U, %U.loc118_10.2 [symbolic = %U.loc118_10.1 (constants.%U)]
// CHECK:STDOUT:     %.loc118_39.2: <specific function> = specific_function %Get.ref, @Get(constants.%T, constants.%U) [symbolic = %.loc118_39.3 (constants.%.7)]
// CHECK:STDOUT:     %.loc118_42.1: ref @Get.%.loc118_28.1 (%.5) = temporary_storage
// CHECK:STDOUT:     %Get.call: init @Get.%.loc118_28.1 (%.5) = call %.loc118_39.2() to %.loc118_42.1
// CHECK:STDOUT:     %.loc118_42.2: ref @Get.%.loc118_28.1 (%.5) = temporary %.loc118_42.1, %Get.call
// CHECK:STDOUT:     %.loc118_42.3: ref @Get.%T (%T) = tuple_access %.loc118_42.2, element0
// CHECK:STDOUT:     %.loc118_42.4: @Get.%T (%T) = bind_value %.loc118_42.3
// CHECK:STDOUT:     %.loc118_42.5: ref @Get.%T (%T) = tuple_access %return, element0
// CHECK:STDOUT:     %.loc118_42.6: init @Get.%T (%T) = initialize_from %.loc118_42.4 to %.loc118_42.5
// CHECK:STDOUT:     %.loc118_42.7: ref @Get.%U.loc118_10.1 (%U) = tuple_access %.loc118_42.2, element1
// CHECK:STDOUT:     %.loc118_42.8: @Get.%U.loc118_10.1 (%U) = bind_value %.loc118_42.7
// CHECK:STDOUT:     %.loc118_42.9: ref @Get.%U.loc118_10.1 (%U) = tuple_access %return, element1
// CHECK:STDOUT:     %.loc118_42.10: init @Get.%U.loc118_10.1 (%U) = initialize_from %.loc118_42.8 to %.loc118_42.9
// CHECK:STDOUT:     %.loc118_42.11: init @Get.%.loc118_28.1 (%.5) = tuple_init (%.loc118_42.6, %.loc118_42.10) to %return
// CHECK:STDOUT:     %.loc118_45: init @Get.%.loc118_28.1 (%.5) = converted %Get.call, %.loc118_42.11
// CHECK:STDOUT:     return %.loc118_45 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetNoDeduce(@Class.%T.loc117_13.1: type, %U.loc119_24.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %U.loc119_24.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc119_24.1 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc119_24.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc119_24.1 (constants.%U.patt.2)]
// CHECK:STDOUT:   %.loc119_42.1: type = tuple_type (@GetNoDeduce.%T (%T), @GetNoDeduce.%U.loc119_24.1 (%U)) [symbolic = %.loc119_42.1 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GetNoDeduce.type: type = fn_type @GetNoDeduce, @Class(%T) [symbolic = %GetNoDeduce.type (constants.%GetNoDeduce.type.1)]
// CHECK:STDOUT:   %GetNoDeduce: @GetNoDeduce.%GetNoDeduce.type (%GetNoDeduce.type.1) = struct_value () [symbolic = %GetNoDeduce (constants.%GetNoDeduce.1)]
// CHECK:STDOUT:   %.loc119_53.3: <specific function> = specific_function %GetNoDeduce, @GetNoDeduce(%T, %U.loc119_24.1) [symbolic = %.loc119_53.3 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: @GetNoDeduce.%T (%T), %U.param_patt: type) -> %return: @GetNoDeduce.%.loc119_42.1 (%.5) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc119_53.1: @GetNoDeduce.%GetNoDeduce.type (%GetNoDeduce.type.1) = specific_constant @Class.%GetNoDeduce.decl, @Class(constants.%T) [symbolic = %GetNoDeduce (constants.%GetNoDeduce.1)]
// CHECK:STDOUT:     %GetNoDeduce.ref: @GetNoDeduce.%GetNoDeduce.type (%GetNoDeduce.type.1) = name_ref GetNoDeduce, %.loc119_53.1 [symbolic = %GetNoDeduce (constants.%GetNoDeduce.1)]
// CHECK:STDOUT:     %x.ref: @GetNoDeduce.%T (%T) = name_ref x, %x
// CHECK:STDOUT:     %U.ref.loc119_68: type = name_ref U, %U.loc119_24.2 [symbolic = %U.loc119_24.1 (constants.%U)]
// CHECK:STDOUT:     %.loc119_53.2: <specific function> = specific_function %GetNoDeduce.ref, @GetNoDeduce(constants.%T, constants.%U) [symbolic = %.loc119_53.3 (constants.%.8)]
// CHECK:STDOUT:     %.loc119_64.1: ref @GetNoDeduce.%.loc119_42.1 (%.5) = temporary_storage
// CHECK:STDOUT:     %GetNoDeduce.call: init @GetNoDeduce.%.loc119_42.1 (%.5) = call %.loc119_53.2(%x.ref) to %.loc119_64.1
// CHECK:STDOUT:     %.loc119_64.2: ref @GetNoDeduce.%.loc119_42.1 (%.5) = temporary %.loc119_64.1, %GetNoDeduce.call
// CHECK:STDOUT:     %.loc119_64.3: ref @GetNoDeduce.%T (%T) = tuple_access %.loc119_64.2, element0
// CHECK:STDOUT:     %.loc119_64.4: @GetNoDeduce.%T (%T) = bind_value %.loc119_64.3
// CHECK:STDOUT:     %.loc119_64.5: ref @GetNoDeduce.%T (%T) = tuple_access %return, element0
// CHECK:STDOUT:     %.loc119_64.6: init @GetNoDeduce.%T (%T) = initialize_from %.loc119_64.4 to %.loc119_64.5
// CHECK:STDOUT:     %.loc119_64.7: ref @GetNoDeduce.%U.loc119_24.1 (%U) = tuple_access %.loc119_64.2, element1
// CHECK:STDOUT:     %.loc119_64.8: @GetNoDeduce.%U.loc119_24.1 (%U) = bind_value %.loc119_64.7
// CHECK:STDOUT:     %.loc119_64.9: ref @GetNoDeduce.%U.loc119_24.1 (%U) = tuple_access %return, element1
// CHECK:STDOUT:     %.loc119_64.10: init @GetNoDeduce.%U.loc119_24.1 (%U) = initialize_from %.loc119_64.8 to %.loc119_64.9
// CHECK:STDOUT:     %.loc119_64.11: init @GetNoDeduce.%.loc119_42.1 (%.5) = tuple_init (%.loc119_64.6, %.loc119_64.10) to %return
// CHECK:STDOUT:     %.loc119_70: init @GetNoDeduce.%.loc119_42.1 (%.5) = converted %GetNoDeduce.call, %.loc119_64.11
// CHECK:STDOUT:     return %.loc119_70 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethod(%c.param_patt: %Class.3) -> %return: %.9 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.3 = name_ref c, %c
// CHECK:STDOUT:   %.loc123_11.1: %Get.type.2 = specific_constant @Class.%Get.decl, @Class(constants.%A) [template = constants.%Get.2]
// CHECK:STDOUT:   %Get.ref: %Get.type.2 = name_ref Get, %.loc123_11.1 [template = constants.%Get.2]
// CHECK:STDOUT:   %B.ref.loc123: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc123_11.2: <specific function> = specific_function %Get.ref, @Get(constants.%A, constants.%B) [template = constants.%.13]
// CHECK:STDOUT:   %.loc122_35.2: ref %.9 = splice_block %return {}
// CHECK:STDOUT:   %Get.call: init %.9 = call %.loc123_11.2() to %.loc122_35.2
// CHECK:STDOUT:   return %Get.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGenericMethodWithNonDeducedParam(%c.param_patt: %Class.3) -> %return: %.9 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.3 = name_ref c, %c
// CHECK:STDOUT:   %.loc127_11.1: %GetNoDeduce.type.2 = specific_constant @Class.%GetNoDeduce.decl, @Class(constants.%A) [template = constants.%GetNoDeduce.2]
// CHECK:STDOUT:   %GetNoDeduce.ref: %GetNoDeduce.type.2 = name_ref GetNoDeduce, %.loc127_11.1 [template = constants.%GetNoDeduce.2]
// CHECK:STDOUT:   %.loc127_25.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %B.ref.loc127: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc127_11.2: <specific function> = specific_function %GetNoDeduce.ref, @GetNoDeduce(constants.%A, constants.%B) [template = constants.%.14]
// CHECK:STDOUT:   %.loc126_54.2: ref %.9 = splice_block %return {}
// CHECK:STDOUT:   %.loc127_25.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc127_25.3: init %A = class_init (), %.loc127_25.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc127_25.4: ref %A = temporary %.loc127_25.2, %.loc127_25.3
// CHECK:STDOUT:   %.loc127_25.5: ref %A = converted %.loc127_25.1, %.loc127_25.4
// CHECK:STDOUT:   %.loc127_25.6: %A = bind_value %.loc127_25.5
// CHECK:STDOUT:   %GetNoDeduce.call: init %.9 = call %.loc127_11.2(%.loc127_25.6) to %.loc126_54.2
// CHECK:STDOUT:   return %GetNoDeduce.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc117_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc117_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type => constants.%Get.type.1
// CHECK:STDOUT:   %Get => constants.%Get.1
// CHECK:STDOUT:   %GetNoDeduce.type => constants.%GetNoDeduce.type.1
// CHECK:STDOUT:   %GetNoDeduce => constants.%GetNoDeduce.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc118_10.1 => constants.%U
// CHECK:STDOUT:   %U.patt.loc118_10.1 => constants.%U
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %.loc118_28.1 => constants.%.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type => constants.%Get.type.1
// CHECK:STDOUT:   %Get => constants.%Get.1
// CHECK:STDOUT:   %.loc118_39.3 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetNoDeduce(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %U.loc119_24.1 => constants.%U
// CHECK:STDOUT:   %U.patt.loc119_24.1 => constants.%U
// CHECK:STDOUT:   %.loc119_42.1 => constants.%.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GetNoDeduce.type => constants.%GetNoDeduce.type.1
// CHECK:STDOUT:   %GetNoDeduce => constants.%GetNoDeduce.1
// CHECK:STDOUT:   %.loc119_53.3 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Class.%T.loc117_13.2) {
// CHECK:STDOUT:   %T.loc117_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc117_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Get.%T) {
// CHECK:STDOUT:   %T.loc117_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc117_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(@Get.%T, @Get.%U.loc118_10.1) {
// CHECK:STDOUT:   %U.loc118_10.1 => constants.%U
// CHECK:STDOUT:   %U.patt.loc118_10.1 => constants.%U
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %.loc118_28.1 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@GetNoDeduce.%T) {
// CHECK:STDOUT:   %T.loc117_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc117_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetNoDeduce(@GetNoDeduce.%T, @GetNoDeduce.%U.loc119_24.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %U.loc119_24.1 => constants.%U
// CHECK:STDOUT:   %U.patt.loc119_24.1 => constants.%U
// CHECK:STDOUT:   %.loc119_42.1 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%A) {
// CHECK:STDOUT:   %T.loc117_13.2 => constants.%A
// CHECK:STDOUT:   %T.patt.loc117_13.2 => constants.%A
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type => constants.%Get.type.2
// CHECK:STDOUT:   %Get => constants.%Get.2
// CHECK:STDOUT:   %GetNoDeduce.type => constants.%GetNoDeduce.type.2
// CHECK:STDOUT:   %GetNoDeduce => constants.%GetNoDeduce.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%A, constants.%B) {
// CHECK:STDOUT:   %U.loc118_10.1 => constants.%B
// CHECK:STDOUT:   %U.patt.loc118_10.1 => constants.%B
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:   %.loc118_28.1 => constants.%.9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Get.type => constants.%Get.type.2
// CHECK:STDOUT:   %Get => constants.%Get.2
// CHECK:STDOUT:   %.loc118_39.3 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetNoDeduce(constants.%A, constants.%B) {
// CHECK:STDOUT:   %T => constants.%A
// CHECK:STDOUT:   %U.loc119_24.1 => constants.%B
// CHECK:STDOUT:   %U.patt.loc119_24.1 => constants.%B
// CHECK:STDOUT:   %.loc119_42.1 => constants.%.9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GetNoDeduce.type => constants.%GetNoDeduce.type.2
// CHECK:STDOUT:   %GetNoDeduce => constants.%GetNoDeduce.2
// CHECK:STDOUT:   %.loc119_53.3 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
