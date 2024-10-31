// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/field.carbon
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

class Class(T:! type) {
  var x: T;
}

fn F(c: Class(i32)) -> i32 {
  return c.x;
}

fn G(T:! type, c: Class(T)) -> T {
  return c.x;
}

fn H(U:! type, c: Class(U)) -> U {
  return c.x;
}

// CHECK:STDOUT: --- field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Class.3, i32 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.x: i32} [template]
// CHECK:STDOUT:   %.7: <witness> = complete_type_witness %.6 [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.6 [template]
// CHECK:STDOUT:   %T.patt.2: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %Class.4: type = class_type @Class, @Class(%U) [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %.10: type = unbound_element_type %Class.4, %U [symbolic]
// CHECK:STDOUT:   %.11: type = struct_type {.x: %U} [symbolic]
// CHECK:STDOUT:   %.12: <witness> = complete_type_witness %.11 [symbolic]
// CHECK:STDOUT:   %.13: type = ptr_type %.11 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
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
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.patt.loc114_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc114_13.2 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc114_13.1, runtime_param<invalid> [symbolic = %T.patt.loc114_13.2 (constants.%T.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc114_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc114_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %c.patt: %Class.3 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %Class.3 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %int.make_type_32.loc118_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc118_14.1: type = value_of_initializer %int.make_type_32.loc118_15 [template = i32]
// CHECK:STDOUT:     %.loc118_14.2: type = converted %int.make_type_32.loc118_15, %.loc118_14.1 [template = i32]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(i32) [template = constants.%Class.3]
// CHECK:STDOUT:     %int.make_type_32.loc118_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc118_24.1: type = value_of_initializer %int.make_type_32.loc118_24 [template = i32]
// CHECK:STDOUT:     %.loc118_24.2: type = converted %int.make_type_32.loc118_24, %.loc118_24.1 [template = i32]
// CHECK:STDOUT:     %c.param: %Class.3 = value_param runtime_param0
// CHECK:STDOUT:     %c: %Class.3 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc122_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc122_6.2 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc122_6.1, runtime_param<invalid> [symbolic = %T.patt.loc122_6.2 (constants.%T.patt.2)]
// CHECK:STDOUT:     %c.patt: @G.%Class.loc122_24.2 (%Class.2) = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: @G.%Class.loc122_24.2 (%Class.2) = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @G.%T.loc122_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @G.%T.loc122_6.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %T.ref.loc122_25: type = name_ref T, %T.loc122_6.1 [symbolic = %T.loc122_6.2 (constants.%T)]
// CHECK:STDOUT:     %Class.loc122_24.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc122_24.2 (constants.%Class.2)]
// CHECK:STDOUT:     %T.ref.loc122_32: type = name_ref T, %T.loc122_6.1 [symbolic = %T.loc122_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc122_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc122_6.2 (constants.%T)]
// CHECK:STDOUT:     %c.param: @G.%Class.loc122_24.2 (%Class.2) = value_param runtime_param0
// CHECK:STDOUT:     %c: @G.%Class.loc122_24.2 (%Class.2) = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref @G.%T.loc122_6.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @G.%T.loc122_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %U.patt.loc126_6.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc126_6.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc126_6.1, runtime_param<invalid> [symbolic = %U.patt.loc126_6.2 (constants.%U.patt)]
// CHECK:STDOUT:     %c.patt: @H.%Class.loc126_24.2 (%Class.4) = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: @H.%Class.loc126_24.2 (%Class.4) = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @H.%U.loc126_6.2 (%U) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @H.%U.loc126_6.2 (%U) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %U.ref.loc126_25: type = name_ref U, %U.loc126_6.1 [symbolic = %U.loc126_6.2 (constants.%U)]
// CHECK:STDOUT:     %Class.loc126_24.1: type = class_type @Class, @Class(constants.%U) [symbolic = %Class.loc126_24.2 (constants.%Class.4)]
// CHECK:STDOUT:     %U.ref.loc126_32: type = name_ref U, %U.loc126_6.1 [symbolic = %U.loc126_6.2 (constants.%U)]
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc126_6.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc126_6.2 (constants.%U)]
// CHECK:STDOUT:     %c.param: @H.%Class.loc126_24.2 (%Class.4) = value_param runtime_param0
// CHECK:STDOUT:     %c: @H.%Class.loc126_24.2 (%Class.4) = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref @H.%U.loc126_6.2 (%U) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @H.%U.loc126_6.2 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc114_13.1: type) {
// CHECK:STDOUT:   %T.loc114_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc114_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc114_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc114_13.2 (constants.%T.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc114_13.2) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.loc115_8.2: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T.loc114_13.2 (%T) [symbolic = %.loc115_8.2 (constants.%.2)]
// CHECK:STDOUT:   %.loc116_1.2: type = struct_type {.x: @Class.%T.loc114_13.2 (%T)} [symbolic = %.loc116_1.2 (constants.%.3)]
// CHECK:STDOUT:   %.loc116_1.3: <witness> = complete_type_witness @Class.%.loc116_1.2 (%.3) [symbolic = %.loc116_1.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc114_13.1 [symbolic = %T.loc114_13.2 (constants.%T)]
// CHECK:STDOUT:     %.loc115_8.1: @Class.%.loc115_8.2 (%.2) = field_decl x, element0 [template]
// CHECK:STDOUT:     %.loc116_1.1: <witness> = complete_type_witness %.3 [symbolic = %.loc116_1.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .x = %.loc115_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param_patt: %Class.3) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.3 = name_ref c, %c
// CHECK:STDOUT:   %x.ref: %.5 = name_ref x, @Class.%.loc115_8.1 [template = @Class.%.loc115_8.1]
// CHECK:STDOUT:   %.loc119_11.1: ref i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc119_11.2: i32 = bind_value %.loc119_11.1
// CHECK:STDOUT:   return %.loc119_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc122_6.1: type) {
// CHECK:STDOUT:   %T.loc122_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc122_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc122_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc122_6.2 (constants.%T.patt.2)]
// CHECK:STDOUT:   %Class.loc122_24.2: type = class_type @Class, @Class(%T.loc122_6.2) [symbolic = %Class.loc122_24.2 (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc123_11.3: type = unbound_element_type @G.%Class.loc122_24.2 (%Class.2), @G.%T.loc122_6.2 (%T) [symbolic = %.loc123_11.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %c.param_patt: @G.%Class.loc122_24.2 (%Class.2)) -> @G.%T.loc122_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @G.%Class.loc122_24.2 (%Class.2) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @G.%.loc123_11.3 (%.2) = name_ref x, @Class.%.loc115_8.1 [template = @Class.%.loc115_8.1]
// CHECK:STDOUT:     %.loc123_11.1: ref @G.%T.loc122_6.2 (%T) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc123_11.2: @G.%T.loc122_6.2 (%T) = bind_value %.loc123_11.1
// CHECK:STDOUT:     return %.loc123_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%U.loc126_6.1: type) {
// CHECK:STDOUT:   %U.loc126_6.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc126_6.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc126_6.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc126_6.2 (constants.%U.patt)]
// CHECK:STDOUT:   %Class.loc126_24.2: type = class_type @Class, @Class(%U.loc126_6.2) [symbolic = %Class.loc126_24.2 (constants.%Class.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc127_11.3: type = unbound_element_type @H.%Class.loc126_24.2 (%Class.4), @H.%U.loc126_6.2 (%U) [symbolic = %.loc127_11.3 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param_patt: type, %c.param_patt: @H.%Class.loc126_24.2 (%Class.4)) -> @H.%U.loc126_6.2 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @H.%Class.loc126_24.2 (%Class.4) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @H.%.loc127_11.3 (%.10) = name_ref x, @Class.%.loc115_8.1 [template = @Class.%.loc115_8.1]
// CHECK:STDOUT:     %.loc127_11.1: ref @H.%U.loc126_6.2 (%U) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc127_11.2: @H.%U.loc126_6.2 (%U) = bind_value %.loc127_11.1
// CHECK:STDOUT:     return %.loc127_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc114_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc114_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.loc115_8.2 => constants.%.2
// CHECK:STDOUT:   %.loc116_1.2 => constants.%.3
// CHECK:STDOUT:   %.loc116_1.3 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Class.%T.loc114_13.2) {
// CHECK:STDOUT:   %T.loc114_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc114_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(i32) {
// CHECK:STDOUT:   %T.loc114_13.2 => i32
// CHECK:STDOUT:   %T.patt.loc114_13.2 => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.3
// CHECK:STDOUT:   %.loc115_8.2 => constants.%.5
// CHECK:STDOUT:   %.loc116_1.2 => constants.%.6
// CHECK:STDOUT:   %.loc116_1.3 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@G.%T.loc122_6.2) {
// CHECK:STDOUT:   %T.loc114_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc114_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc122_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc122_6.2 => constants.%T
// CHECK:STDOUT:   %Class.loc122_24.2 => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%U) {
// CHECK:STDOUT:   %T.loc114_13.2 => constants.%U
// CHECK:STDOUT:   %T.patt.loc114_13.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.4
// CHECK:STDOUT:   %.loc115_8.2 => constants.%.10
// CHECK:STDOUT:   %.loc116_1.2 => constants.%.11
// CHECK:STDOUT:   %.loc116_1.3 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@H.%U.loc126_6.2) {
// CHECK:STDOUT:   %T.loc114_13.2 => constants.%U
// CHECK:STDOUT:   %T.patt.loc114_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%U) {
// CHECK:STDOUT:   %U.loc126_6.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc126_6.2 => constants.%U
// CHECK:STDOUT:   %Class.loc126_24.2 => constants.%Class.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
