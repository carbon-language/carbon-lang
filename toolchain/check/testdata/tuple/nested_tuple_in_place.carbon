// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/nested_tuple_in_place.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/nested_tuple_in_place.carbon
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

fn F() -> (i32, i32, i32);

fn G() {
  var v: ((i32, i32, i32), (i32, i32, i32)) = (F(), F());
}

fn H() {
  var v: (i32, (i32, i32, i32), i32) = (1, F(), 2);
}

// CHECK:STDOUT: --- nested_tuple_in_place.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = tuple_type (%.2, %.2) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (%.3, %.3) [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (%.6, %.6) [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = tuple_type (type, %.2, type) [template]
// CHECK:STDOUT:   %.10: type = tuple_type (i32, %.3, i32) [template]
// CHECK:STDOUT:   %.11: type = tuple_type (i32, %.6, i32) [template]
// CHECK:STDOUT:   %.12: type = ptr_type %.11 [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 2 [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.3 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc114_12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc114_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc114_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc114_25.1: %.2 = tuple_literal (%int.make_type_32.loc114_12, %int.make_type_32.loc114_17, %int.make_type_32.loc114_22)
// CHECK:STDOUT:     %.loc114_25.2: type = value_of_initializer %int.make_type_32.loc114_12 [template = i32]
// CHECK:STDOUT:     %.loc114_25.3: type = converted %int.make_type_32.loc114_12, %.loc114_25.2 [template = i32]
// CHECK:STDOUT:     %.loc114_25.4: type = value_of_initializer %int.make_type_32.loc114_17 [template = i32]
// CHECK:STDOUT:     %.loc114_25.5: type = converted %int.make_type_32.loc114_17, %.loc114_25.4 [template = i32]
// CHECK:STDOUT:     %.loc114_25.6: type = value_of_initializer %int.make_type_32.loc114_22 [template = i32]
// CHECK:STDOUT:     %.loc114_25.7: type = converted %int.make_type_32.loc114_22, %.loc114_25.6 [template = i32]
// CHECK:STDOUT:     %.loc114_25.8: type = converted %.loc114_25.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %return.param: ref %.3 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc117_12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc117_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc117_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc117_25: %.2 = tuple_literal (%int.make_type_32.loc117_12, %int.make_type_32.loc117_17, %int.make_type_32.loc117_22)
// CHECK:STDOUT:   %int.make_type_32.loc117_29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc117_34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc117_39: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc117_42: %.2 = tuple_literal (%int.make_type_32.loc117_29, %int.make_type_32.loc117_34, %int.make_type_32.loc117_39)
// CHECK:STDOUT:   %.loc117_43.1: %.4 = tuple_literal (%.loc117_25, %.loc117_42)
// CHECK:STDOUT:   %.loc117_43.2: type = value_of_initializer %int.make_type_32.loc117_12 [template = i32]
// CHECK:STDOUT:   %.loc117_43.3: type = converted %int.make_type_32.loc117_12, %.loc117_43.2 [template = i32]
// CHECK:STDOUT:   %.loc117_43.4: type = value_of_initializer %int.make_type_32.loc117_17 [template = i32]
// CHECK:STDOUT:   %.loc117_43.5: type = converted %int.make_type_32.loc117_17, %.loc117_43.4 [template = i32]
// CHECK:STDOUT:   %.loc117_43.6: type = value_of_initializer %int.make_type_32.loc117_22 [template = i32]
// CHECK:STDOUT:   %.loc117_43.7: type = converted %int.make_type_32.loc117_22, %.loc117_43.6 [template = i32]
// CHECK:STDOUT:   %.loc117_43.8: type = converted %.loc117_25, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc117_43.9: type = value_of_initializer %int.make_type_32.loc117_29 [template = i32]
// CHECK:STDOUT:   %.loc117_43.10: type = converted %int.make_type_32.loc117_29, %.loc117_43.9 [template = i32]
// CHECK:STDOUT:   %.loc117_43.11: type = value_of_initializer %int.make_type_32.loc117_34 [template = i32]
// CHECK:STDOUT:   %.loc117_43.12: type = converted %int.make_type_32.loc117_34, %.loc117_43.11 [template = i32]
// CHECK:STDOUT:   %.loc117_43.13: type = value_of_initializer %int.make_type_32.loc117_39 [template = i32]
// CHECK:STDOUT:   %.loc117_43.14: type = converted %int.make_type_32.loc117_39, %.loc117_43.13 [template = i32]
// CHECK:STDOUT:   %.loc117_43.15: type = converted %.loc117_42, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc117_43.16: type = converted %.loc117_43.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:   %v.var: ref %.5 = var v
// CHECK:STDOUT:   %v: ref %.5 = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc117_48: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc117_56.1: ref %.3 = tuple_access %v.var, element0
// CHECK:STDOUT:   %F.call.loc117_49: init %.3 = call %F.ref.loc117_48() to %.loc117_56.1
// CHECK:STDOUT:   %F.ref.loc117_53: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc117_56.2: ref %.3 = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call.loc117_54: init %.3 = call %F.ref.loc117_53() to %.loc117_56.2
// CHECK:STDOUT:   %.loc117_56.3: %.5 = tuple_literal (%F.call.loc117_49, %F.call.loc117_54)
// CHECK:STDOUT:   %.loc117_56.4: init %.5 = tuple_init (%F.call.loc117_49, %F.call.loc117_54) to %v.var
// CHECK:STDOUT:   %.loc117_57: init %.5 = converted %.loc117_56.3, %.loc117_56.4
// CHECK:STDOUT:   assign %v.var, %.loc117_57
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc121_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc121_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc121_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc121_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc121_30: %.2 = tuple_literal (%int.make_type_32.loc121_17, %int.make_type_32.loc121_22, %int.make_type_32.loc121_27)
// CHECK:STDOUT:   %int.make_type_32.loc121_33: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc121_36.1: %.9 = tuple_literal (%int.make_type_32.loc121_11, %.loc121_30, %int.make_type_32.loc121_33)
// CHECK:STDOUT:   %.loc121_36.2: type = value_of_initializer %int.make_type_32.loc121_11 [template = i32]
// CHECK:STDOUT:   %.loc121_36.3: type = converted %int.make_type_32.loc121_11, %.loc121_36.2 [template = i32]
// CHECK:STDOUT:   %.loc121_36.4: type = value_of_initializer %int.make_type_32.loc121_17 [template = i32]
// CHECK:STDOUT:   %.loc121_36.5: type = converted %int.make_type_32.loc121_17, %.loc121_36.4 [template = i32]
// CHECK:STDOUT:   %.loc121_36.6: type = value_of_initializer %int.make_type_32.loc121_22 [template = i32]
// CHECK:STDOUT:   %.loc121_36.7: type = converted %int.make_type_32.loc121_22, %.loc121_36.6 [template = i32]
// CHECK:STDOUT:   %.loc121_36.8: type = value_of_initializer %int.make_type_32.loc121_27 [template = i32]
// CHECK:STDOUT:   %.loc121_36.9: type = converted %int.make_type_32.loc121_27, %.loc121_36.8 [template = i32]
// CHECK:STDOUT:   %.loc121_36.10: type = converted %.loc121_30, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc121_36.11: type = value_of_initializer %int.make_type_32.loc121_33 [template = i32]
// CHECK:STDOUT:   %.loc121_36.12: type = converted %int.make_type_32.loc121_33, %.loc121_36.11 [template = i32]
// CHECK:STDOUT:   %.loc121_36.13: type = converted %.loc121_36.1, constants.%.10 [template = constants.%.10]
// CHECK:STDOUT:   %v.var: ref %.10 = var v
// CHECK:STDOUT:   %v: ref %.10 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc121_41: i32 = int_literal 1 [template = constants.%.13]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc121_50.1: ref %.3 = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call: init %.3 = call %F.ref() to %.loc121_50.1
// CHECK:STDOUT:   %.loc121_49: i32 = int_literal 2 [template = constants.%.14]
// CHECK:STDOUT:   %.loc121_50.2: %.10 = tuple_literal (%.loc121_41, %F.call, %.loc121_49)
// CHECK:STDOUT:   %.loc121_50.3: ref i32 = tuple_access %v.var, element0
// CHECK:STDOUT:   %.loc121_50.4: init i32 = initialize_from %.loc121_41 to %.loc121_50.3 [template = constants.%.13]
// CHECK:STDOUT:   %.loc121_50.5: ref i32 = tuple_access %v.var, element2
// CHECK:STDOUT:   %.loc121_50.6: init i32 = initialize_from %.loc121_49 to %.loc121_50.5 [template = constants.%.14]
// CHECK:STDOUT:   %.loc121_50.7: init %.10 = tuple_init (%.loc121_50.4, %F.call, %.loc121_50.6) to %v.var
// CHECK:STDOUT:   %.loc121_51: init %.10 = converted %.loc121_50.2, %.loc121_50.7
// CHECK:STDOUT:   assign %v.var, %.loc121_51
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
