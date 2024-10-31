// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/expr_category.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/expr_category.carbon
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

fn F() -> [i32; 3];

fn G(b: [i32; 3]) {
  var a: [i32; 3] = (1, 2, 3);

  // Indexing a durable array reference gives a durable reference.
  var pa: i32* = &a[0];
  a[0] = 4;
}

fn ValueBinding(b: [i32; 3]) {
  var a: [i32; 3] = (1, 2, 3);

  // Index but don't do anything else so we can check that a value binding is
  // produced when appropriate.
  a[0];
  b[0];
  F()[0];
}

// CHECK:STDOUT: --- expr_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.5, %.6, %.2) [template]
// CHECK:STDOUT:   %.9: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %ValueBinding.type: type = fn_type @ValueBinding [template]
// CHECK:STDOUT:   %ValueBinding: %ValueBinding.type = struct_value () [template]
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
// CHECK:STDOUT:     .ValueBinding = %ValueBinding.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.3 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc114_17: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:     %.loc114_12.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc114_12.2: type = converted %int.make_type_32, %.loc114_12.1 [template = i32]
// CHECK:STDOUT:     %.loc114_18: type = array_type %.loc114_17, i32 [template = constants.%.3]
// CHECK:STDOUT:     %return.param: ref %.3 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %b.patt: %.3 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.3 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc116: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc116_15: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:     %.loc116_10.1: type = value_of_initializer %int.make_type_32.loc116 [template = i32]
// CHECK:STDOUT:     %.loc116_10.2: type = converted %int.make_type_32.loc116, %.loc116_10.1 [template = i32]
// CHECK:STDOUT:     %.loc116_16: type = array_type %.loc116_15, i32 [template = constants.%.3]
// CHECK:STDOUT:     %b.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %b: %.3 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ValueBinding.decl: %ValueBinding.type = fn_decl @ValueBinding [template = constants.%ValueBinding] {
// CHECK:STDOUT:     %b.patt: %.3 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.3 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc124: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc124_26: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:     %.loc124_21.1: type = value_of_initializer %int.make_type_32.loc124 [template = i32]
// CHECK:STDOUT:     %.loc124_21.2: type = converted %int.make_type_32.loc124, %.loc124_21.1 [template = i32]
// CHECK:STDOUT:     %.loc124_27: type = array_type %.loc124_26, i32 [template = constants.%.3]
// CHECK:STDOUT:     %b.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %b: %.3 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param_patt: %.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc117: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc117_16: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc117_11.1: type = value_of_initializer %int.make_type_32.loc117 [template = i32]
// CHECK:STDOUT:   %.loc117_11.2: type = converted %int.make_type_32.loc117, %.loc117_11.1 [template = i32]
// CHECK:STDOUT:   %.loc117_17: type = array_type %.loc117_16, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc117_22: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc117_25: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc117_28: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc117_29.1: %.7 = tuple_literal (%.loc117_22, %.loc117_25, %.loc117_28)
// CHECK:STDOUT:   %.loc117_29.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc117_29.3: ref i32 = array_index %a.var, %.loc117_29.2
// CHECK:STDOUT:   %.loc117_29.4: init i32 = initialize_from %.loc117_22 to %.loc117_29.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc117_29.5: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc117_29.6: ref i32 = array_index %a.var, %.loc117_29.5
// CHECK:STDOUT:   %.loc117_29.7: init i32 = initialize_from %.loc117_25 to %.loc117_29.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc117_29.8: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc117_29.9: ref i32 = array_index %a.var, %.loc117_29.8
// CHECK:STDOUT:   %.loc117_29.10: init i32 = initialize_from %.loc117_28 to %.loc117_29.9 [template = constants.%.2]
// CHECK:STDOUT:   %.loc117_29.11: init %.3 = array_init (%.loc117_29.4, %.loc117_29.7, %.loc117_29.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc117_30: init %.3 = converted %.loc117_29.1, %.loc117_29.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc117_30
// CHECK:STDOUT:   %int.make_type_32.loc120: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc120_14.1: type = value_of_initializer %int.make_type_32.loc120 [template = i32]
// CHECK:STDOUT:   %.loc120_14.2: type = converted %int.make_type_32.loc120, %.loc120_14.1 [template = i32]
// CHECK:STDOUT:   %.loc120_14.3: type = ptr_type i32 [template = constants.%.9]
// CHECK:STDOUT:   %pa.var: ref %.9 = var pa
// CHECK:STDOUT:   %pa: ref %.9 = bind_name pa, %pa.var
// CHECK:STDOUT:   %a.ref.loc120: ref %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc120_21: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc120_22: ref i32 = array_index %a.ref.loc120, %.loc120_21
// CHECK:STDOUT:   %.loc120_18: %.9 = addr_of %.loc120_22
// CHECK:STDOUT:   assign %pa.var, %.loc120_18
// CHECK:STDOUT:   %a.ref.loc121: ref %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc121_5: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc121_6: ref i32 = array_index %a.ref.loc121, %.loc121_5
// CHECK:STDOUT:   %.loc121_10: i32 = int_literal 4 [template = constants.%.10]
// CHECK:STDOUT:   assign %.loc121_6, %.loc121_10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ValueBinding(%b.param_patt: %.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc125: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc125_16: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc125_11.1: type = value_of_initializer %int.make_type_32.loc125 [template = i32]
// CHECK:STDOUT:   %.loc125_11.2: type = converted %int.make_type_32.loc125, %.loc125_11.1 [template = i32]
// CHECK:STDOUT:   %.loc125_17: type = array_type %.loc125_16, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc125_22: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc125_25: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc125_28: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc125_29.1: %.7 = tuple_literal (%.loc125_22, %.loc125_25, %.loc125_28)
// CHECK:STDOUT:   %.loc125_29.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc125_29.3: ref i32 = array_index %a.var, %.loc125_29.2
// CHECK:STDOUT:   %.loc125_29.4: init i32 = initialize_from %.loc125_22 to %.loc125_29.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc125_29.5: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc125_29.6: ref i32 = array_index %a.var, %.loc125_29.5
// CHECK:STDOUT:   %.loc125_29.7: init i32 = initialize_from %.loc125_25 to %.loc125_29.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc125_29.8: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc125_29.9: ref i32 = array_index %a.var, %.loc125_29.8
// CHECK:STDOUT:   %.loc125_29.10: init i32 = initialize_from %.loc125_28 to %.loc125_29.9 [template = constants.%.2]
// CHECK:STDOUT:   %.loc125_29.11: init %.3 = array_init (%.loc125_29.4, %.loc125_29.7, %.loc125_29.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc125_30: init %.3 = converted %.loc125_29.1, %.loc125_29.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc125_30
// CHECK:STDOUT:   %a.ref: ref %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc129_5: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc129_6: ref i32 = array_index %a.ref, %.loc129_5
// CHECK:STDOUT:   %b.ref: %.3 = name_ref b, %b
// CHECK:STDOUT:   %.loc130_5: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc130_6.1: ref %.3 = value_as_ref %b.ref
// CHECK:STDOUT:   %.loc130_6.2: ref i32 = array_index %.loc130_6.1, %.loc130_5
// CHECK:STDOUT:   %.loc130_6.3: i32 = bind_value %.loc130_6.2
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc131_4.1: ref %.3 = temporary_storage
// CHECK:STDOUT:   %F.call: init %.3 = call %F.ref() to %.loc131_4.1
// CHECK:STDOUT:   %.loc131_7: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc131_4.2: ref %.3 = temporary %.loc131_4.1, %F.call
// CHECK:STDOUT:   %.loc131_8.1: ref i32 = array_index %.loc131_4.2, %.loc131_7
// CHECK:STDOUT:   %.loc131_8.2: i32 = bind_value %.loc131_8.1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
