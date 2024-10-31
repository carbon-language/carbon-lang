// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/numeric_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/numeric_literals.carbon
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

fn F() {
  // 8 and 9 trigger special behavior in APInt when mishandling signed versus
  // unsigned, so we pay extra attention to those.
  var ints: [i32; 6] = (
    8,
    9,
    0x8,
    0b1000,
    2147483647,
    0x7FFFFFFF,
  );
  var floats: [f64; 6] = (
    0.9,
    8.0,
    80.0,
    1.0e7,
    1.0e8,
    1.0e-8
  );
}

// CHECK:STDOUT: --- numeric_literals.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 8 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 9 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 2147483647 [template]
// CHECK:STDOUT:   %.8: type = tuple_type (i32, i32, i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.12: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %array.1: %.3 = tuple_value (%.5, %.6, %.5, %.5, %.7, %.7) [template]
// CHECK:STDOUT:   %.15: Core.BigInt = int_literal 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %.16: type = array_type %.2, f64 [template]
// CHECK:STDOUT:   %.17: type = ptr_type %.16 [template]
// CHECK:STDOUT:   %.18: f64 = float_literal 0.90000000000000002 [template]
// CHECK:STDOUT:   %.19: f64 = float_literal 8 [template]
// CHECK:STDOUT:   %.20: f64 = float_literal 80 [template]
// CHECK:STDOUT:   %.21: f64 = float_literal 1.0E+7 [template]
// CHECK:STDOUT:   %.22: f64 = float_literal 1.0E+8 [template]
// CHECK:STDOUT:   %.23: f64 = float_literal 1.0E-8 [template]
// CHECK:STDOUT:   %.24: type = tuple_type (f64, f64, f64, f64, f64, f64) [template]
// CHECK:STDOUT:   %array.2: %.16 = tuple_value (%.18, %.19, %.20, %.21, %.22, %.23) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .Float = %import_ref.2
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
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Float.type = import_ref Core//prelude/types, inst+60, loaded [template = constants.%Float]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc117_19: i32 = int_literal 6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc117_14.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc117_14.2: type = converted %int.make_type_32, %.loc117_14.1 [template = i32]
// CHECK:STDOUT:   %.loc117_20: type = array_type %.loc117_19, i32 [template = constants.%.3]
// CHECK:STDOUT:   %ints.var: ref %.3 = var ints
// CHECK:STDOUT:   %ints: ref %.3 = bind_name ints, %ints.var
// CHECK:STDOUT:   %.loc118: i32 = int_literal 8 [template = constants.%.5]
// CHECK:STDOUT:   %.loc119: i32 = int_literal 9 [template = constants.%.6]
// CHECK:STDOUT:   %.loc120: i32 = int_literal 8 [template = constants.%.5]
// CHECK:STDOUT:   %.loc121: i32 = int_literal 8 [template = constants.%.5]
// CHECK:STDOUT:   %.loc122: i32 = int_literal 2147483647 [template = constants.%.7]
// CHECK:STDOUT:   %.loc123: i32 = int_literal 2147483647 [template = constants.%.7]
// CHECK:STDOUT:   %.loc124_3.1: %.8 = tuple_literal (%.loc118, %.loc119, %.loc120, %.loc121, %.loc122, %.loc123)
// CHECK:STDOUT:   %.loc124_3.2: i32 = int_literal 0 [template = constants.%.9]
// CHECK:STDOUT:   %.loc124_3.3: ref i32 = array_index %ints.var, %.loc124_3.2
// CHECK:STDOUT:   %.loc124_3.4: init i32 = initialize_from %.loc118 to %.loc124_3.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc124_3.5: i32 = int_literal 1 [template = constants.%.10]
// CHECK:STDOUT:   %.loc124_3.6: ref i32 = array_index %ints.var, %.loc124_3.5
// CHECK:STDOUT:   %.loc124_3.7: init i32 = initialize_from %.loc119 to %.loc124_3.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc124_3.8: i32 = int_literal 2 [template = constants.%.11]
// CHECK:STDOUT:   %.loc124_3.9: ref i32 = array_index %ints.var, %.loc124_3.8
// CHECK:STDOUT:   %.loc124_3.10: init i32 = initialize_from %.loc120 to %.loc124_3.9 [template = constants.%.5]
// CHECK:STDOUT:   %.loc124_3.11: i32 = int_literal 3 [template = constants.%.12]
// CHECK:STDOUT:   %.loc124_3.12: ref i32 = array_index %ints.var, %.loc124_3.11
// CHECK:STDOUT:   %.loc124_3.13: init i32 = initialize_from %.loc121 to %.loc124_3.12 [template = constants.%.5]
// CHECK:STDOUT:   %.loc124_3.14: i32 = int_literal 4 [template = constants.%.13]
// CHECK:STDOUT:   %.loc124_3.15: ref i32 = array_index %ints.var, %.loc124_3.14
// CHECK:STDOUT:   %.loc124_3.16: init i32 = initialize_from %.loc122 to %.loc124_3.15 [template = constants.%.7]
// CHECK:STDOUT:   %.loc124_3.17: i32 = int_literal 5 [template = constants.%.14]
// CHECK:STDOUT:   %.loc124_3.18: ref i32 = array_index %ints.var, %.loc124_3.17
// CHECK:STDOUT:   %.loc124_3.19: init i32 = initialize_from %.loc123 to %.loc124_3.18 [template = constants.%.7]
// CHECK:STDOUT:   %.loc124_3.20: init %.3 = array_init (%.loc124_3.4, %.loc124_3.7, %.loc124_3.10, %.loc124_3.13, %.loc124_3.16, %.loc124_3.19) to %ints.var [template = constants.%array.1]
// CHECK:STDOUT:   %.loc124_4: init %.3 = converted %.loc124_3.1, %.loc124_3.20 [template = constants.%array.1]
// CHECK:STDOUT:   assign %ints.var, %.loc124_4
// CHECK:STDOUT:   %.loc125_16.1: Core.BigInt = int_literal 64 [template = constants.%.15]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%.loc125_16.1) [template = f64]
// CHECK:STDOUT:   %.loc125_21: i32 = int_literal 6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc125_16.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc125_16.3: type = converted %float.make_type, %.loc125_16.2 [template = f64]
// CHECK:STDOUT:   %.loc125_22: type = array_type %.loc125_21, f64 [template = constants.%.16]
// CHECK:STDOUT:   %floats.var: ref %.16 = var floats
// CHECK:STDOUT:   %floats: ref %.16 = bind_name floats, %floats.var
// CHECK:STDOUT:   %.loc126: f64 = float_literal 0.90000000000000002 [template = constants.%.18]
// CHECK:STDOUT:   %.loc127: f64 = float_literal 8 [template = constants.%.19]
// CHECK:STDOUT:   %.loc128: f64 = float_literal 80 [template = constants.%.20]
// CHECK:STDOUT:   %.loc129: f64 = float_literal 1.0E+7 [template = constants.%.21]
// CHECK:STDOUT:   %.loc130: f64 = float_literal 1.0E+8 [template = constants.%.22]
// CHECK:STDOUT:   %.loc131: f64 = float_literal 1.0E-8 [template = constants.%.23]
// CHECK:STDOUT:   %.loc132_3.1: %.24 = tuple_literal (%.loc126, %.loc127, %.loc128, %.loc129, %.loc130, %.loc131)
// CHECK:STDOUT:   %.loc132_3.2: i32 = int_literal 0 [template = constants.%.9]
// CHECK:STDOUT:   %.loc132_3.3: ref f64 = array_index %floats.var, %.loc132_3.2
// CHECK:STDOUT:   %.loc132_3.4: init f64 = initialize_from %.loc126 to %.loc132_3.3 [template = constants.%.18]
// CHECK:STDOUT:   %.loc132_3.5: i32 = int_literal 1 [template = constants.%.10]
// CHECK:STDOUT:   %.loc132_3.6: ref f64 = array_index %floats.var, %.loc132_3.5
// CHECK:STDOUT:   %.loc132_3.7: init f64 = initialize_from %.loc127 to %.loc132_3.6 [template = constants.%.19]
// CHECK:STDOUT:   %.loc132_3.8: i32 = int_literal 2 [template = constants.%.11]
// CHECK:STDOUT:   %.loc132_3.9: ref f64 = array_index %floats.var, %.loc132_3.8
// CHECK:STDOUT:   %.loc132_3.10: init f64 = initialize_from %.loc128 to %.loc132_3.9 [template = constants.%.20]
// CHECK:STDOUT:   %.loc132_3.11: i32 = int_literal 3 [template = constants.%.12]
// CHECK:STDOUT:   %.loc132_3.12: ref f64 = array_index %floats.var, %.loc132_3.11
// CHECK:STDOUT:   %.loc132_3.13: init f64 = initialize_from %.loc129 to %.loc132_3.12 [template = constants.%.21]
// CHECK:STDOUT:   %.loc132_3.14: i32 = int_literal 4 [template = constants.%.13]
// CHECK:STDOUT:   %.loc132_3.15: ref f64 = array_index %floats.var, %.loc132_3.14
// CHECK:STDOUT:   %.loc132_3.16: init f64 = initialize_from %.loc130 to %.loc132_3.15 [template = constants.%.22]
// CHECK:STDOUT:   %.loc132_3.17: i32 = int_literal 5 [template = constants.%.14]
// CHECK:STDOUT:   %.loc132_3.18: ref f64 = array_index %floats.var, %.loc132_3.17
// CHECK:STDOUT:   %.loc132_3.19: init f64 = initialize_from %.loc131 to %.loc132_3.18 [template = constants.%.23]
// CHECK:STDOUT:   %.loc132_3.20: init %.16 = array_init (%.loc132_3.4, %.loc132_3.7, %.loc132_3.10, %.loc132_3.13, %.loc132_3.16, %.loc132_3.19) to %floats.var [template = constants.%array.2]
// CHECK:STDOUT:   %.loc132_4: init %.16 = converted %.loc132_3.1, %.loc132_3.20 [template = constants.%array.2]
// CHECK:STDOUT:   assign %floats.var, %.loc132_4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size.param_patt: Core.BigInt) -> type = "float.make_type";
// CHECK:STDOUT:
