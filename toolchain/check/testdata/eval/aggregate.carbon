// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/aggregate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/aggregate.carbon
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

var tuple_copy: (i32, i32) = (1, 2) as (i32, i32);

var struct_copy: {.a: i32, .b: i32, .c: i32} = {.c = 3, .b = 2, .a = 1} as {.b: i32, .a: i32, .c: i32};

var tuple_index: [i32; 1] = (0,) as [i32; (5, 7, 1, 9).2];

var struct_access: [i32; 1] = (0,) as [i32; {.a = 3, .b = 1}.b];

// CHECK:STDOUT: --- aggregate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %tuple.1: %.3 = tuple_value (%.5, %.6) [template]
// CHECK:STDOUT:   %.7: type = struct_type {.a: i32, .b: i32, .c: i32} [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.10: type = struct_type {.c: i32, .b: i32, .a: i32} [template]
// CHECK:STDOUT:   %.11: type = struct_type {.b: i32, .a: i32, .c: i32} [template]
// CHECK:STDOUT:   %.12: type = ptr_type %.11 [template]
// CHECK:STDOUT:   %struct.1: %.11 = struct_value (%.6, %.5, %.9) [template]
// CHECK:STDOUT:   %struct.2: %.7 = struct_value (%.5, %.6, %.9) [template]
// CHECK:STDOUT:   %.13: type = array_type %.5, i32 [template]
// CHECK:STDOUT:   %.14: type = ptr_type %.13 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.16: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.17: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.18: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.19: i32 = int_literal 9 [template]
// CHECK:STDOUT:   %.20: type = tuple_type (i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.21: type = ptr_type %.20 [template]
// CHECK:STDOUT:   %tuple.2: %.20 = tuple_value (%.17, %.18, %.5, %.19) [template]
// CHECK:STDOUT:   %array: %.13 = tuple_value (%.15) [template]
// CHECK:STDOUT:   %.22: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.23: type = ptr_type %.22 [template]
// CHECK:STDOUT:   %struct.3: %.22 = struct_value (%.9, %.5) [template]
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
// CHECK:STDOUT:     .tuple_copy = %tuple_copy
// CHECK:STDOUT:     .struct_copy = %struct_copy
// CHECK:STDOUT:     .tuple_index = %tuple_index
// CHECK:STDOUT:     .struct_access = %struct_access
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc114_18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc114_23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc114_26.1: %.2 = tuple_literal (%int.make_type_32.loc114_18, %int.make_type_32.loc114_23)
// CHECK:STDOUT:   %.loc114_26.2: type = value_of_initializer %int.make_type_32.loc114_18 [template = i32]
// CHECK:STDOUT:   %.loc114_26.3: type = converted %int.make_type_32.loc114_18, %.loc114_26.2 [template = i32]
// CHECK:STDOUT:   %.loc114_26.4: type = value_of_initializer %int.make_type_32.loc114_23 [template = i32]
// CHECK:STDOUT:   %.loc114_26.5: type = converted %int.make_type_32.loc114_23, %.loc114_26.4 [template = i32]
// CHECK:STDOUT:   %.loc114_26.6: type = converted %.loc114_26.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %tuple_copy.var: ref %.3 = var tuple_copy
// CHECK:STDOUT:   %tuple_copy: ref %.3 = bind_name tuple_copy, %tuple_copy.var
// CHECK:STDOUT:   %int.make_type_32.loc116_23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc116_23.1: type = value_of_initializer %int.make_type_32.loc116_23 [template = i32]
// CHECK:STDOUT:   %.loc116_23.2: type = converted %int.make_type_32.loc116_23, %.loc116_23.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc116_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc116_32.1: type = value_of_initializer %int.make_type_32.loc116_32 [template = i32]
// CHECK:STDOUT:   %.loc116_32.2: type = converted %int.make_type_32.loc116_32, %.loc116_32.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc116_41: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc116_41.1: type = value_of_initializer %int.make_type_32.loc116_41 [template = i32]
// CHECK:STDOUT:   %.loc116_41.2: type = converted %int.make_type_32.loc116_41, %.loc116_41.1 [template = i32]
// CHECK:STDOUT:   %.loc116_44: type = struct_type {.a: i32, .b: i32, .c: i32} [template = constants.%.7]
// CHECK:STDOUT:   %struct_copy.var: ref %.7 = var struct_copy
// CHECK:STDOUT:   %struct_copy: ref %.7 = bind_name struct_copy, %struct_copy.var
// CHECK:STDOUT:   %int.make_type_32.loc118: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc118_24: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc118_19.1: type = value_of_initializer %int.make_type_32.loc118 [template = i32]
// CHECK:STDOUT:   %.loc118_19.2: type = converted %int.make_type_32.loc118, %.loc118_19.1 [template = i32]
// CHECK:STDOUT:   %.loc118_25: type = array_type %.loc118_24, i32 [template = constants.%.13]
// CHECK:STDOUT:   %tuple_index.var: ref %.13 = var tuple_index
// CHECK:STDOUT:   %tuple_index: ref %.13 = bind_name tuple_index, %tuple_index.var
// CHECK:STDOUT:   %int.make_type_32.loc120: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc120_26: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc120_21.1: type = value_of_initializer %int.make_type_32.loc120 [template = i32]
// CHECK:STDOUT:   %.loc120_21.2: type = converted %int.make_type_32.loc120, %.loc120_21.1 [template = i32]
// CHECK:STDOUT:   %.loc120_27: type = array_type %.loc120_26, i32 [template = constants.%.13]
// CHECK:STDOUT:   %struct_access.var: ref %.13 = var struct_access
// CHECK:STDOUT:   %struct_access: ref %.13 = bind_name struct_access, %struct_access.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc114_31: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc114_34: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc114_35.1: %.3 = tuple_literal (%.loc114_31, %.loc114_34)
// CHECK:STDOUT:   %int.make_type_32.loc114_41: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc114_46: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc114_49.1: %.2 = tuple_literal (%int.make_type_32.loc114_41, %int.make_type_32.loc114_46)
// CHECK:STDOUT:   %.loc114_49.2: type = value_of_initializer %int.make_type_32.loc114_41 [template = i32]
// CHECK:STDOUT:   %.loc114_49.3: type = converted %int.make_type_32.loc114_41, %.loc114_49.2 [template = i32]
// CHECK:STDOUT:   %.loc114_49.4: type = value_of_initializer %int.make_type_32.loc114_46 [template = i32]
// CHECK:STDOUT:   %.loc114_49.5: type = converted %int.make_type_32.loc114_46, %.loc114_49.4 [template = i32]
// CHECK:STDOUT:   %.loc114_49.6: type = converted %.loc114_49.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc114_35.2: ref i32 = tuple_access file.%tuple_copy.var, element0
// CHECK:STDOUT:   %.loc114_35.3: init i32 = initialize_from %.loc114_31 to %.loc114_35.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc114_35.4: ref i32 = tuple_access file.%tuple_copy.var, element1
// CHECK:STDOUT:   %.loc114_35.5: init i32 = initialize_from %.loc114_34 to %.loc114_35.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc114_35.6: init %.3 = tuple_init (%.loc114_35.3, %.loc114_35.5) to file.%tuple_copy.var [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc114_50: init %.3 = converted %.loc114_35.1, %.loc114_35.6 [template = constants.%tuple.1]
// CHECK:STDOUT:   assign file.%tuple_copy.var, %.loc114_50
// CHECK:STDOUT:   %.loc116_54: i32 = int_literal 3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc116_62: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc116_70: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc116_71: %.10 = struct_literal (%.loc116_54, %.loc116_62, %.loc116_70)
// CHECK:STDOUT:   %int.make_type_32.loc116_81: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc116_81.1: type = value_of_initializer %int.make_type_32.loc116_81 [template = i32]
// CHECK:STDOUT:   %.loc116_81.2: type = converted %int.make_type_32.loc116_81, %.loc116_81.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc116_90: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc116_90.1: type = value_of_initializer %int.make_type_32.loc116_90 [template = i32]
// CHECK:STDOUT:   %.loc116_90.2: type = converted %int.make_type_32.loc116_90, %.loc116_90.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc116_99: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc116_99.1: type = value_of_initializer %int.make_type_32.loc116_99 [template = i32]
// CHECK:STDOUT:   %.loc116_99.2: type = converted %int.make_type_32.loc116_99, %.loc116_99.1 [template = i32]
// CHECK:STDOUT:   %.loc116_102: type = struct_type {.b: i32, .a: i32, .c: i32} [template = constants.%.11]
// CHECK:STDOUT:   %struct.loc116: %.11 = struct_value (%.loc116_62, %.loc116_70, %.loc116_54) [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc116_73.1: %.11 = converted %.loc116_71, %struct.loc116 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc116_73.2: i32 = struct_access %.loc116_73.1, element1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc116_73.3: ref i32 = struct_access file.%struct_copy.var, element1
// CHECK:STDOUT:   %.loc116_73.4: init i32 = initialize_from %.loc116_73.2 to %.loc116_73.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc116_73.5: i32 = struct_access %.loc116_73.1, element0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc116_73.6: ref i32 = struct_access file.%struct_copy.var, element0
// CHECK:STDOUT:   %.loc116_73.7: init i32 = initialize_from %.loc116_73.5 to %.loc116_73.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc116_73.8: i32 = struct_access %.loc116_73.1, element2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc116_73.9: ref i32 = struct_access file.%struct_copy.var, element2
// CHECK:STDOUT:   %.loc116_73.10: init i32 = initialize_from %.loc116_73.8 to %.loc116_73.9 [template = constants.%.9]
// CHECK:STDOUT:   %.loc116_73.11: init %.7 = struct_init (%.loc116_73.4, %.loc116_73.7, %.loc116_73.10) to file.%struct_copy.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc116_103: init %.7 = converted %.loc116_73.1, %.loc116_73.11 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%struct_copy.var, %.loc116_103
// CHECK:STDOUT:   %.loc118_30: i32 = int_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %.loc118_32.1: %.16 = tuple_literal (%.loc118_30)
// CHECK:STDOUT:   %int.make_type_32.loc118: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc118_44: i32 = int_literal 5 [template = constants.%.17]
// CHECK:STDOUT:   %.loc118_47: i32 = int_literal 7 [template = constants.%.18]
// CHECK:STDOUT:   %.loc118_50: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc118_53: i32 = int_literal 9 [template = constants.%.19]
// CHECK:STDOUT:   %.loc118_54.1: %.20 = tuple_literal (%.loc118_44, %.loc118_47, %.loc118_50, %.loc118_53)
// CHECK:STDOUT:   %.loc118_56: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %tuple: %.20 = tuple_value (%.loc118_44, %.loc118_47, %.loc118_50, %.loc118_53) [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc118_54.2: %.20 = converted %.loc118_54.1, %tuple [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc118_55: i32 = tuple_access %.loc118_54.2, element2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc118_38.1: type = value_of_initializer %int.make_type_32.loc118 [template = i32]
// CHECK:STDOUT:   %.loc118_38.2: type = converted %int.make_type_32.loc118, %.loc118_38.1 [template = i32]
// CHECK:STDOUT:   %.loc118_57: type = array_type %.loc118_55, i32 [template = constants.%.13]
// CHECK:STDOUT:   %.loc118_5: ref %.13 = splice_block file.%tuple_index.var {}
// CHECK:STDOUT:   %.loc118_32.2: i32 = int_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %.loc118_32.3: ref i32 = array_index %.loc118_5, %.loc118_32.2
// CHECK:STDOUT:   %.loc118_32.4: init i32 = initialize_from %.loc118_30 to %.loc118_32.3 [template = constants.%.15]
// CHECK:STDOUT:   %.loc118_32.5: init %.13 = array_init (%.loc118_32.4) to %.loc118_5 [template = constants.%array]
// CHECK:STDOUT:   %.loc118_34: init %.13 = converted %.loc118_32.1, %.loc118_32.5 [template = constants.%array]
// CHECK:STDOUT:   assign file.%tuple_index.var, %.loc118_34
// CHECK:STDOUT:   %.loc120_32: i32 = int_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %.loc120_34.1: %.16 = tuple_literal (%.loc120_32)
// CHECK:STDOUT:   %int.make_type_32.loc120: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc120_51: i32 = int_literal 3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc120_59: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc120_60.1: %.22 = struct_literal (%.loc120_51, %.loc120_59)
// CHECK:STDOUT:   %struct.loc120: %.22 = struct_value (%.loc120_51, %.loc120_59) [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc120_60.2: %.22 = converted %.loc120_60.1, %struct.loc120 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc120_61: i32 = struct_access %.loc120_60.2, element1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc120_40.1: type = value_of_initializer %int.make_type_32.loc120 [template = i32]
// CHECK:STDOUT:   %.loc120_40.2: type = converted %int.make_type_32.loc120, %.loc120_40.1 [template = i32]
// CHECK:STDOUT:   %.loc120_63: type = array_type %.loc120_61, i32 [template = constants.%.13]
// CHECK:STDOUT:   %.loc120_5: ref %.13 = splice_block file.%struct_access.var {}
// CHECK:STDOUT:   %.loc120_34.2: i32 = int_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %.loc120_34.3: ref i32 = array_index %.loc120_5, %.loc120_34.2
// CHECK:STDOUT:   %.loc120_34.4: init i32 = initialize_from %.loc120_32 to %.loc120_34.3 [template = constants.%.15]
// CHECK:STDOUT:   %.loc120_34.5: init %.13 = array_init (%.loc120_34.4) to %.loc120_5 [template = constants.%array]
// CHECK:STDOUT:   %.loc120_36: init %.13 = converted %.loc120_34.1, %.loc120_34.5 [template = constants.%array]
// CHECK:STDOUT:   assign file.%struct_access.var, %.loc120_36
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
