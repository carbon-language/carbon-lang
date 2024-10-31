// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/base.carbon
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

var a: [i32; 1] = (1,);
var b: [f64; 2] = (11.1, 2.2,);
var c: [(); 5] = ((), (), (), (), (),);

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array.1: %.3 = tuple_value (%.2) [template]
// CHECK:STDOUT:   %.7: Core.BigInt = int_literal 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.9: type = array_type %.8, f64 [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.9 [template]
// CHECK:STDOUT:   %.11: f64 = float_literal 11.100000000000001 [template]
// CHECK:STDOUT:   %.12: f64 = float_literal 2.2000000000000002 [template]
// CHECK:STDOUT:   %.13: type = tuple_type (f64, f64) [template]
// CHECK:STDOUT:   %array.2: %.9 = tuple_value (%.11, %.12) [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.15: type = array_type %.14, %.1 [template]
// CHECK:STDOUT:   %.16: type = ptr_type %.15 [template]
// CHECK:STDOUT:   %.17: type = tuple_type (%.1, %.1, %.1, %.1, %.1) [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %.18: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.19: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %array.3: %.15 = tuple_value (%tuple, %tuple, %tuple, %tuple, %tuple) [template]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc114_14: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc114_9.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc114_9.2: type = converted %int.make_type_32, %.loc114_9.1 [template = i32]
// CHECK:STDOUT:   %.loc114_15: type = array_type %.loc114_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc115_9.1: Core.BigInt = int_literal 64 [template = constants.%.7]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%.loc115_9.1) [template = f64]
// CHECK:STDOUT:   %.loc115_14: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc115_9.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc115_9.3: type = converted %float.make_type, %.loc115_9.2 [template = f64]
// CHECK:STDOUT:   %.loc115_15: type = array_type %.loc115_14, f64 [template = constants.%.9]
// CHECK:STDOUT:   %b.var: ref %.9 = var b
// CHECK:STDOUT:   %b: ref %.9 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc116_10.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc116_13: i32 = int_literal 5 [template = constants.%.14]
// CHECK:STDOUT:   %.loc116_10.2: type = converted %.loc116_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc116_14: type = array_type %.loc116_13, %.1 [template = constants.%.15]
// CHECK:STDOUT:   %c.var: ref %.15 = var c
// CHECK:STDOUT:   %c: ref %.15 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size.param_patt: Core.BigInt) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc114_20: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc114_22.1: %.5 = tuple_literal (%.loc114_20)
// CHECK:STDOUT:   %.loc114_22.2: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc114_22.3: ref i32 = array_index file.%a.var, %.loc114_22.2
// CHECK:STDOUT:   %.loc114_22.4: init i32 = initialize_from %.loc114_20 to %.loc114_22.3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc114_22.5: init %.3 = array_init (%.loc114_22.4) to file.%a.var [template = constants.%array.1]
// CHECK:STDOUT:   %.loc114_23: init %.3 = converted %.loc114_22.1, %.loc114_22.5 [template = constants.%array.1]
// CHECK:STDOUT:   assign file.%a.var, %.loc114_23
// CHECK:STDOUT:   %.loc115_20: f64 = float_literal 11.100000000000001 [template = constants.%.11]
// CHECK:STDOUT:   %.loc115_26: f64 = float_literal 2.2000000000000002 [template = constants.%.12]
// CHECK:STDOUT:   %.loc115_30.1: %.13 = tuple_literal (%.loc115_20, %.loc115_26)
// CHECK:STDOUT:   %.loc115_30.2: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc115_30.3: ref f64 = array_index file.%b.var, %.loc115_30.2
// CHECK:STDOUT:   %.loc115_30.4: init f64 = initialize_from %.loc115_20 to %.loc115_30.3 [template = constants.%.11]
// CHECK:STDOUT:   %.loc115_30.5: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc115_30.6: ref f64 = array_index file.%b.var, %.loc115_30.5
// CHECK:STDOUT:   %.loc115_30.7: init f64 = initialize_from %.loc115_26 to %.loc115_30.6 [template = constants.%.12]
// CHECK:STDOUT:   %.loc115_30.8: init %.9 = array_init (%.loc115_30.4, %.loc115_30.7) to file.%b.var [template = constants.%array.2]
// CHECK:STDOUT:   %.loc115_31: init %.9 = converted %.loc115_30.1, %.loc115_30.8 [template = constants.%array.2]
// CHECK:STDOUT:   assign file.%b.var, %.loc115_31
// CHECK:STDOUT:   %.loc116_20.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc116_24.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc116_28.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc116_32.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc116_36.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc116_38.1: %.17 = tuple_literal (%.loc116_20.1, %.loc116_24.1, %.loc116_28.1, %.loc116_32.1, %.loc116_36.1)
// CHECK:STDOUT:   %.loc116_38.2: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc116_38.3: ref %.1 = array_index file.%c.var, %.loc116_38.2
// CHECK:STDOUT:   %.loc116_20.2: init %.1 = tuple_init () to %.loc116_38.3 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc116_38.4: init %.1 = converted %.loc116_20.1, %.loc116_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc116_38.5: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc116_38.6: ref %.1 = array_index file.%c.var, %.loc116_38.5
// CHECK:STDOUT:   %.loc116_24.2: init %.1 = tuple_init () to %.loc116_38.6 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc116_38.7: init %.1 = converted %.loc116_24.1, %.loc116_24.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc116_38.8: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc116_38.9: ref %.1 = array_index file.%c.var, %.loc116_38.8
// CHECK:STDOUT:   %.loc116_28.2: init %.1 = tuple_init () to %.loc116_38.9 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc116_38.10: init %.1 = converted %.loc116_28.1, %.loc116_28.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc116_38.11: i32 = int_literal 3 [template = constants.%.18]
// CHECK:STDOUT:   %.loc116_38.12: ref %.1 = array_index file.%c.var, %.loc116_38.11
// CHECK:STDOUT:   %.loc116_32.2: init %.1 = tuple_init () to %.loc116_38.12 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc116_38.13: init %.1 = converted %.loc116_32.1, %.loc116_32.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc116_38.14: i32 = int_literal 4 [template = constants.%.19]
// CHECK:STDOUT:   %.loc116_38.15: ref %.1 = array_index file.%c.var, %.loc116_38.14
// CHECK:STDOUT:   %.loc116_36.2: init %.1 = tuple_init () to %.loc116_38.15 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc116_38.16: init %.1 = converted %.loc116_36.1, %.loc116_36.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc116_38.17: init %.15 = array_init (%.loc116_38.4, %.loc116_38.7, %.loc116_38.10, %.loc116_38.13, %.loc116_38.16) to file.%c.var [template = constants.%array.3]
// CHECK:STDOUT:   %.loc116_39: init %.15 = converted %.loc116_38.1, %.loc116_38.17 [template = constants.%array.3]
// CHECK:STDOUT:   assign file.%c.var, %.loc116_39
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
