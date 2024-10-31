// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/address_of_lvalue.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/address_of_lvalue.carbon
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
  var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};

  var p: {.a: i32, .b: i32}* = &s;
  var q: i32* = &s.a;
  var r: i32* = &s.b;

  var t: (i32, i32) = (1, 2);
  var t0: i32* = &t.0;
  var t1: i32* = &t.1;
}

// CHECK:STDOUT: --- address_of_lvalue.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct: %.2 = struct_value (%.4, %.5) [template]
// CHECK:STDOUT:   %.6: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.8: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %tuple: %.8 = tuple_value (%.4, %.5) [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 0 [template]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc115_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc115_15.1: type = value_of_initializer %int.make_type_32.loc115_15 [template = i32]
// CHECK:STDOUT:   %.loc115_15.2: type = converted %int.make_type_32.loc115_15, %.loc115_15.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc115_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc115_24.1: type = value_of_initializer %int.make_type_32.loc115_24 [template = i32]
// CHECK:STDOUT:   %.loc115_24.2: type = converted %int.make_type_32.loc115_24, %.loc115_24.1 [template = i32]
// CHECK:STDOUT:   %.loc115_27: type = struct_type {.a: i32, .b: i32} [template = constants.%.2]
// CHECK:STDOUT:   %s.var: ref %.2 = var s
// CHECK:STDOUT:   %s: ref %.2 = bind_name s, %s.var
// CHECK:STDOUT:   %.loc115_37: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc115_45: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc115_46.1: %.2 = struct_literal (%.loc115_37, %.loc115_45)
// CHECK:STDOUT:   %.loc115_46.2: ref i32 = struct_access %s.var, element0
// CHECK:STDOUT:   %.loc115_46.3: init i32 = initialize_from %.loc115_37 to %.loc115_46.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc115_46.4: ref i32 = struct_access %s.var, element1
// CHECK:STDOUT:   %.loc115_46.5: init i32 = initialize_from %.loc115_45 to %.loc115_46.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc115_46.6: init %.2 = struct_init (%.loc115_46.3, %.loc115_46.5) to %s.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc115_47: init %.2 = converted %.loc115_46.1, %.loc115_46.6 [template = constants.%struct]
// CHECK:STDOUT:   assign %s.var, %.loc115_47
// CHECK:STDOUT:   %int.make_type_32.loc117_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc117_15.1: type = value_of_initializer %int.make_type_32.loc117_15 [template = i32]
// CHECK:STDOUT:   %.loc117_15.2: type = converted %int.make_type_32.loc117_15, %.loc117_15.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc117_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc117_24.1: type = value_of_initializer %int.make_type_32.loc117_24 [template = i32]
// CHECK:STDOUT:   %.loc117_24.2: type = converted %int.make_type_32.loc117_24, %.loc117_24.1 [template = i32]
// CHECK:STDOUT:   %.loc117_27: type = struct_type {.a: i32, .b: i32} [template = constants.%.2]
// CHECK:STDOUT:   %.loc117_28: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:   %p.var: ref %.3 = var p
// CHECK:STDOUT:   %p: ref %.3 = bind_name p, %p.var
// CHECK:STDOUT:   %s.ref.loc117: ref %.2 = name_ref s, %s
// CHECK:STDOUT:   %.loc117_32: %.3 = addr_of %s.ref.loc117
// CHECK:STDOUT:   assign %p.var, %.loc117_32
// CHECK:STDOUT:   %int.make_type_32.loc118: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc118_13.1: type = value_of_initializer %int.make_type_32.loc118 [template = i32]
// CHECK:STDOUT:   %.loc118_13.2: type = converted %int.make_type_32.loc118, %.loc118_13.1 [template = i32]
// CHECK:STDOUT:   %.loc118_13.3: type = ptr_type i32 [template = constants.%.6]
// CHECK:STDOUT:   %q.var: ref %.6 = var q
// CHECK:STDOUT:   %q: ref %.6 = bind_name q, %q.var
// CHECK:STDOUT:   %s.ref.loc118: ref %.2 = name_ref s, %s
// CHECK:STDOUT:   %.loc118_19: ref i32 = struct_access %s.ref.loc118, element0
// CHECK:STDOUT:   %.loc118_17: %.6 = addr_of %.loc118_19
// CHECK:STDOUT:   assign %q.var, %.loc118_17
// CHECK:STDOUT:   %int.make_type_32.loc119: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc119_13.1: type = value_of_initializer %int.make_type_32.loc119 [template = i32]
// CHECK:STDOUT:   %.loc119_13.2: type = converted %int.make_type_32.loc119, %.loc119_13.1 [template = i32]
// CHECK:STDOUT:   %.loc119_13.3: type = ptr_type i32 [template = constants.%.6]
// CHECK:STDOUT:   %r.var: ref %.6 = var r
// CHECK:STDOUT:   %r: ref %.6 = bind_name r, %r.var
// CHECK:STDOUT:   %s.ref.loc119: ref %.2 = name_ref s, %s
// CHECK:STDOUT:   %.loc119_19: ref i32 = struct_access %s.ref.loc119, element1
// CHECK:STDOUT:   %.loc119_17: %.6 = addr_of %.loc119_19
// CHECK:STDOUT:   assign %r.var, %.loc119_17
// CHECK:STDOUT:   %int.make_type_32.loc121_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc121_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc121_19.1: %.7 = tuple_literal (%int.make_type_32.loc121_11, %int.make_type_32.loc121_16)
// CHECK:STDOUT:   %.loc121_19.2: type = value_of_initializer %int.make_type_32.loc121_11 [template = i32]
// CHECK:STDOUT:   %.loc121_19.3: type = converted %int.make_type_32.loc121_11, %.loc121_19.2 [template = i32]
// CHECK:STDOUT:   %.loc121_19.4: type = value_of_initializer %int.make_type_32.loc121_16 [template = i32]
// CHECK:STDOUT:   %.loc121_19.5: type = converted %int.make_type_32.loc121_16, %.loc121_19.4 [template = i32]
// CHECK:STDOUT:   %.loc121_19.6: type = converted %.loc121_19.1, constants.%.8 [template = constants.%.8]
// CHECK:STDOUT:   %t.var: ref %.8 = var t
// CHECK:STDOUT:   %t: ref %.8 = bind_name t, %t.var
// CHECK:STDOUT:   %.loc121_24: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc121_27: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc121_28.1: %.8 = tuple_literal (%.loc121_24, %.loc121_27)
// CHECK:STDOUT:   %.loc121_28.2: ref i32 = tuple_access %t.var, element0
// CHECK:STDOUT:   %.loc121_28.3: init i32 = initialize_from %.loc121_24 to %.loc121_28.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc121_28.4: ref i32 = tuple_access %t.var, element1
// CHECK:STDOUT:   %.loc121_28.5: init i32 = initialize_from %.loc121_27 to %.loc121_28.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc121_28.6: init %.8 = tuple_init (%.loc121_28.3, %.loc121_28.5) to %t.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc121_29: init %.8 = converted %.loc121_28.1, %.loc121_28.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign %t.var, %.loc121_29
// CHECK:STDOUT:   %int.make_type_32.loc122: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc122_14.1: type = value_of_initializer %int.make_type_32.loc122 [template = i32]
// CHECK:STDOUT:   %.loc122_14.2: type = converted %int.make_type_32.loc122, %.loc122_14.1 [template = i32]
// CHECK:STDOUT:   %.loc122_14.3: type = ptr_type i32 [template = constants.%.6]
// CHECK:STDOUT:   %t0.var: ref %.6 = var t0
// CHECK:STDOUT:   %t0: ref %.6 = bind_name t0, %t0.var
// CHECK:STDOUT:   %t.ref.loc122: ref %.8 = name_ref t, %t
// CHECK:STDOUT:   %.loc122_21: i32 = int_literal 0 [template = constants.%.10]
// CHECK:STDOUT:   %.loc122_20: ref i32 = tuple_access %t.ref.loc122, element0
// CHECK:STDOUT:   %.loc122_18: %.6 = addr_of %.loc122_20
// CHECK:STDOUT:   assign %t0.var, %.loc122_18
// CHECK:STDOUT:   %int.make_type_32.loc123: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc123_14.1: type = value_of_initializer %int.make_type_32.loc123 [template = i32]
// CHECK:STDOUT:   %.loc123_14.2: type = converted %int.make_type_32.loc123, %.loc123_14.1 [template = i32]
// CHECK:STDOUT:   %.loc123_14.3: type = ptr_type i32 [template = constants.%.6]
// CHECK:STDOUT:   %t1.var: ref %.6 = var t1
// CHECK:STDOUT:   %t1: ref %.6 = bind_name t1, %t1.var
// CHECK:STDOUT:   %t.ref.loc123: ref %.8 = name_ref t, %t
// CHECK:STDOUT:   %.loc123_21: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc123_20: ref i32 = tuple_access %t.ref.loc123, element1
// CHECK:STDOUT:   %.loc123_18: %.6 = addr_of %.loc123_20
// CHECK:STDOUT:   assign %t1.var, %.loc123_18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
