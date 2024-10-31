// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/assignment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/assignment.carbon
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

fn Main() {
  var a: i32 = 12;
  a = 9;

  var b: (i32, i32) = (1, 2);
  b.0 = 3;
  b.1 = 4;

  var c: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  c.a = 3;
  c.b = 4;

  var p: i32* = &a;
  *p = 5;

  *(if true then p else &a) = 10;
}

// CHECK:STDOUT: --- assignment.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 9 [template]
// CHECK:STDOUT:   %.4: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %tuple: %.5 = tuple_value (%.7, %.8) [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.12: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.13: type = ptr_type %.12 [template]
// CHECK:STDOUT:   %struct: %.12 = struct_value (%.7, %.8) [template]
// CHECK:STDOUT:   %.14: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.16: bool = bool_literal true [template]
// CHECK:STDOUT:   %.17: i32 = int_literal 10 [template]
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
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc115: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc115_10.1: type = value_of_initializer %int.make_type_32.loc115 [template = i32]
// CHECK:STDOUT:   %.loc115_10.2: type = converted %int.make_type_32.loc115, %.loc115_10.1 [template = i32]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc115_16: i32 = int_literal 12 [template = constants.%.2]
// CHECK:STDOUT:   assign %a.var, %.loc115_16
// CHECK:STDOUT:   %a.ref.loc116: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc116: i32 = int_literal 9 [template = constants.%.3]
// CHECK:STDOUT:   assign %a.ref.loc116, %.loc116
// CHECK:STDOUT:   %int.make_type_32.loc118_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc118_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc118_19.1: %.4 = tuple_literal (%int.make_type_32.loc118_11, %int.make_type_32.loc118_16)
// CHECK:STDOUT:   %.loc118_19.2: type = value_of_initializer %int.make_type_32.loc118_11 [template = i32]
// CHECK:STDOUT:   %.loc118_19.3: type = converted %int.make_type_32.loc118_11, %.loc118_19.2 [template = i32]
// CHECK:STDOUT:   %.loc118_19.4: type = value_of_initializer %int.make_type_32.loc118_16 [template = i32]
// CHECK:STDOUT:   %.loc118_19.5: type = converted %int.make_type_32.loc118_16, %.loc118_19.4 [template = i32]
// CHECK:STDOUT:   %.loc118_19.6: type = converted %.loc118_19.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:   %b.var: ref %.5 = var b
// CHECK:STDOUT:   %b: ref %.5 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc118_24: i32 = int_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc118_27: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc118_28.1: %.5 = tuple_literal (%.loc118_24, %.loc118_27)
// CHECK:STDOUT:   %.loc118_28.2: ref i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc118_28.3: init i32 = initialize_from %.loc118_24 to %.loc118_28.2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc118_28.4: ref i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc118_28.5: init i32 = initialize_from %.loc118_27 to %.loc118_28.4 [template = constants.%.8]
// CHECK:STDOUT:   %.loc118_28.6: init %.5 = tuple_init (%.loc118_28.3, %.loc118_28.5) to %b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc118_29: init %.5 = converted %.loc118_28.1, %.loc118_28.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc118_29
// CHECK:STDOUT:   %b.ref.loc119: ref %.5 = name_ref b, %b
// CHECK:STDOUT:   %.loc119_5: i32 = int_literal 0 [template = constants.%.9]
// CHECK:STDOUT:   %.loc119_4: ref i32 = tuple_access %b.ref.loc119, element0
// CHECK:STDOUT:   %.loc119_9: i32 = int_literal 3 [template = constants.%.10]
// CHECK:STDOUT:   assign %.loc119_4, %.loc119_9
// CHECK:STDOUT:   %b.ref.loc120: ref %.5 = name_ref b, %b
// CHECK:STDOUT:   %.loc120_5: i32 = int_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc120_4: ref i32 = tuple_access %b.ref.loc120, element1
// CHECK:STDOUT:   %.loc120_9: i32 = int_literal 4 [template = constants.%.11]
// CHECK:STDOUT:   assign %.loc120_4, %.loc120_9
// CHECK:STDOUT:   %int.make_type_32.loc122_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc122_15.1: type = value_of_initializer %int.make_type_32.loc122_15 [template = i32]
// CHECK:STDOUT:   %.loc122_15.2: type = converted %int.make_type_32.loc122_15, %.loc122_15.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc122_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc122_24.1: type = value_of_initializer %int.make_type_32.loc122_24 [template = i32]
// CHECK:STDOUT:   %.loc122_24.2: type = converted %int.make_type_32.loc122_24, %.loc122_24.1 [template = i32]
// CHECK:STDOUT:   %.loc122_27: type = struct_type {.a: i32, .b: i32} [template = constants.%.12]
// CHECK:STDOUT:   %c.var: ref %.12 = var c
// CHECK:STDOUT:   %c: ref %.12 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc122_37: i32 = int_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc122_45: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc122_46.1: %.12 = struct_literal (%.loc122_37, %.loc122_45)
// CHECK:STDOUT:   %.loc122_46.2: ref i32 = struct_access %c.var, element0
// CHECK:STDOUT:   %.loc122_46.3: init i32 = initialize_from %.loc122_37 to %.loc122_46.2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc122_46.4: ref i32 = struct_access %c.var, element1
// CHECK:STDOUT:   %.loc122_46.5: init i32 = initialize_from %.loc122_45 to %.loc122_46.4 [template = constants.%.8]
// CHECK:STDOUT:   %.loc122_46.6: init %.12 = struct_init (%.loc122_46.3, %.loc122_46.5) to %c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc122_47: init %.12 = converted %.loc122_46.1, %.loc122_46.6 [template = constants.%struct]
// CHECK:STDOUT:   assign %c.var, %.loc122_47
// CHECK:STDOUT:   %c.ref.loc123: ref %.12 = name_ref c, %c
// CHECK:STDOUT:   %.loc123_4: ref i32 = struct_access %c.ref.loc123, element0
// CHECK:STDOUT:   %.loc123_9: i32 = int_literal 3 [template = constants.%.10]
// CHECK:STDOUT:   assign %.loc123_4, %.loc123_9
// CHECK:STDOUT:   %c.ref.loc124: ref %.12 = name_ref c, %c
// CHECK:STDOUT:   %.loc124_4: ref i32 = struct_access %c.ref.loc124, element1
// CHECK:STDOUT:   %.loc124_9: i32 = int_literal 4 [template = constants.%.11]
// CHECK:STDOUT:   assign %.loc124_4, %.loc124_9
// CHECK:STDOUT:   %int.make_type_32.loc126: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc126_13.1: type = value_of_initializer %int.make_type_32.loc126 [template = i32]
// CHECK:STDOUT:   %.loc126_13.2: type = converted %int.make_type_32.loc126, %.loc126_13.1 [template = i32]
// CHECK:STDOUT:   %.loc126_13.3: type = ptr_type i32 [template = constants.%.14]
// CHECK:STDOUT:   %p.var: ref %.14 = var p
// CHECK:STDOUT:   %p: ref %.14 = bind_name p, %p.var
// CHECK:STDOUT:   %a.ref.loc126: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc126_17: %.14 = addr_of %a.ref.loc126
// CHECK:STDOUT:   assign %p.var, %.loc126_17
// CHECK:STDOUT:   %p.ref.loc127: ref %.14 = name_ref p, %p
// CHECK:STDOUT:   %.loc127_4: %.14 = bind_value %p.ref.loc127
// CHECK:STDOUT:   %.loc127_3: ref i32 = deref %.loc127_4
// CHECK:STDOUT:   %.loc127_8: i32 = int_literal 5 [template = constants.%.15]
// CHECK:STDOUT:   assign %.loc127_3, %.loc127_8
// CHECK:STDOUT:   %.loc129_8: bool = bool_literal true [template = constants.%.16]
// CHECK:STDOUT:   if %.loc129_8 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %p.ref.loc129: ref %.14 = name_ref p, %p
// CHECK:STDOUT:   %.loc129_18: %.14 = bind_value %p.ref.loc129
// CHECK:STDOUT:   br !if.expr.result(%.loc129_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc129: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc129_25: %.14 = addr_of %a.ref.loc129
// CHECK:STDOUT:   br !if.expr.result(%.loc129_25)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc129_5: %.14 = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc129_3: ref i32 = deref %.loc129_5
// CHECK:STDOUT:   %.loc129_31: i32 = int_literal 10 [template = constants.%.17]
// CHECK:STDOUT:   assign %.loc129_3, %.loc129_31
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
