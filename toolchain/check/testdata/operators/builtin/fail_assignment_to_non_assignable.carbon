// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_assignment_to_non_assignable.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_assignment_to_non_assignable.carbon
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
// CHECK:STDERR:

fn F() -> i32;

fn Main() {
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   1 = 2;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  1 = 2;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   F() = 1;
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  F() = 1;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   (1, 2) = (3, 4);
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  (1, 2) = (3, 4);
  var n: i32 = 0;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   (n, n) = (1, 2);
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  (n, n) = (1, 2);
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   i32 = i32*;
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  i32 = i32*;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   (if true then 1 else 2) = 3;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (if true then 1 else 2) = 3;

  // Under #911, if expressions are never reference expressions.
  var a: i32;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+3]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   (if true then a else a) = 10;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  (if true then a else a) = 10;
}

// CHECK:STDOUT: --- fail_assignment_to_non_assignable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %tuple.1: %.4 = tuple_value (%.5, %.6) [template]
// CHECK:STDOUT:   %tuple.2: %.4 = tuple_value (%.2, %.3) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.9: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.10: type = struct_type {.x: i32, .y: i32} [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.10 [template]
// CHECK:STDOUT:   %struct.1: %.10 = struct_value (%.5, %.6) [template]
// CHECK:STDOUT:   %struct.2: %.10 = struct_value (%.2, %.3) [template]
// CHECK:STDOUT:   %.12: bool = bool_literal true [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 10 [template]
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
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc115_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc115_11.2: type = converted %int.make_type_32, %.loc115_11.1 [template = i32]
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc122_3: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc122_7: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   assign %.loc122_3, %.loc122_7
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init i32 = call %F.ref()
// CHECK:STDOUT:   %.loc127: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   assign %F.call, %.loc127
// CHECK:STDOUT:   %.loc132_4: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc132_7: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc132_8.1: %.4 = tuple_literal (%.loc132_4, %.loc132_7)
// CHECK:STDOUT:   %.loc132_13: i32 = int_literal 3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc132_16: i32 = int_literal 4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc132_17.1: %.4 = tuple_literal (%.loc132_13, %.loc132_16)
// CHECK:STDOUT:   %.loc132_17.2: i32 = tuple_access %.loc132_8.1, element0
// CHECK:STDOUT:   %.loc132_17.3: init i32 = initialize_from %.loc132_13 to %.loc132_17.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc132_17.4: i32 = tuple_access %.loc132_8.1, element1
// CHECK:STDOUT:   %.loc132_17.5: init i32 = initialize_from %.loc132_16 to %.loc132_17.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc132_17.6: init %.4 = tuple_init (%.loc132_17.3, %.loc132_17.5) to %.loc132_8.1 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc132_10: init %.4 = converted %.loc132_17.1, %.loc132_17.6 [template = constants.%tuple.1]
// CHECK:STDOUT:   assign %.loc132_8.1, %.loc132_10
// CHECK:STDOUT:   %tuple.loc132: %.4 = tuple_value (%.loc132_4, %.loc132_7) [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc132_8.2: %.4 = converted %.loc132_8.1, %tuple.loc132 [template = constants.%tuple.2]
// CHECK:STDOUT:   %int.make_type_32.loc133: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc133_10.1: type = value_of_initializer %int.make_type_32.loc133 [template = i32]
// CHECK:STDOUT:   %.loc133_10.2: type = converted %int.make_type_32.loc133, %.loc133_10.1 [template = i32]
// CHECK:STDOUT:   %n.var: ref i32 = var n
// CHECK:STDOUT:   %n: ref i32 = bind_name n, %n.var
// CHECK:STDOUT:   %.loc133_16: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   assign %n.var, %.loc133_16
// CHECK:STDOUT:   %n.ref.loc138_4: ref i32 = name_ref n, %n
// CHECK:STDOUT:   %n.ref.loc138_7: ref i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc138_8.1: %.4 = tuple_literal (%n.ref.loc138_4, %n.ref.loc138_7)
// CHECK:STDOUT:   %.loc138_13: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc138_16: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc138_17.1: %.4 = tuple_literal (%.loc138_13, %.loc138_16)
// CHECK:STDOUT:   %.loc138_17.2: i32 = tuple_access %.loc138_8.1, element0
// CHECK:STDOUT:   %.loc138_17.3: init i32 = initialize_from %.loc138_13 to %.loc138_17.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc138_17.4: i32 = tuple_access %.loc138_8.1, element1
// CHECK:STDOUT:   %.loc138_17.5: init i32 = initialize_from %.loc138_16 to %.loc138_17.4 [template = constants.%.3]
// CHECK:STDOUT:   %.loc138_17.6: init %.4 = tuple_init (%.loc138_17.3, %.loc138_17.5) to %.loc138_8.1 [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc138_10: init %.4 = converted %.loc138_17.1, %.loc138_17.6 [template = constants.%tuple.2]
// CHECK:STDOUT:   assign %.loc138_8.1, %.loc138_10
// CHECK:STDOUT:   %.loc138_4: i32 = bind_value %n.ref.loc138_4
// CHECK:STDOUT:   %.loc138_7: i32 = bind_value %n.ref.loc138_7
// CHECK:STDOUT:   %tuple.loc138: %.4 = tuple_value (%.loc138_4, %.loc138_7)
// CHECK:STDOUT:   %.loc138_8.2: %.4 = converted %.loc138_8.1, %tuple.loc138
// CHECK:STDOUT:   %int.make_type_32.loc143_3: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc143_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc143_12.1: type = value_of_initializer %int.make_type_32.loc143_9 [template = i32]
// CHECK:STDOUT:   %.loc143_12.2: type = converted %int.make_type_32.loc143_9, %.loc143_12.1 [template = i32]
// CHECK:STDOUT:   %.loc143_12.3: type = ptr_type i32 [template = constants.%.9]
// CHECK:STDOUT:   assign %int.make_type_32.loc143_3, %.loc143_12.3
// CHECK:STDOUT:   %.loc148_9: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc148_17: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc148_18.1: %.10 = struct_literal (%.loc148_9, %.loc148_17)
// CHECK:STDOUT:   %.loc148_28: i32 = int_literal 3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc148_36: i32 = int_literal 4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc148_37.1: %.10 = struct_literal (%.loc148_28, %.loc148_36)
// CHECK:STDOUT:   %.loc148_37.2: i32 = struct_access %.loc148_18.1, element0
// CHECK:STDOUT:   %.loc148_37.3: init i32 = initialize_from %.loc148_28 to %.loc148_37.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc148_37.4: i32 = struct_access %.loc148_18.1, element1
// CHECK:STDOUT:   %.loc148_37.5: init i32 = initialize_from %.loc148_36 to %.loc148_37.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc148_37.6: init %.10 = struct_init (%.loc148_37.3, %.loc148_37.5) to %.loc148_18.1 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc148_20: init %.10 = converted %.loc148_37.1, %.loc148_37.6 [template = constants.%struct.1]
// CHECK:STDOUT:   assign %.loc148_18.1, %.loc148_20
// CHECK:STDOUT:   %struct: %.10 = struct_value (%.loc148_9, %.loc148_17) [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc148_18.2: %.10 = converted %.loc148_18.1, %struct [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc153_7: bool = bool_literal true [template = constants.%.12]
// CHECK:STDOUT:   if %.loc153_7 br !if.expr.then.loc153 else br !if.expr.else.loc153
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc153:
// CHECK:STDOUT:   %.loc153_17: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   br !if.expr.result.loc153(%.loc153_17)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc153:
// CHECK:STDOUT:   %.loc153_24: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   br !if.expr.result.loc153(%.loc153_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc153:
// CHECK:STDOUT:   %.loc153_4: i32 = block_arg !if.expr.result.loc153 [template = constants.%.2]
// CHECK:STDOUT:   %.loc153_29: i32 = int_literal 3 [template = constants.%.5]
// CHECK:STDOUT:   assign %.loc153_4, %.loc153_29
// CHECK:STDOUT:   %int.make_type_32.loc156: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc156_10.1: type = value_of_initializer %int.make_type_32.loc156 [template = i32]
// CHECK:STDOUT:   %.loc156_10.2: type = converted %int.make_type_32.loc156, %.loc156_10.1 [template = i32]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc160_7: bool = bool_literal true [template = constants.%.12]
// CHECK:STDOUT:   if %.loc160_7 br !if.expr.then.loc160 else br !if.expr.else.loc160
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc160:
// CHECK:STDOUT:   %a.ref.loc160_17: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc160_17: i32 = bind_value %a.ref.loc160_17
// CHECK:STDOUT:   br !if.expr.result.loc160(%.loc160_17)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc160:
// CHECK:STDOUT:   %a.ref.loc160_24: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc160_24: i32 = bind_value %a.ref.loc160_24
// CHECK:STDOUT:   br !if.expr.result.loc160(%.loc160_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc160:
// CHECK:STDOUT:   %.loc160_4: i32 = block_arg !if.expr.result.loc160
// CHECK:STDOUT:   %.loc160_29: i32 = int_literal 10 [template = constants.%.13]
// CHECK:STDOUT:   assign %.loc160_4, %.loc160_29
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
