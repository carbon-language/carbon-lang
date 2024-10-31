// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/fail_redefined.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/fail_redefined.carbon
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

fn A(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+7]]:1: error: redefinition of `fn A` [RedeclRedef]
// CHECK:STDERR: fn A(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: fn A(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn A(n: i32, m: i32) -> i32 { return n; }

fn B(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+7]]:1: error: redefinition of `fn B` [RedeclRedef]
// CHECK:STDERR: fn B(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: fn B(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn B(n: i32, m: i32) -> i32 = "int.sadd";

fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+6]]:1: error: redefinition of `fn C` [RedeclRedef]
// CHECK:STDERR: fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn C(n: i32, m: i32) -> i32 = "int.sadd";

// CHECK:STDOUT: --- fail_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
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
// CHECK:STDOUT:     .A = %A.decl.loc115
// CHECK:STDOUT:     .B = %B.decl.loc125
// CHECK:STDOUT:     .C = %C.decl.loc135
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl.loc115: %A.type = fn_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %m.patt: i32 = binding_pattern m
// CHECK:STDOUT:     %m.param_patt: i32 = value_param_pattern %m.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc115_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc115_9.1: type = value_of_initializer %int.make_type_32.loc115_9 [template = i32]
// CHECK:STDOUT:     %.loc115_9.2: type = converted %int.make_type_32.loc115_9, %.loc115_9.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc115_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc115_17.1: type = value_of_initializer %int.make_type_32.loc115_17 [template = i32]
// CHECK:STDOUT:     %.loc115_17.2: type = converted %int.make_type_32.loc115_17, %.loc115_17.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc115_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc115_25.1: type = value_of_initializer %int.make_type_32.loc115_25 [template = i32]
// CHECK:STDOUT:     %.loc115_25.2: type = converted %int.make_type_32.loc115_25, %.loc115_25.1 [template = i32]
// CHECK:STDOUT:     %n.param.loc115: i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc115: i32 = bind_name n, %n.param.loc115
// CHECK:STDOUT:     %m.param.loc115: i32 = value_param runtime_param1
// CHECK:STDOUT:     %m.loc115: i32 = bind_name m, %m.param.loc115
// CHECK:STDOUT:     %return.param.loc115: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %.loc115_22: ref i32 = return_slot %return.param.loc115
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc123: %A.type = fn_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %m.patt: i32 = binding_pattern m
// CHECK:STDOUT:     %m.param_patt: i32 = value_param_pattern %m.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc123_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc123_9.1: type = value_of_initializer %int.make_type_32.loc123_9 [template = i32]
// CHECK:STDOUT:     %.loc123_9.2: type = converted %int.make_type_32.loc123_9, %.loc123_9.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc123_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc123_17.1: type = value_of_initializer %int.make_type_32.loc123_17 [template = i32]
// CHECK:STDOUT:     %.loc123_17.2: type = converted %int.make_type_32.loc123_17, %.loc123_17.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc123_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc123_25.1: type = value_of_initializer %int.make_type_32.loc123_25 [template = i32]
// CHECK:STDOUT:     %.loc123_25.2: type = converted %int.make_type_32.loc123_25, %.loc123_25.1 [template = i32]
// CHECK:STDOUT:     %n.param.loc123: i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc123: i32 = bind_name n, %n.param.loc123
// CHECK:STDOUT:     %m.param.loc123: i32 = value_param runtime_param1
// CHECK:STDOUT:     %m.loc123: i32 = bind_name m, %m.param.loc123
// CHECK:STDOUT:     %return.param.loc123: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param.loc123
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc125: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %m.patt: i32 = binding_pattern m
// CHECK:STDOUT:     %m.param_patt: i32 = value_param_pattern %m.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc125_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc125_9.1: type = value_of_initializer %int.make_type_32.loc125_9 [template = i32]
// CHECK:STDOUT:     %.loc125_9.2: type = converted %int.make_type_32.loc125_9, %.loc125_9.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc125_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc125_17.1: type = value_of_initializer %int.make_type_32.loc125_17 [template = i32]
// CHECK:STDOUT:     %.loc125_17.2: type = converted %int.make_type_32.loc125_17, %.loc125_17.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc125_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc125_25.1: type = value_of_initializer %int.make_type_32.loc125_25 [template = i32]
// CHECK:STDOUT:     %.loc125_25.2: type = converted %int.make_type_32.loc125_25, %.loc125_25.1 [template = i32]
// CHECK:STDOUT:     %n.param.loc125: i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc125: i32 = bind_name n, %n.param.loc125
// CHECK:STDOUT:     %m.param.loc125: i32 = value_param runtime_param1
// CHECK:STDOUT:     %m.loc125: i32 = bind_name m, %m.param.loc125
// CHECK:STDOUT:     %return.param.loc125: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %.loc125_22: ref i32 = return_slot %return.param.loc125
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc133: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %m.patt: i32 = binding_pattern m
// CHECK:STDOUT:     %m.param_patt: i32 = value_param_pattern %m.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc133_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc133_9.1: type = value_of_initializer %int.make_type_32.loc133_9 [template = i32]
// CHECK:STDOUT:     %.loc133_9.2: type = converted %int.make_type_32.loc133_9, %.loc133_9.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc133_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc133_17.1: type = value_of_initializer %int.make_type_32.loc133_17 [template = i32]
// CHECK:STDOUT:     %.loc133_17.2: type = converted %int.make_type_32.loc133_17, %.loc133_17.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc133_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc133_25.1: type = value_of_initializer %int.make_type_32.loc133_25 [template = i32]
// CHECK:STDOUT:     %.loc133_25.2: type = converted %int.make_type_32.loc133_25, %.loc133_25.1 [template = i32]
// CHECK:STDOUT:     %n.param.loc133: i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc133: i32 = bind_name n, %n.param.loc133
// CHECK:STDOUT:     %m.param.loc133: i32 = value_param runtime_param1
// CHECK:STDOUT:     %m.loc133: i32 = bind_name m, %m.param.loc133
// CHECK:STDOUT:     %return.param.loc133: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param.loc133
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc135: %C.type = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %m.patt: i32 = binding_pattern m
// CHECK:STDOUT:     %m.param_patt: i32 = value_param_pattern %m.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc135_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc135_9.1: type = value_of_initializer %int.make_type_32.loc135_9 [template = i32]
// CHECK:STDOUT:     %.loc135_9.2: type = converted %int.make_type_32.loc135_9, %.loc135_9.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc135_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc135_17.1: type = value_of_initializer %int.make_type_32.loc135_17 [template = i32]
// CHECK:STDOUT:     %.loc135_17.2: type = converted %int.make_type_32.loc135_17, %.loc135_17.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc135_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc135_25.1: type = value_of_initializer %int.make_type_32.loc135_25 [template = i32]
// CHECK:STDOUT:     %.loc135_25.2: type = converted %int.make_type_32.loc135_25, %.loc135_25.1 [template = i32]
// CHECK:STDOUT:     %n.param.loc135: i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc135: i32 = bind_name n, %n.param.loc135
// CHECK:STDOUT:     %m.param.loc135: i32 = value_param runtime_param1
// CHECK:STDOUT:     %m.loc135: i32 = bind_name m, %m.param.loc135
// CHECK:STDOUT:     %return.param.loc135: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %.loc135_22: ref i32 = return_slot %return.param.loc135
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc142: %C.type = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %m.patt: i32 = binding_pattern m
// CHECK:STDOUT:     %m.param_patt: i32 = value_param_pattern %m.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc142_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc142_9.1: type = value_of_initializer %int.make_type_32.loc142_9 [template = i32]
// CHECK:STDOUT:     %.loc142_9.2: type = converted %int.make_type_32.loc142_9, %.loc142_9.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc142_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc142_17.1: type = value_of_initializer %int.make_type_32.loc142_17 [template = i32]
// CHECK:STDOUT:     %.loc142_17.2: type = converted %int.make_type_32.loc142_17, %.loc142_17.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc142_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc142_25.1: type = value_of_initializer %int.make_type_32.loc142_25 [template = i32]
// CHECK:STDOUT:     %.loc142_25.2: type = converted %int.make_type_32.loc142_25, %.loc142_25.1 [template = i32]
// CHECK:STDOUT:     %n.param.loc142: i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc142: i32 = bind_name n, %n.param.loc142
// CHECK:STDOUT:     %m.param.loc142: i32 = value_param runtime_param1
// CHECK:STDOUT:     %m.loc142: i32 = bind_name m, %m.param.loc142
// CHECK:STDOUT:     %return.param.loc142: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param.loc142
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%n.param_patt: i32, %m.param_patt: i32) -> i32 = "int.sadd" {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n.loc123
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%n.param_patt: i32, %m.param_patt: i32) -> i32 = "int.sadd" {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n.loc125
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%n.param_patt: i32, %m.param_patt: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
