// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/and.carbon
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

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn And() -> bool {
  return F() and G();
}

fn Constant() {
  var a: if true and true then bool else () = true;
  var b: if true and false then bool else () = ();
  var c: if false and true then bool else () = ();
  var d: if false and false then bool else () = ();
}

fn PartialConstant(x: bool) {
  var a: if false and x then bool else () = ();
}

// CHECK:STDOUT: --- and.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %And.type: type = fn_type @And [template]
// CHECK:STDOUT:   %And: %And.type = struct_value () [template]
// CHECK:STDOUT:   %.3: bool = bool_literal false [template]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [template]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %PartialConstant.type: type = fn_type @PartialConstant [template]
// CHECK:STDOUT:   %PartialConstant: %PartialConstant.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref
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
// CHECK:STDOUT:   %import_ref: %Bool.type = import_ref Core//prelude/types/bool, inst+5, loaded [template = constants.%Bool]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:     .PartialConstant = %PartialConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc114_11.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc114_11.2: type = converted %bool.make_type, %.loc114_11.1 [template = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc115_11.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc115_11.2: type = converted %bool.make_type, %.loc115_11.1 [template = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [template = constants.%And] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc117_13.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc117_13.2: type = converted %bool.make_type, %.loc117_13.1 [template = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [template = constants.%Constant] {} {}
// CHECK:STDOUT:   %PartialConstant.decl: %PartialConstant.type = fn_decl @PartialConstant [template = constants.%PartialConstant] {
// CHECK:STDOUT:     %x.patt: bool = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: bool = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc128: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc128_23.1: type = value_of_initializer %bool.make_type.loc128 [template = bool]
// CHECK:STDOUT:     %.loc128_23.2: type = converted %bool.make_type.loc128, %.loc128_23.1 [template = bool]
// CHECK:STDOUT:     %x.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %x: bool = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc114_25: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   return %.loc114_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc115_25: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   return %.loc115_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @And() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc118_14.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc118_14.2: bool = converted %F.call, %.loc118_14.1
// CHECK:STDOUT:   %.loc118_14.3: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   if %.loc118_14.2 br !and.rhs else br !and.result(%.loc118_14.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %G.call: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc118_14.4: bool = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc118_14.5: bool = converted %G.call, %.loc118_14.4
// CHECK:STDOUT:   br !and.result(%.loc118_14.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc118_14.6: bool = block_arg !and.result
// CHECK:STDOUT:   return %.loc118_14.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc122_13: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   %.loc122_18.1: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   if %.loc122_13 br !and.rhs.loc122 else br !and.result.loc122(%.loc122_18.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc122:
// CHECK:STDOUT:   %.loc122_22: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   br !and.result.loc122(%.loc122_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc122:
// CHECK:STDOUT:   %.loc122_18.2: bool = block_arg !and.result.loc122 [template = constants.%.2]
// CHECK:STDOUT:   if %.loc122_18.2 br !if.expr.then.loc122 else br !if.expr.else.loc122
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc122:
// CHECK:STDOUT:   %bool.make_type.loc122: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc122_32.1: type = value_of_initializer %bool.make_type.loc122 [template = bool]
// CHECK:STDOUT:   %.loc122_32.2: type = converted %bool.make_type.loc122, %.loc122_32.1 [template = bool]
// CHECK:STDOUT:   br !if.expr.result.loc122(%.loc122_32.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc122:
// CHECK:STDOUT:   %.loc122_43: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc122_37: type = converted %.loc122_43, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result.loc122(%.loc122_37)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc122:
// CHECK:STDOUT:   %.loc122_10: type = block_arg !if.expr.result.loc122 [template = bool]
// CHECK:STDOUT:   %a.var: ref bool = var a
// CHECK:STDOUT:   %a: ref bool = bind_name a, %a.var
// CHECK:STDOUT:   %.loc122_47: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   assign %a.var, %.loc122_47
// CHECK:STDOUT:   %.loc123_13: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   %.loc123_18.1: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   if %.loc123_13 br !and.rhs.loc123 else br !and.result.loc123(%.loc123_18.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc123:
// CHECK:STDOUT:   %.loc123_22: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   br !and.result.loc123(%.loc123_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc123:
// CHECK:STDOUT:   %.loc123_18.2: bool = block_arg !and.result.loc123 [template = constants.%.3]
// CHECK:STDOUT:   if %.loc123_18.2 br !if.expr.then.loc123 else br !if.expr.else.loc123
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc123:
// CHECK:STDOUT:   %bool.make_type.loc123: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc123_33.1: type = value_of_initializer %bool.make_type.loc123 [template = bool]
// CHECK:STDOUT:   %.loc123_33.2: type = converted %bool.make_type.loc123, %.loc123_33.1 [template = bool]
// CHECK:STDOUT:   br !if.expr.result.loc123(%.loc123_33.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc123:
// CHECK:STDOUT:   %.loc123_44: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc123_38: type = converted %.loc123_44, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result.loc123(%.loc123_38)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc123:
// CHECK:STDOUT:   %.loc123_10: type = block_arg !if.expr.result.loc123 [template = constants.%.1]
// CHECK:STDOUT:   %b.var: ref %.1 = var b
// CHECK:STDOUT:   %b: ref %.1 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc123_49.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc123_49.2: init %.1 = tuple_init () to %b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc123_50: init %.1 = converted %.loc123_49.1, %.loc123_49.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc123_50
// CHECK:STDOUT:   %.loc124_13: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   %.loc124_19.1: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   if %.loc124_13 br !and.rhs.loc124 else br !and.result.loc124(%.loc124_19.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc124:
// CHECK:STDOUT:   %.loc124_23: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   br !and.result.loc124(%.loc124_23)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc124:
// CHECK:STDOUT:   %.loc124_19.2: bool = block_arg !and.result.loc124 [template = constants.%.3]
// CHECK:STDOUT:   if %.loc124_19.2 br !if.expr.then.loc124 else br !if.expr.else.loc124
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc124:
// CHECK:STDOUT:   %bool.make_type.loc124: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc124_33.1: type = value_of_initializer %bool.make_type.loc124 [template = bool]
// CHECK:STDOUT:   %.loc124_33.2: type = converted %bool.make_type.loc124, %.loc124_33.1 [template = bool]
// CHECK:STDOUT:   br !if.expr.result.loc124(%.loc124_33.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc124:
// CHECK:STDOUT:   %.loc124_44: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc124_38: type = converted %.loc124_44, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result.loc124(%.loc124_38)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc124:
// CHECK:STDOUT:   %.loc124_10: type = block_arg !if.expr.result.loc124 [template = constants.%.1]
// CHECK:STDOUT:   %c.var: ref %.1 = var c
// CHECK:STDOUT:   %c: ref %.1 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc124_49.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc124_49.2: init %.1 = tuple_init () to %c.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc124_50: init %.1 = converted %.loc124_49.1, %.loc124_49.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %c.var, %.loc124_50
// CHECK:STDOUT:   %.loc125_13: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   %.loc125_19.1: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   if %.loc125_13 br !and.rhs.loc125 else br !and.result.loc125(%.loc125_19.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc125:
// CHECK:STDOUT:   %.loc125_23: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   br !and.result.loc125(%.loc125_23)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc125:
// CHECK:STDOUT:   %.loc125_19.2: bool = block_arg !and.result.loc125 [template = constants.%.3]
// CHECK:STDOUT:   if %.loc125_19.2 br !if.expr.then.loc125 else br !if.expr.else.loc125
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc125:
// CHECK:STDOUT:   %bool.make_type.loc125: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc125_34.1: type = value_of_initializer %bool.make_type.loc125 [template = bool]
// CHECK:STDOUT:   %.loc125_34.2: type = converted %bool.make_type.loc125, %.loc125_34.1 [template = bool]
// CHECK:STDOUT:   br !if.expr.result.loc125(%.loc125_34.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc125:
// CHECK:STDOUT:   %.loc125_45: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc125_39: type = converted %.loc125_45, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result.loc125(%.loc125_39)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc125:
// CHECK:STDOUT:   %.loc125_10: type = block_arg !if.expr.result.loc125 [template = constants.%.1]
// CHECK:STDOUT:   %d.var: ref %.1 = var d
// CHECK:STDOUT:   %d: ref %.1 = bind_name d, %d.var
// CHECK:STDOUT:   %.loc125_50.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc125_50.2: init %.1 = tuple_init () to %d.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc125_51: init %.1 = converted %.loc125_50.1, %.loc125_50.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %d.var, %.loc125_51
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x.param_patt: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc129_13: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   %.loc129_19.1: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   if %.loc129_13 br !and.rhs else br !and.result(%.loc129_19.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %x.ref: bool = name_ref x, %x
// CHECK:STDOUT:   br !and.result(%x.ref)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc129_19.2: bool = block_arg !and.result [template = constants.%.3]
// CHECK:STDOUT:   if %.loc129_19.2 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %bool.make_type.loc129: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc129_30.1: type = value_of_initializer %bool.make_type.loc129 [template = bool]
// CHECK:STDOUT:   %.loc129_30.2: type = converted %bool.make_type.loc129, %.loc129_30.1 [template = bool]
// CHECK:STDOUT:   br !if.expr.result(%.loc129_30.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc129_41: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc129_35: type = converted %.loc129_41, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result(%.loc129_35)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc129_10: type = block_arg !if.expr.result [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref %.1 = var a
// CHECK:STDOUT:   %a: ref %.1 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc129_46.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc129_46.2: init %.1 = tuple_init () to %a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc129_47: init %.1 = converted %.loc129_46.1, %.loc129_46.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %a.var, %.loc129_47
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
