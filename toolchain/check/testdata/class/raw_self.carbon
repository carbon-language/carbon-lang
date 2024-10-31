// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/raw_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/raw_self.carbon
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

class Class {
  fn F[addr self: Self*](r#self: i32);
  fn G[self: Self](r#self: i32) -> (i32, i32);
  var n: i32;
}

fn Class.F[addr self: Self*](r#self: i32) {
  (*self).n = r#self;
}

fn Class.G[self: Self](r#self: i32) -> (i32, i32) {
  return (self.n, r#self);
}

// CHECK:STDOUT: --- raw_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type %Class [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Class, i32 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.n: i32} [template]
// CHECK:STDOUT:   %.7: <witness> = complete_type_witness %.6 [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.6 [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.4 [template]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt.loc120_17: %.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt.loc120_21: %.1 = value_param_pattern %self.patt.loc120_17, runtime_param0
// CHECK:STDOUT:     %.loc120_12: auto = addr_pattern %self.param_patt.loc120_21
// CHECK:STDOUT:     %self.patt.loc120_30: i32 = binding_pattern r#self
// CHECK:STDOUT:     %self.param_patt.loc120_36: i32 = value_param_pattern %self.patt.loc120_30, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc120: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc120_27: type = ptr_type %Class [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_32.loc120: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc120_38.1: type = value_of_initializer %int.make_type_32.loc120 [template = i32]
// CHECK:STDOUT:     %.loc120_38.2: type = converted %int.make_type_32.loc120, %.loc120_38.1 [template = i32]
// CHECK:STDOUT:     %self.param.loc120_21: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc120_17: %.1 = bind_name self, %self.param.loc120_21
// CHECK:STDOUT:     %self.param.loc120_36: i32 = value_param runtime_param1
// CHECK:STDOUT:     %self.loc120_30: i32 = bind_name r#self, %self.param.loc120_36
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %self.patt.loc124_12: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt.loc124_16: %Class = value_param_pattern %self.patt.loc124_12, runtime_param0
// CHECK:STDOUT:     %self.patt.loc124_24: i32 = binding_pattern r#self
// CHECK:STDOUT:     %self.param_patt.loc124_30: i32 = value_param_pattern %self.patt.loc124_24, runtime_param1
// CHECK:STDOUT:     %return.patt: %.4 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.4 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc124: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %int.make_type_32.loc124_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc124_32.1: type = value_of_initializer %int.make_type_32.loc124_32 [template = i32]
// CHECK:STDOUT:     %.loc124_32.2: type = converted %int.make_type_32.loc124_32, %.loc124_32.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc124_41: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc124_46: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc124_49.1: %.3 = tuple_literal (%int.make_type_32.loc124_41, %int.make_type_32.loc124_46)
// CHECK:STDOUT:     %.loc124_49.2: type = value_of_initializer %int.make_type_32.loc124_41 [template = i32]
// CHECK:STDOUT:     %.loc124_49.3: type = converted %int.make_type_32.loc124_41, %.loc124_49.2 [template = i32]
// CHECK:STDOUT:     %.loc124_49.4: type = value_of_initializer %int.make_type_32.loc124_46 [template = i32]
// CHECK:STDOUT:     %.loc124_49.5: type = converted %int.make_type_32.loc124_46, %.loc124_49.4 [template = i32]
// CHECK:STDOUT:     %.loc124_49.6: type = converted %.loc124_49.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:     %self.param.loc124_16: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self.loc124_12: %Class = bind_name self, %self.param.loc124_16
// CHECK:STDOUT:     %self.param.loc124_30: i32 = value_param runtime_param1
// CHECK:STDOUT:     %self.loc124_24: i32 = bind_name r#self, %self.param.loc124_30
// CHECK:STDOUT:     %return.param.loc124: ref %.4 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %.4 = return_slot %return.param.loc124
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt.loc120_17: %.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt.loc120_21: %.1 = value_param_pattern %self.patt.loc120_17, runtime_param0
// CHECK:STDOUT:     %.loc120_12: auto = addr_pattern %self.param_patt.loc120_21
// CHECK:STDOUT:     %self.patt.loc120_30: i32 = binding_pattern r#self
// CHECK:STDOUT:     %self.param_patt.loc120_36: i32 = value_param_pattern %self.patt.loc120_30, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc115: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc115_23: type = ptr_type %Class [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_32.loc115: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc115_34.1: type = value_of_initializer %int.make_type_32.loc115 [template = i32]
// CHECK:STDOUT:     %.loc115_34.2: type = converted %int.make_type_32.loc115, %.loc115_34.1 [template = i32]
// CHECK:STDOUT:     %self.param.loc115_17: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc115_13: %.1 = bind_name self, %self.param.loc115_17
// CHECK:STDOUT:     %self.param.loc115_32: i32 = value_param runtime_param1
// CHECK:STDOUT:     %self.loc115_26: i32 = bind_name r#self, %self.param.loc115_32
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %self.patt.loc124_12: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt.loc124_16: %Class = value_param_pattern %self.patt.loc124_12, runtime_param0
// CHECK:STDOUT:     %self.patt.loc124_24: i32 = binding_pattern r#self
// CHECK:STDOUT:     %self.param_patt.loc124_30: i32 = value_param_pattern %self.patt.loc124_24, runtime_param1
// CHECK:STDOUT:     %return.patt: %.4 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.4 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc116: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %int.make_type_32.loc116_28: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc116_28.1: type = value_of_initializer %int.make_type_32.loc116_28 [template = i32]
// CHECK:STDOUT:     %.loc116_28.2: type = converted %int.make_type_32.loc116_28, %.loc116_28.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc116_37: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc116_42: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc116_45.1: %.3 = tuple_literal (%int.make_type_32.loc116_37, %int.make_type_32.loc116_42)
// CHECK:STDOUT:     %.loc116_45.2: type = value_of_initializer %int.make_type_32.loc116_37 [template = i32]
// CHECK:STDOUT:     %.loc116_45.3: type = converted %int.make_type_32.loc116_37, %.loc116_45.2 [template = i32]
// CHECK:STDOUT:     %.loc116_45.4: type = value_of_initializer %int.make_type_32.loc116_42 [template = i32]
// CHECK:STDOUT:     %.loc116_45.5: type = converted %int.make_type_32.loc116_42, %.loc116_45.4 [template = i32]
// CHECK:STDOUT:     %.loc116_45.6: type = converted %.loc116_45.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:     %self.param.loc116_12: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self.loc116_8: %Class = bind_name self, %self.param.loc116_12
// CHECK:STDOUT:     %self.param.loc116_26: i32 = value_param runtime_param1
// CHECK:STDOUT:     %self.loc116_20: i32 = bind_name r#self, %self.param.loc116_26
// CHECK:STDOUT:     %return.param.loc116: ref %.4 = out_param runtime_param2
// CHECK:STDOUT:     %.loc116_33: ref %.4 = return_slot %return.param.loc116
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc117_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc117_10.2: type = converted %int.make_type_32, %.loc117_10.1 [template = i32]
// CHECK:STDOUT:   %.loc117_8: %.5 = field_decl n, element0 [template]
// CHECK:STDOUT:   %.loc118: <witness> = complete_type_witness %.6 [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .n = %.loc117_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[addr %self.param_patt.loc120_21: %.1](%self.param_patt.loc120_36: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc121_5: %.1 = name_ref self, %self.loc120_17
// CHECK:STDOUT:   %.loc121_4: ref %Class = deref %self.ref.loc121_5
// CHECK:STDOUT:   %n.ref: %.5 = name_ref n, @Class.%.loc117_8 [template = @Class.%.loc117_8]
// CHECK:STDOUT:   %.loc121_10: ref i32 = class_element_access %.loc121_4, element0
// CHECK:STDOUT:   %self.ref.loc121_15: i32 = name_ref r#self, %self.loc120_30
// CHECK:STDOUT:   assign %.loc121_10, %self.ref.loc121_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[%self.param_patt.loc124_16: %Class](%self.param_patt.loc124_30: i32) -> %return: %.4 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc125_11: %Class = name_ref self, %self.loc124_12
// CHECK:STDOUT:   %n.ref: %.5 = name_ref n, @Class.%.loc117_8 [template = @Class.%.loc117_8]
// CHECK:STDOUT:   %.loc125_15.1: ref i32 = class_element_access %self.ref.loc125_11, element0
// CHECK:STDOUT:   %.loc125_15.2: i32 = bind_value %.loc125_15.1
// CHECK:STDOUT:   %self.ref.loc125_19: i32 = name_ref r#self, %self.loc124_24
// CHECK:STDOUT:   %.loc125_25.1: %.4 = tuple_literal (%.loc125_15.2, %self.ref.loc125_19)
// CHECK:STDOUT:   %.loc125_25.2: ref i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc125_25.3: init i32 = initialize_from %.loc125_15.2 to %.loc125_25.2
// CHECK:STDOUT:   %.loc125_25.4: ref i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc125_25.5: init i32 = initialize_from %self.ref.loc125_19 to %.loc125_25.4
// CHECK:STDOUT:   %.loc125_25.6: init %.4 = tuple_init (%.loc125_25.3, %.loc125_25.5) to %return
// CHECK:STDOUT:   %.loc125_26: init %.4 = converted %.loc125_25.1, %.loc125_25.6
// CHECK:STDOUT:   return %.loc125_26 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
