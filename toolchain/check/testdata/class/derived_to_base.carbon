// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/derived_to_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/derived_to_base.carbon
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

base class A {
  var a: i32;
}

base class B {
  extend base: A;
  var b: i32;
}

class C {
  extend base: B;
  var c: i32;
}

fn ConvertCToB(p: C*) -> B* { return p; }
fn ConvertBToA(p: B*) -> A* { return p; }
fn ConvertCToA(p: C*) -> A* { return p; }

fn ConvertValue(c: C) {
  let a: A = c;
}

fn ConvertRef(c: C*) -> A* {
  return &(*c as A);
}

fn ConvertInit() {
  let a: A = {.base = {.base = {.a = 1}, .b = 2}, .c = 3} as C;
}

// CHECK:STDOUT: --- derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %A, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %B, i32 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %A, .b: i32} [template]
// CHECK:STDOUT:   %.9: <witness> = complete_type_witness %.8 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.10: type = struct_type {.base: %.5, .b: i32} [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.10 [template]
// CHECK:STDOUT:   %.12: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %.13: type = unbound_element_type %C, %B [template]
// CHECK:STDOUT:   %.14: type = unbound_element_type %C, i32 [template]
// CHECK:STDOUT:   %.15: type = struct_type {.base: %B, .c: i32} [template]
// CHECK:STDOUT:   %.16: <witness> = complete_type_witness %.15 [template]
// CHECK:STDOUT:   %.17: type = ptr_type %C [template]
// CHECK:STDOUT:   %.18: type = ptr_type %B [template]
// CHECK:STDOUT:   %ConvertCToB.type: type = fn_type @ConvertCToB [template]
// CHECK:STDOUT:   %ConvertCToB: %ConvertCToB.type = struct_value () [template]
// CHECK:STDOUT:   %.19: type = struct_type {.base: %.12, .c: i32} [template]
// CHECK:STDOUT:   %.20: type = ptr_type %.19 [template]
// CHECK:STDOUT:   %.21: type = ptr_type %.15 [template]
// CHECK:STDOUT:   %.22: type = ptr_type %A [template]
// CHECK:STDOUT:   %ConvertBToA.type: type = fn_type @ConvertBToA [template]
// CHECK:STDOUT:   %ConvertBToA: %ConvertBToA.type = struct_value () [template]
// CHECK:STDOUT:   %ConvertCToA.type: type = fn_type @ConvertCToA [template]
// CHECK:STDOUT:   %ConvertCToA: %ConvertCToA.type = struct_value () [template]
// CHECK:STDOUT:   %ConvertValue.type: type = fn_type @ConvertValue [template]
// CHECK:STDOUT:   %ConvertValue: %ConvertValue.type = struct_value () [template]
// CHECK:STDOUT:   %ConvertRef.type: type = fn_type @ConvertRef [template]
// CHECK:STDOUT:   %ConvertRef: %ConvertRef.type = struct_value () [template]
// CHECK:STDOUT:   %ConvertInit.type: type = fn_type @ConvertInit [template]
// CHECK:STDOUT:   %ConvertInit: %ConvertInit.type = struct_value () [template]
// CHECK:STDOUT:   %.23: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.24: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.25: type = struct_type {.base: %.3, .b: i32} [template]
// CHECK:STDOUT:   %.26: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.27: type = struct_type {.base: %.25, .c: i32} [template]
// CHECK:STDOUT:   %struct.1: %A = struct_value (%.23) [template]
// CHECK:STDOUT:   %struct.2: %B = struct_value (%struct.1, %.24) [template]
// CHECK:STDOUT:   %struct.3: %C = struct_value (%struct.2, %.26) [template]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .ConvertCToB = %ConvertCToB.decl
// CHECK:STDOUT:     .ConvertBToA = %ConvertBToA.decl
// CHECK:STDOUT:     .ConvertCToA = %ConvertCToA.decl
// CHECK:STDOUT:     .ConvertValue = %ConvertValue.decl
// CHECK:STDOUT:     .ConvertRef = %ConvertRef.decl
// CHECK:STDOUT:     .ConvertInit = %ConvertInit.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %ConvertCToB.decl: %ConvertCToB.type = fn_decl @ConvertCToB [template = constants.%ConvertCToB] {
// CHECK:STDOUT:     %p.patt: %.17 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.17 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.18 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.18 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc128_20: type = ptr_type %C [template = constants.%.17]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc128_27: type = ptr_type %B [template = constants.%.18]
// CHECK:STDOUT:     %p.param: %.17 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.17 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.18 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.18 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertBToA.decl: %ConvertBToA.type = fn_decl @ConvertBToA [template = constants.%ConvertBToA] {
// CHECK:STDOUT:     %p.patt: %.18 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.18 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.22 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.22 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc129_20: type = ptr_type %B [template = constants.%.18]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc129_27: type = ptr_type %A [template = constants.%.22]
// CHECK:STDOUT:     %p.param: %.18 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.18 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.22 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.22 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertCToA.decl: %ConvertCToA.type = fn_decl @ConvertCToA [template = constants.%ConvertCToA] {
// CHECK:STDOUT:     %p.patt: %.17 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.17 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.22 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.22 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc130_20: type = ptr_type %C [template = constants.%.17]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc130_27: type = ptr_type %A [template = constants.%.22]
// CHECK:STDOUT:     %p.param: %.17 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.17 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.22 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.22 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertValue.decl: %ConvertValue.type = fn_decl @ConvertValue [template = constants.%ConvertValue] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertRef.decl: %ConvertRef.type = fn_decl @ConvertRef [template = constants.%ConvertRef] {
// CHECK:STDOUT:     %c.patt: %.17 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %.17 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.22 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.22 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc136_19: type = ptr_type %C [template = constants.%.17]
// CHECK:STDOUT:     %A.ref.loc136: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc136_26: type = ptr_type %A [template = constants.%.22]
// CHECK:STDOUT:     %c.param: %.17 = value_param runtime_param0
// CHECK:STDOUT:     %c: %.17 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %.22 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.22 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertInit.decl: %ConvertInit.type = fn_decl @ConvertInit [template = constants.%ConvertInit] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc115_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc115_10.2: type = converted %int.make_type_32, %.loc115_10.1 [template = i32]
// CHECK:STDOUT:   %.loc115_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc116: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .a = %.loc115_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc119: %.6 = base_decl %A, element0 [template]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc120_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc120_10.2: type = converted %int.make_type_32, %.loc120_10.1 [template = i32]
// CHECK:STDOUT:   %.loc120_8: %.7 = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc121: <witness> = complete_type_witness %.8 [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc119
// CHECK:STDOUT:   .b = %.loc120_8
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc124: %.13 = base_decl %B, element0 [template]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc125_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc125_10.2: type = converted %int.make_type_32, %.loc125_10.1 [template = i32]
// CHECK:STDOUT:   %.loc125_8: %.14 = field_decl c, element1 [template]
// CHECK:STDOUT:   %.loc126: <witness> = complete_type_witness %.15 [template = constants.%.16]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc124
// CHECK:STDOUT:   .c = %.loc125_8
// CHECK:STDOUT:   extend name_scope3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToB(%p.param_patt: %.17) -> %.18 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.17 = name_ref p, %p
// CHECK:STDOUT:   %.loc128_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc128_39.2: ref %B = class_element_access %.loc128_39.1, element0
// CHECK:STDOUT:   %.loc128_39.3: %.18 = addr_of %.loc128_39.2
// CHECK:STDOUT:   %.loc128_39.4: %.18 = converted %p.ref, %.loc128_39.3
// CHECK:STDOUT:   return %.loc128_39.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertBToA(%p.param_patt: %.18) -> %.22 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.18 = name_ref p, %p
// CHECK:STDOUT:   %.loc129_39.1: ref %B = deref %p.ref
// CHECK:STDOUT:   %.loc129_39.2: ref %A = class_element_access %.loc129_39.1, element0
// CHECK:STDOUT:   %.loc129_39.3: %.22 = addr_of %.loc129_39.2
// CHECK:STDOUT:   %.loc129_39.4: %.22 = converted %p.ref, %.loc129_39.3
// CHECK:STDOUT:   return %.loc129_39.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToA(%p.param_patt: %.17) -> %.22 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.17 = name_ref p, %p
// CHECK:STDOUT:   %.loc130_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc130_39.2: ref %B = class_element_access %.loc130_39.1, element0
// CHECK:STDOUT:   %.loc130_39.3: ref %A = class_element_access %.loc130_39.2, element0
// CHECK:STDOUT:   %.loc130_39.4: %.22 = addr_of %.loc130_39.3
// CHECK:STDOUT:   %.loc130_39.5: %.22 = converted %p.ref, %.loc130_39.4
// CHECK:STDOUT:   return %.loc130_39.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertValue(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc133_15.1: ref %B = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc133_15.2: ref %A = class_element_access %.loc133_15.1, element0
// CHECK:STDOUT:   %.loc133_15.3: ref %A = converted %c.ref, %.loc133_15.2
// CHECK:STDOUT:   %.loc133_15.4: %A = bind_value %.loc133_15.3
// CHECK:STDOUT:   %a: %A = bind_name a, %.loc133_15.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertRef(%c.param_patt: %.17) -> %.22 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %.17 = name_ref c, %c
// CHECK:STDOUT:   %.loc137_12: ref %C = deref %c.ref
// CHECK:STDOUT:   %A.ref.loc137: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc137_15.1: ref %B = class_element_access %.loc137_12, element0
// CHECK:STDOUT:   %.loc137_15.2: ref %A = class_element_access %.loc137_15.1, element0
// CHECK:STDOUT:   %.loc137_15.3: ref %A = converted %.loc137_12, %.loc137_15.2
// CHECK:STDOUT:   %.loc137_10: %.22 = addr_of %.loc137_15.3
// CHECK:STDOUT:   return %.loc137_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertInit() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc141_38: i32 = int_literal 1 [template = constants.%.23]
// CHECK:STDOUT:   %.loc141_39.1: %.3 = struct_literal (%.loc141_38)
// CHECK:STDOUT:   %.loc141_47: i32 = int_literal 2 [template = constants.%.24]
// CHECK:STDOUT:   %.loc141_48.1: %.25 = struct_literal (%.loc141_39.1, %.loc141_47)
// CHECK:STDOUT:   %.loc141_56: i32 = int_literal 3 [template = constants.%.26]
// CHECK:STDOUT:   %.loc141_57.1: %.27 = struct_literal (%.loc141_48.1, %.loc141_56)
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc141_57.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc141_57.3: ref %B = class_element_access %.loc141_57.2, element0
// CHECK:STDOUT:   %.loc141_48.2: ref %A = class_element_access %.loc141_57.3, element0
// CHECK:STDOUT:   %.loc141_39.2: ref i32 = class_element_access %.loc141_48.2, element0
// CHECK:STDOUT:   %.loc141_39.3: init i32 = initialize_from %.loc141_38 to %.loc141_39.2 [template = constants.%.23]
// CHECK:STDOUT:   %.loc141_39.4: init %A = class_init (%.loc141_39.3), %.loc141_48.2 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc141_48.3: init %A = converted %.loc141_39.1, %.loc141_39.4 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc141_48.4: ref i32 = class_element_access %.loc141_57.3, element1
// CHECK:STDOUT:   %.loc141_48.5: init i32 = initialize_from %.loc141_47 to %.loc141_48.4 [template = constants.%.24]
// CHECK:STDOUT:   %.loc141_48.6: init %B = class_init (%.loc141_48.3, %.loc141_48.5), %.loc141_57.3 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc141_57.4: init %B = converted %.loc141_48.1, %.loc141_48.6 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc141_57.5: ref i32 = class_element_access %.loc141_57.2, element1
// CHECK:STDOUT:   %.loc141_57.6: init i32 = initialize_from %.loc141_56 to %.loc141_57.5 [template = constants.%.26]
// CHECK:STDOUT:   %.loc141_57.7: init %C = class_init (%.loc141_57.4, %.loc141_57.6), %.loc141_57.2 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc141_57.8: ref %C = temporary %.loc141_57.2, %.loc141_57.7
// CHECK:STDOUT:   %.loc141_59: ref %C = converted %.loc141_57.1, %.loc141_57.8
// CHECK:STDOUT:   %.loc141_63.1: ref %B = class_element_access %.loc141_59, element0
// CHECK:STDOUT:   %.loc141_63.2: ref %A = class_element_access %.loc141_63.1, element0
// CHECK:STDOUT:   %.loc141_63.3: ref %A = converted %.loc141_59, %.loc141_63.2
// CHECK:STDOUT:   %.loc141_63.4: %A = bind_value %.loc141_63.3
// CHECK:STDOUT:   %a: %A = bind_name a, %.loc141_63.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
