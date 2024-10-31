// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_field_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_field_modifiers.carbon
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

class Class {

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: error: `default` not allowed on `var` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   default var j: i32;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default var j: i32;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: error: `final` not allowed on `var` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   final var k: i32;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  final var k: i32;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
  // CHECK:STDERR:   default let l: i32 = 0;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default let l: i32 = 0;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+3]]:3: error: `final` not allowed; requires interface scope [ModifierRequiresInterface]
  // CHECK:STDERR:   final let m: i32 = 1;
  // CHECK:STDERR:   ^~~~~
  final let m: i32 = 1;
}

// CHECK:STDOUT: --- fail_field_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class, i32 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.j: i32, .k: i32} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int.make_type_32.loc121: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc121_18.1: type = value_of_initializer %int.make_type_32.loc121 [template = i32]
// CHECK:STDOUT:   %.loc121_18.2: type = converted %int.make_type_32.loc121, %.loc121_18.1 [template = i32]
// CHECK:STDOUT:   %.loc121_16: %.2 = field_decl j, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc127: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc127_16.1: type = value_of_initializer %int.make_type_32.loc127 [template = i32]
// CHECK:STDOUT:   %.loc127_16.2: type = converted %int.make_type_32.loc127, %.loc127_16.1 [template = i32]
// CHECK:STDOUT:   %.loc127_14: %.2 = field_decl k, element1 [template]
// CHECK:STDOUT:   %int.make_type_32.loc133: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc133_18.1: type = value_of_initializer %int.make_type_32.loc133 [template = i32]
// CHECK:STDOUT:   %.loc133_18.2: type = converted %int.make_type_32.loc133, %.loc133_18.1 [template = i32]
// CHECK:STDOUT:   %.loc133_24: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %l: i32 = bind_name l, %.loc133_24
// CHECK:STDOUT:   %int.make_type_32.loc138: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc138_16.1: type = value_of_initializer %int.make_type_32.loc138 [template = i32]
// CHECK:STDOUT:   %.loc138_16.2: type = converted %int.make_type_32.loc138, %.loc138_16.1 [template = i32]
// CHECK:STDOUT:   %.loc138_22: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %m: i32 = bind_name m, %.loc138_22
// CHECK:STDOUT:   %.loc139: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .j = %.loc121_16
// CHECK:STDOUT:   .k = %.loc127_14
// CHECK:STDOUT:   .l = %l
// CHECK:STDOUT:   .m = %m
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
