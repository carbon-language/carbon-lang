// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/base_method_shadow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/base_method_shadow.carbon
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
  fn F[addr self: Self*]();
}

base class B {
  extend base: A;
  fn F[addr self: Self*]();
}

class C {
  extend base: B;
  fn F[addr self: Self*]();
}

class D {
  extend base: B;
}

fn Call(a: A*, b: B*, c: C*, d: D*) {
  (*a).F();
  (*b).F();
  (*c).F();
  (*d).F();
}

// CHECK:STDOUT: --- base_method_shadow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = ptr_type %A [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %.7: type = ptr_type %B [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %A} [template]
// CHECK:STDOUT:   %.9: <witness> = complete_type_witness %.8 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.10: type = struct_type {.base: %.5} [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %.12: type = unbound_element_type %C, %B [template]
// CHECK:STDOUT:   %.13: type = ptr_type %C [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.14: type = struct_type {.base: %B} [template]
// CHECK:STDOUT:   %.15: <witness> = complete_type_witness %.14 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.16: type = unbound_element_type %D, %B [template]
// CHECK:STDOUT:   %.17: type = ptr_type %D [template]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
// CHECK:STDOUT:   %.18: type = struct_type {.base: %.11} [template]
// CHECK:STDOUT:   %.19: type = ptr_type %.14 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {
// CHECK:STDOUT:     %a.patt: %.1 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.1 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %.7 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.7 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %c.patt: %.13 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %.13 = value_param_pattern %c.patt, runtime_param2
// CHECK:STDOUT:     %d.patt: %.17 = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %.17 = value_param_pattern %d.patt, runtime_param3
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc132_13: type = ptr_type %A [template = constants.%.1]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc132_20: type = ptr_type %B [template = constants.%.7]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc132_27: type = ptr_type %C [template = constants.%.13]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %.loc132_34: type = ptr_type %D [template = constants.%.17]
// CHECK:STDOUT:     %a.param: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.1 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %.7 = value_param runtime_param1
// CHECK:STDOUT:     %b: %.7 = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: %.13 = value_param runtime_param2
// CHECK:STDOUT:     %c: %.13 = bind_name c, %c.param
// CHECK:STDOUT:     %d.param: %.17 = value_param runtime_param3
// CHECK:STDOUT:     %d: %.17 = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: %.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.1 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc115_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [template = constants.%A]
// CHECK:STDOUT:     %.loc115_23: type = ptr_type %A [template = constants.%.1]
// CHECK:STDOUT:     %self.param: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.1 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc116: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc119: %.6 = base_decl %A, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: %.7 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.7 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc120_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [template = constants.%B]
// CHECK:STDOUT:     %.loc120_23: type = ptr_type %B [template = constants.%.7]
// CHECK:STDOUT:     %self.param: %.7 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.7 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc121: <witness> = complete_type_witness %.8 [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc119
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc124: %.12 = base_decl %B, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.3 [template = constants.%F.3] {
// CHECK:STDOUT:     %self.patt: %.13 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.13 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc125_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %.loc125_23: type = ptr_type %C [template = constants.%.13]
// CHECK:STDOUT:     %self.param: %.13 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.13 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc126: <witness> = complete_type_witness %.14 [template = constants.%.15]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc124
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend name_scope3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc129: %.16 = base_decl %B, element0 [template]
// CHECK:STDOUT:   %.loc130: <witness> = complete_type_witness %.14 [template = constants.%.15]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .base = %.loc129
// CHECK:STDOUT:   extend name_scope3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr %self.param_patt: %.1]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[addr %self.param_patt: %.7]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[addr %self.param_patt: %.13]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a.param_patt: %.1, %b.param_patt: %.7, %c.param_patt: %.13, %d.param_patt: %.17) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %.1 = name_ref a, %a
// CHECK:STDOUT:   %.loc133_4.1: ref %A = deref %a.ref
// CHECK:STDOUT:   %F.ref.loc133: %F.type.1 = name_ref F, @A.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc133_7: <bound method> = bound_method %.loc133_4.1, %F.ref.loc133
// CHECK:STDOUT:   %.loc133_4.2: %.1 = addr_of %.loc133_4.1
// CHECK:STDOUT:   %F.call.loc133: init %.2 = call %.loc133_7(%.loc133_4.2)
// CHECK:STDOUT:   %b.ref: %.7 = name_ref b, %b
// CHECK:STDOUT:   %.loc134_4.1: ref %B = deref %b.ref
// CHECK:STDOUT:   %F.ref.loc134: %F.type.2 = name_ref F, @B.%F.decl [template = constants.%F.2]
// CHECK:STDOUT:   %.loc134_7: <bound method> = bound_method %.loc134_4.1, %F.ref.loc134
// CHECK:STDOUT:   %.loc134_4.2: %.7 = addr_of %.loc134_4.1
// CHECK:STDOUT:   %F.call.loc134: init %.2 = call %.loc134_7(%.loc134_4.2)
// CHECK:STDOUT:   %c.ref: %.13 = name_ref c, %c
// CHECK:STDOUT:   %.loc135_4.1: ref %C = deref %c.ref
// CHECK:STDOUT:   %F.ref.loc135: %F.type.3 = name_ref F, @C.%F.decl [template = constants.%F.3]
// CHECK:STDOUT:   %.loc135_7: <bound method> = bound_method %.loc135_4.1, %F.ref.loc135
// CHECK:STDOUT:   %.loc135_4.2: %.13 = addr_of %.loc135_4.1
// CHECK:STDOUT:   %F.call.loc135: init %.2 = call %.loc135_7(%.loc135_4.2)
// CHECK:STDOUT:   %d.ref: %.17 = name_ref d, %d
// CHECK:STDOUT:   %.loc136_4.1: ref %D = deref %d.ref
// CHECK:STDOUT:   %F.ref.loc136: %F.type.2 = name_ref F, @B.%F.decl [template = constants.%F.2]
// CHECK:STDOUT:   %.loc136_7: <bound method> = bound_method %.loc136_4.1, %F.ref.loc136
// CHECK:STDOUT:   %.loc136_4.2: %.17 = addr_of %.loc136_4.1
// CHECK:STDOUT:   %.loc136_4.3: ref %D = deref %.loc136_4.2
// CHECK:STDOUT:   %.loc136_4.4: ref %B = class_element_access %.loc136_4.3, element0
// CHECK:STDOUT:   %.loc136_4.5: %.7 = addr_of %.loc136_4.4
// CHECK:STDOUT:   %.loc136_4.6: %.7 = converted %.loc136_4.2, %.loc136_4.5
// CHECK:STDOUT:   %F.call.loc136: init %.2 = call %.loc136_7(%.loc136_4.6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
