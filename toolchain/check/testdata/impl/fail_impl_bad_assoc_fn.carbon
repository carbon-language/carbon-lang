// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_impl_bad_assoc_fn.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_impl_bad_assoc_fn.carbon
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

interface I { fn F(); }

class NoF {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+4]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as I {}
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as I {}
}

class FNotFunction {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+4]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as I {
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as I {
    class F;
  }
}

fn PossiblyF();

// TODO: Should this be permitted?
class FAlias {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+4]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as I {
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as I {
    alias F = PossiblyF;
  }
}

class FExtraParam {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+4]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as I {
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as I {
    fn F(b: bool);
  }
}

class FExtraImplicitParam {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+4]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as I {
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as I {
    fn F[self: Self]();
  }
}

// TODO: Should this be permitted?
class FExtraReturnType {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+4]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as I {
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as I {
    fn F() -> bool;
  }
}

interface J { fn F[self: bool](b: bool) -> bool; }

class FMissingParam {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+4]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as J {
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as J {
    fn F[self: bool]() -> bool;
  }
}

class FMissingImplicitParam {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+4]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as J {
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as J {
    fn F(b: bool) -> bool;
  }
}

class FMissingReturnType {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+4]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as J {
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as J {
    fn F[self: bool](b: bool);
  }
}

class FDifferentParamType {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+4]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as J {
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as J {
    fn F[self: bool](b: Self) -> bool;
  }
}

class FDifferentImplicitParamType {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+4]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as J {
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as J {
    fn F[self: Self](b: bool) -> bool;
  }
}

class FDifferentReturnType {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+4]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as J {
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as J {
    fn F[self: bool](b: bool) -> Self;
  }
}

// TODO: This should probably be permitted.
class FDifferentParamName {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+4]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as J {
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as J {
    fn F[self: bool](not_b: bool) -> bool;
  }
}

interface SelfNested {
  fn F(x: (Self*, {.x: Self, .y: i32})) -> [Self; 4];
}

class SelfNestedBadParam {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+4]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as SelfNested {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl as SelfNested {
    fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> [SelfNestedBadParam; 4];
  }
}

class SelfNestedBadReturnType {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+3]]:3: error: impl as non-facet-type [ImplAsNonFacetType]
  // CHECK:STDERR:   impl as SelfNested {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  impl as SelfNested {
    fn F(x: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> [SelfNestedBadParam; 4];
  }
}

// CHECK:STDOUT: --- fail_impl_bad_assoc_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = interface_type @I [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %I.type, %F.type.1 [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %NoF: type = class_type @NoF [template]
// CHECK:STDOUT:   %.4: type = struct_type {} [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [template]
// CHECK:STDOUT:   %FNotFunction: type = class_type @FNotFunction [template]
// CHECK:STDOUT:   %F.2: type = class_type @F.16 [template]
// CHECK:STDOUT:   %PossiblyF.type: type = fn_type @PossiblyF [template]
// CHECK:STDOUT:   %PossiblyF: %PossiblyF.type = struct_value () [template]
// CHECK:STDOUT:   %FAlias: type = class_type @FAlias [template]
// CHECK:STDOUT:   %FExtraParam: type = class_type @FExtraParam [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.3: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %FExtraImplicitParam: type = class_type @FExtraImplicitParam [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.4: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %FExtraReturnType: type = class_type @FExtraReturnType [template]
// CHECK:STDOUT:   %F.type.4: type = fn_type @F.4 [template]
// CHECK:STDOUT:   %F.5: %F.type.4 = struct_value () [template]
// CHECK:STDOUT:   %J.type: type = interface_type @J [template]
// CHECK:STDOUT:   %Self.2: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.5: type = fn_type @F.5 [template]
// CHECK:STDOUT:   %F.6: %F.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %J.type, %F.type.5 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, @J.%F.decl [template]
// CHECK:STDOUT:   %FMissingParam: type = class_type @FMissingParam [template]
// CHECK:STDOUT:   %F.type.6: type = fn_type @F.6 [template]
// CHECK:STDOUT:   %F.7: %F.type.6 = struct_value () [template]
// CHECK:STDOUT:   %FMissingImplicitParam: type = class_type @FMissingImplicitParam [template]
// CHECK:STDOUT:   %F.type.7: type = fn_type @F.7 [template]
// CHECK:STDOUT:   %F.8: %F.type.7 = struct_value () [template]
// CHECK:STDOUT:   %FMissingReturnType: type = class_type @FMissingReturnType [template]
// CHECK:STDOUT:   %F.type.8: type = fn_type @F.8 [template]
// CHECK:STDOUT:   %F.9: %F.type.8 = struct_value () [template]
// CHECK:STDOUT:   %FDifferentParamType: type = class_type @FDifferentParamType [template]
// CHECK:STDOUT:   %F.type.9: type = fn_type @F.9 [template]
// CHECK:STDOUT:   %F.10: %F.type.9 = struct_value () [template]
// CHECK:STDOUT:   %FDifferentImplicitParamType: type = class_type @FDifferentImplicitParamType [template]
// CHECK:STDOUT:   %F.type.10: type = fn_type @F.10 [template]
// CHECK:STDOUT:   %F.11: %F.type.10 = struct_value () [template]
// CHECK:STDOUT:   %FDifferentReturnType: type = class_type @FDifferentReturnType [template]
// CHECK:STDOUT:   %F.type.11: type = fn_type @F.11 [template]
// CHECK:STDOUT:   %F.12: %F.type.11 = struct_value () [template]
// CHECK:STDOUT:   %FDifferentParamName: type = class_type @FDifferentParamName [template]
// CHECK:STDOUT:   %F.type.12: type = fn_type @F.12 [template]
// CHECK:STDOUT:   %F.13: %F.type.12 = struct_value () [template]
// CHECK:STDOUT:   %SelfNested.type: type = interface_type @SelfNested [template]
// CHECK:STDOUT:   %Self.3: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.8: type = ptr_type %Self.3 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = struct_type {.x: %Self.3, .y: i32} [symbolic]
// CHECK:STDOUT:   %.10: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.11: type = tuple_type (%.8, %.9) [symbolic]
// CHECK:STDOUT:   %.12: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.13: type = array_type %.12, %Self.3 [symbolic]
// CHECK:STDOUT:   %F.type.13: type = fn_type @F.13 [template]
// CHECK:STDOUT:   %F.14: %F.type.13 = struct_value () [template]
// CHECK:STDOUT:   %.14: type = assoc_entity_type %SelfNested.type, %F.type.13 [template]
// CHECK:STDOUT:   %.15: %.14 = assoc_entity element0, @SelfNested.%F.decl [template]
// CHECK:STDOUT:   %SelfNestedBadParam: type = class_type @SelfNestedBadParam [template]
// CHECK:STDOUT:   %.16: type = ptr_type %SelfNestedBadParam [template]
// CHECK:STDOUT:   %.17: type = struct_type {.x: i32, .y: i32} [template]
// CHECK:STDOUT:   %.18: type = tuple_type (%.16, %.17) [template]
// CHECK:STDOUT:   %.19: type = array_type %.12, %SelfNestedBadParam [template]
// CHECK:STDOUT:   %F.type.14: type = fn_type @F.14 [template]
// CHECK:STDOUT:   %F.15: %F.type.14 = struct_value () [template]
// CHECK:STDOUT:   %SelfNestedBadReturnType: type = class_type @SelfNestedBadReturnType [template]
// CHECK:STDOUT:   %.20: type = ptr_type %SelfNestedBadReturnType [template]
// CHECK:STDOUT:   %.21: type = struct_type {.x: %SelfNestedBadReturnType, .y: i32} [template]
// CHECK:STDOUT:   %.22: type = tuple_type (%.20, %.21) [template]
// CHECK:STDOUT:   %F.type.15: type = fn_type @F.15 [template]
// CHECK:STDOUT:   %F.16: %F.type.15 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Int32 = %import_ref.2
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
// CHECK:STDOUT:   %import_ref.1: %Bool.type = import_ref Core//prelude/types/bool, inst+5, loaded [template = constants.%Bool]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .NoF = %NoF.decl
// CHECK:STDOUT:     .FNotFunction = %FNotFunction.decl
// CHECK:STDOUT:     .PossiblyF = %PossiblyF.decl
// CHECK:STDOUT:     .FAlias = %FAlias.decl
// CHECK:STDOUT:     .FExtraParam = %FExtraParam.decl
// CHECK:STDOUT:     .FExtraImplicitParam = %FExtraImplicitParam.decl
// CHECK:STDOUT:     .FExtraReturnType = %FExtraReturnType.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .FMissingParam = %FMissingParam.decl
// CHECK:STDOUT:     .FMissingImplicitParam = %FMissingImplicitParam.decl
// CHECK:STDOUT:     .FMissingReturnType = %FMissingReturnType.decl
// CHECK:STDOUT:     .FDifferentParamType = %FDifferentParamType.decl
// CHECK:STDOUT:     .FDifferentImplicitParamType = %FDifferentImplicitParamType.decl
// CHECK:STDOUT:     .FDifferentReturnType = %FDifferentReturnType.decl
// CHECK:STDOUT:     .FDifferentParamName = %FDifferentParamName.decl
// CHECK:STDOUT:     .SelfNested = %SelfNested.decl
// CHECK:STDOUT:     .SelfNestedBadParam = %SelfNestedBadParam.decl
// CHECK:STDOUT:     .SelfNestedBadReturnType = %SelfNestedBadReturnType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   %NoF.decl: type = class_decl @NoF [template = constants.%NoF] {} {}
// CHECK:STDOUT:   %FNotFunction.decl: type = class_decl @FNotFunction [template = constants.%FNotFunction] {} {}
// CHECK:STDOUT:   %PossiblyF.decl: %PossiblyF.type = fn_decl @PossiblyF [template = constants.%PossiblyF] {} {}
// CHECK:STDOUT:   %FAlias.decl: type = class_decl @FAlias [template = constants.%FAlias] {} {}
// CHECK:STDOUT:   %FExtraParam.decl: type = class_decl @FExtraParam [template = constants.%FExtraParam] {} {}
// CHECK:STDOUT:   %FExtraImplicitParam.decl: type = class_decl @FExtraImplicitParam [template = constants.%FExtraImplicitParam] {} {}
// CHECK:STDOUT:   %FExtraReturnType.decl: type = class_decl @FExtraReturnType [template = constants.%FExtraReturnType] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%J.type] {} {}
// CHECK:STDOUT:   %FMissingParam.decl: type = class_decl @FMissingParam [template = constants.%FMissingParam] {} {}
// CHECK:STDOUT:   %FMissingImplicitParam.decl: type = class_decl @FMissingImplicitParam [template = constants.%FMissingImplicitParam] {} {}
// CHECK:STDOUT:   %FMissingReturnType.decl: type = class_decl @FMissingReturnType [template = constants.%FMissingReturnType] {} {}
// CHECK:STDOUT:   %FDifferentParamType.decl: type = class_decl @FDifferentParamType [template = constants.%FDifferentParamType] {} {}
// CHECK:STDOUT:   %FDifferentImplicitParamType.decl: type = class_decl @FDifferentImplicitParamType [template = constants.%FDifferentImplicitParamType] {} {}
// CHECK:STDOUT:   %FDifferentReturnType.decl: type = class_decl @FDifferentReturnType [template = constants.%FDifferentReturnType] {} {}
// CHECK:STDOUT:   %FDifferentParamName.decl: type = class_decl @FDifferentParamName [template = constants.%FDifferentParamName] {} {}
// CHECK:STDOUT:   %SelfNested.decl: type = interface_decl @SelfNested [template = constants.%SelfNested.type] {} {}
// CHECK:STDOUT:   %SelfNestedBadParam.decl: type = class_decl @SelfNestedBadParam [template = constants.%SelfNestedBadParam] {} {}
// CHECK:STDOUT:   %SelfNestedBadReturnType.decl: type = class_decl @SelfNestedBadReturnType [template = constants.%SelfNestedBadReturnType] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc115: %.2 = assoc_entity element0, %F.decl [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc115
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %F.decl: %F.type.5 = fn_decl @F.5 [template = constants.%F.6] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc179_26: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc179_26.1: type = value_of_initializer %bool.make_type.loc179_26 [template = bool]
// CHECK:STDOUT:     %.loc179_26.2: type = converted %bool.make_type.loc179_26, %.loc179_26.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc179_35: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc179_35.1: type = value_of_initializer %bool.make_type.loc179_35 [template = bool]
// CHECK:STDOUT:     %.loc179_35.2: type = converted %bool.make_type.loc179_35, %.loc179_35.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc179_44: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc179_44.1: type = value_of_initializer %bool.make_type.loc179_44 [template = bool]
// CHECK:STDOUT:     %.loc179_44.2: type = converted %bool.make_type.loc179_44, %.loc179_44.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc179: %.6 = assoc_entity element0, %F.decl [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc179
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.3]
// CHECK:STDOUT:   %F.decl: %F.type.13 = fn_decl @F.13 [template = constants.%F.14] {
// CHECK:STDOUT:     %x.patt: @F.13.%.loc253_38.1 (%.11) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.13.%.loc253_38.1 (%.11) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.13.%.loc253_52.1 (%.13) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.13.%.loc253_52.1 (%.13) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc253_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc253_16.2: type = facet_type_access %Self.ref.loc253_12 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc253_16.3: type = converted %Self.ref.loc253_12, %.loc253_16.2 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc253_16.4: type = ptr_type %Self.3 [symbolic = %.loc253_16.1 (constants.%.8)]
// CHECK:STDOUT:     %Self.ref.loc253_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc253_24.1: type = facet_type_access %Self.ref.loc253_24 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc253_24.2: type = converted %Self.ref.loc253_24, %.loc253_24.1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc253_34.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc253_34.2: type = converted %int.make_type_32, %.loc253_34.1 [template = i32]
// CHECK:STDOUT:     %.loc253_37.2: type = struct_type {.x: %Self.3, .y: i32} [symbolic = %.loc253_37.1 (constants.%.9)]
// CHECK:STDOUT:     %.loc253_38.2: %.10 = tuple_literal (%.loc253_16.4, %.loc253_37.2)
// CHECK:STDOUT:     %.loc253_38.3: type = converted %.loc253_38.2, constants.%.11 [symbolic = %.loc253_38.1 (constants.%.11)]
// CHECK:STDOUT:     %Self.ref.loc253_45: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc253_51: i32 = int_literal 4 [template = constants.%.12]
// CHECK:STDOUT:     %.loc253_45.1: type = facet_type_access %Self.ref.loc253_45 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc253_45.2: type = converted %Self.ref.loc253_45, %.loc253_45.1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc253_52.2: type = array_type %.loc253_51, %Self.3 [symbolic = %.loc253_52.1 (constants.%.13)]
// CHECK:STDOUT:     %x.param: @F.13.%.loc253_38.1 (%.11) = value_param runtime_param0
// CHECK:STDOUT:     %x: @F.13.%.loc253_38.1 (%.11) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @F.13.%.loc253_52.1 (%.13) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.13.%.loc253_52.1 (%.13) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc253: %.14 = assoc_entity element0, %F.decl [template = constants.%.15]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc253
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: type = class_decl @F.16 [template = constants.%F.2] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %Self.ref as %I.ref {
// CHECK:STDOUT:   %PossiblyF.ref: %PossiblyF.type = name_ref PossiblyF, file.%PossiblyF.decl [template = constants.%PossiblyF]
// CHECK:STDOUT:   %F: %PossiblyF.type = bind_alias F, file.%PossiblyF.decl [template = constants.%PossiblyF]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.3] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc154_13.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc154_13.2: type = converted %bool.make_type, %.loc154_13.1 [template = bool]
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.3 [template = constants.%F.4] {
// CHECK:STDOUT:     %self.patt: %FExtraImplicitParam = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %FExtraImplicitParam = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [template = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %self.param: %FExtraImplicitParam = value_param runtime_param0
// CHECK:STDOUT:     %self: %FExtraImplicitParam = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.4 = fn_decl @F.4 [template = constants.%F.5] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc175_15.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc175_15.2: type = converted %bool.make_type, %.loc175_15.1 [template = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.6 = fn_decl @F.6 [template = constants.%F.7] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc187_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc187_16.1: type = value_of_initializer %bool.make_type.loc187_16 [template = bool]
// CHECK:STDOUT:     %.loc187_16.2: type = converted %bool.make_type.loc187_16, %.loc187_16.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc187_27: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc187_27.1: type = value_of_initializer %bool.make_type.loc187_27 [template = bool]
// CHECK:STDOUT:     %.loc187_27.2: type = converted %bool.make_type.loc187_27, %.loc187_27.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.7 = fn_decl @F.7 [template = constants.%F.8] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc197_13: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc197_13.1: type = value_of_initializer %bool.make_type.loc197_13 [template = bool]
// CHECK:STDOUT:     %.loc197_13.2: type = converted %bool.make_type.loc197_13, %.loc197_13.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc197_22: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc197_22.1: type = value_of_initializer %bool.make_type.loc197_22 [template = bool]
// CHECK:STDOUT:     %.loc197_22.2: type = converted %bool.make_type.loc197_22, %.loc197_22.1 [template = bool]
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.8 = fn_decl @F.8 [template = constants.%F.9] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc207_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc207_16.1: type = value_of_initializer %bool.make_type.loc207_16 [template = bool]
// CHECK:STDOUT:     %.loc207_16.2: type = converted %bool.make_type.loc207_16, %.loc207_16.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc207_25: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc207_25.1: type = value_of_initializer %bool.make_type.loc207_25 [template = bool]
// CHECK:STDOUT:     %.loc207_25.2: type = converted %bool.make_type.loc207_25, %.loc207_25.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.10: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.9 = fn_decl @F.9 [template = constants.%F.10] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %FDifferentParamType = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %FDifferentParamType = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc217_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc217_16.1: type = value_of_initializer %bool.make_type.loc217_16 [template = bool]
// CHECK:STDOUT:     %.loc217_16.2: type = converted %bool.make_type.loc217_16, %.loc217_16.1 [template = bool]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [template = constants.%FDifferentParamType]
// CHECK:STDOUT:     %bool.make_type.loc217_34: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc217_34.1: type = value_of_initializer %bool.make_type.loc217_34 [template = bool]
// CHECK:STDOUT:     %.loc217_34.2: type = converted %bool.make_type.loc217_34, %.loc217_34.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: %FDifferentParamType = value_param runtime_param1
// CHECK:STDOUT:     %b: %FDifferentParamType = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.11: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.10 = fn_decl @F.10 [template = constants.%F.11] {
// CHECK:STDOUT:     %self.patt: %FDifferentImplicitParamType = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %FDifferentImplicitParamType = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [template = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %bool.make_type.loc227_25: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc227_25.1: type = value_of_initializer %bool.make_type.loc227_25 [template = bool]
// CHECK:STDOUT:     %.loc227_25.2: type = converted %bool.make_type.loc227_25, %.loc227_25.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc227_34: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc227_34.1: type = value_of_initializer %bool.make_type.loc227_34 [template = bool]
// CHECK:STDOUT:     %.loc227_34.2: type = converted %bool.make_type.loc227_34, %.loc227_34.1 [template = bool]
// CHECK:STDOUT:     %self.param: %FDifferentImplicitParamType = value_param runtime_param0
// CHECK:STDOUT:     %self: %FDifferentImplicitParamType = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.12: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.11 = fn_decl @F.11 [template = constants.%F.12] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %FDifferentReturnType = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %FDifferentReturnType = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc237_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc237_16.1: type = value_of_initializer %bool.make_type.loc237_16 [template = bool]
// CHECK:STDOUT:     %.loc237_16.2: type = converted %bool.make_type.loc237_16, %.loc237_16.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc237_25: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc237_25.1: type = value_of_initializer %bool.make_type.loc237_25 [template = bool]
// CHECK:STDOUT:     %.loc237_25.2: type = converted %bool.make_type.loc237_25, %.loc237_25.1 [template = bool]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [template = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %self.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %FDifferentReturnType = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %FDifferentReturnType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.13: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.12 = fn_decl @F.12 [template = constants.%F.13] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %not_b.patt: bool = binding_pattern not_b
// CHECK:STDOUT:     %not_b.param_patt: bool = value_param_pattern %not_b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc248_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc248_16.1: type = value_of_initializer %bool.make_type.loc248_16 [template = bool]
// CHECK:STDOUT:     %.loc248_16.2: type = converted %bool.make_type.loc248_16, %.loc248_16.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc248_29: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc248_29.1: type = value_of_initializer %bool.make_type.loc248_29 [template = bool]
// CHECK:STDOUT:     %.loc248_29.2: type = converted %bool.make_type.loc248_29, %.loc248_29.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc248_38: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc248_38.1: type = value_of_initializer %bool.make_type.loc248_38 [template = bool]
// CHECK:STDOUT:     %.loc248_38.2: type = converted %bool.make_type.loc248_38, %.loc248_38.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %not_b.param: bool = value_param runtime_param1
// CHECK:STDOUT:     %not_b: bool = bind_name not_b, %not_b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.14: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.14 = fn_decl @F.14 [template = constants.%F.15] {
// CHECK:STDOUT:     %x.patt: %.18 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %.18 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.19 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.19 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc262_14: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %.loc262_32: type = ptr_type %SelfNestedBadParam [template = constants.%.16]
// CHECK:STDOUT:     %int.make_type_32.loc262_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc262_40.1: type = value_of_initializer %int.make_type_32.loc262_40 [template = i32]
// CHECK:STDOUT:     %.loc262_40.2: type = converted %int.make_type_32.loc262_40, %.loc262_40.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc262_49: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc262_49.1: type = value_of_initializer %int.make_type_32.loc262_49 [template = i32]
// CHECK:STDOUT:     %.loc262_49.2: type = converted %int.make_type_32.loc262_49, %.loc262_49.1 [template = i32]
// CHECK:STDOUT:     %.loc262_52: type = struct_type {.x: i32, .y: i32} [template = constants.%.17]
// CHECK:STDOUT:     %.loc262_53.1: %.10 = tuple_literal (%.loc262_32, %.loc262_52)
// CHECK:STDOUT:     %.loc262_53.2: type = converted %.loc262_53.1, constants.%.18 [template = constants.%.18]
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc262_60: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %.loc262_80: i32 = int_literal 4 [template = constants.%.12]
// CHECK:STDOUT:     %.loc262_81: type = array_type %.loc262_80, %SelfNestedBadParam [template = constants.%.19]
// CHECK:STDOUT:     %x.param: %.18 = value_param runtime_param0
// CHECK:STDOUT:     %x: %.18 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %.19 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.19 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.15: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.15 = fn_decl @F.15 [template = constants.%F.16] {
// CHECK:STDOUT:     %x.patt: %.22 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %.22 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.19 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.19 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadReturnType.ref.loc271_14: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [template = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:     %.loc271_37: type = ptr_type %SelfNestedBadReturnType [template = constants.%.20]
// CHECK:STDOUT:     %SelfNestedBadReturnType.ref.loc271_45: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [template = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc271_74.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc271_74.2: type = converted %int.make_type_32, %.loc271_74.1 [template = i32]
// CHECK:STDOUT:     %.loc271_77: type = struct_type {.x: %SelfNestedBadReturnType, .y: i32} [template = constants.%.21]
// CHECK:STDOUT:     %.loc271_78.1: %.10 = tuple_literal (%.loc271_37, %.loc271_77)
// CHECK:STDOUT:     %.loc271_78.2: type = converted %.loc271_78.1, constants.%.22 [template = constants.%.22]
// CHECK:STDOUT:     %SelfNestedBadParam.ref: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %.loc271_105: i32 = int_literal 4 [template = constants.%.12]
// CHECK:STDOUT:     %.loc271_106: type = array_type %.loc271_105, %SelfNestedBadParam [template = constants.%.19]
// CHECK:STDOUT:     %x.param: %.22 = value_param runtime_param0
// CHECK:STDOUT:     %x: %.22 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %.19 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.19 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoF {
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoF [template = constants.%NoF]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc123: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoF
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FNotFunction {
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FNotFunction [template = constants.%FNotFunction]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc133: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FNotFunction
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @F.16;
// CHECK:STDOUT:
// CHECK:STDOUT: class @FAlias {
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FAlias [template = constants.%FAlias]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc146: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FAlias
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraParam {
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraParam [template = constants.%FExtraParam]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc156: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraImplicitParam {
// CHECK:STDOUT:   impl_decl @impl.5 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [template = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc166: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraImplicitParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraReturnType {
// CHECK:STDOUT:   impl_decl @impl.6 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraReturnType [template = constants.%FExtraReturnType]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc177: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraReturnType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingParam {
// CHECK:STDOUT:   impl_decl @impl.7 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingParam [template = constants.%FMissingParam]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc189: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingImplicitParam {
// CHECK:STDOUT:   impl_decl @impl.8 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingImplicitParam [template = constants.%FMissingImplicitParam]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc199: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingImplicitParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingReturnType {
// CHECK:STDOUT:   impl_decl @impl.9 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingReturnType [template = constants.%FMissingReturnType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc209: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingReturnType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentParamType {
// CHECK:STDOUT:   impl_decl @impl.10 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [template = constants.%FDifferentParamType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc219: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentParamType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentImplicitParamType {
// CHECK:STDOUT:   impl_decl @impl.11 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [template = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc229: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentImplicitParamType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentReturnType {
// CHECK:STDOUT:   impl_decl @impl.12 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [template = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc239: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentReturnType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentParamName {
// CHECK:STDOUT:   impl_decl @impl.13 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamName [template = constants.%FDifferentParamName]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc250: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentParamName
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SelfNestedBadParam {
// CHECK:STDOUT:   impl_decl @impl.14 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SelfNestedBadParam [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [template = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc264: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SelfNestedBadParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SelfNestedBadReturnType {
// CHECK:STDOUT:   impl_decl @impl.15 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SelfNestedBadReturnType [template = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [template = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc273: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SelfNestedBadReturnType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PossiblyF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%b.param_patt: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[%self.param_patt: %FExtraImplicitParam]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.4() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.5(@J.%Self: %J.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: bool](%b.param_patt: bool) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.6[%self.param_patt: bool]() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.7(%b.param_patt: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.8[%self.param_patt: bool](%b.param_patt: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.9[%self.param_patt: bool](%b.param_patt: %FDifferentParamType) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.10[%self.param_patt: %FDifferentImplicitParamType](%b.param_patt: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.11[%self.param_patt: bool](%b.param_patt: bool) -> %FDifferentReturnType;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.12[%self.param_patt: bool](%not_b.param_patt: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.13(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %.loc253_16.1: type = ptr_type @F.13.%Self (%Self.3) [symbolic = %.loc253_16.1 (constants.%.8)]
// CHECK:STDOUT:   %.loc253_37.1: type = struct_type {.x: @F.13.%Self (%Self.3), .y: i32} [symbolic = %.loc253_37.1 (constants.%.9)]
// CHECK:STDOUT:   %.loc253_38.1: type = tuple_type (@F.13.%.loc253_16.1 (%.8), @F.13.%.loc253_37.1 (%.9)) [symbolic = %.loc253_38.1 (constants.%.11)]
// CHECK:STDOUT:   %.loc253_52.1: type = array_type constants.%.12, @F.13.%Self (%Self.3) [symbolic = %.loc253_52.1 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: @F.13.%.loc253_38.1 (%.11)) -> @F.13.%.loc253_52.1 (%.13);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.14(%x.param_patt: %.18) -> %.19;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.15(%x.param_patt: %.22) -> %.19;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.13(constants.%Self.3) {
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %.loc253_16.1 => constants.%.8
// CHECK:STDOUT:   %.loc253_37.1 => constants.%.9
// CHECK:STDOUT:   %.loc253_38.1 => constants.%.11
// CHECK:STDOUT:   %.loc253_52.1 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
