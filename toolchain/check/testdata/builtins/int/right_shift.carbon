// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/right_shift.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/right_shift.carbon
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/io.carbon:18:26: error: member access into undefined facet type `Negate` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn EOF() -> i32 { return -1; }
// CHECK:STDERR:                          ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/io.carbon:9:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Negate {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:115:13: error: member access into undefined facet type `Add` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self + other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:10:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Add {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:122:13: error: member access into undefined facet type `BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self & other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:15:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:129:13: error: member access into undefined facet type `BitOr` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self | other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitOr {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:136:13: error: member access into undefined facet type `BitXor` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self ^ other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:35:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitXor {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:143:13: error: member access into undefined facet type `Div` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self / other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:55:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Div {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:150:13: error: member access into undefined facet type `LeftShift` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self << other;
// CHECK:STDERR:             ^~~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:45:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface LeftShift {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:157:13: error: member access into undefined facet type `Mod` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self % other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:65:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Mod {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:164:13: error: member access into undefined facet type `Mul` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self * other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:45:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Mul {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:171:13: error: member access into undefined facet type `RightShift` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self >> other;
// CHECK:STDERR:             ^~~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:55:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface RightShift {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:178:13: error: member access into undefined facet type `Sub` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self - other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:30:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Sub {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:187:5: error: member access into undefined facet type `SubAssign` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self -= 1;
// CHECK:STDERR:     ^~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:35:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface SubAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:194:5: error: member access into undefined facet type `AddAssign` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self += 1;
// CHECK:STDERR:     ^~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:15:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface AddAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:126:13: error: member access into undefined facet type `Add` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self + other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:10:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Add {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:133:13: error: member access into undefined facet type `BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self & other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:15:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:140:13: error: member access into undefined facet type `BitOr` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self | other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitOr {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:147:13: error: member access into undefined facet type `BitXor` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self ^ other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:35:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitXor {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:154:13: error: member access into undefined facet type `Div` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self / other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:55:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Div {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:161:13: error: member access into undefined facet type `LeftShift` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self << other;
// CHECK:STDERR:             ^~~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:45:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface LeftShift {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:168:13: error: member access into undefined facet type `Mod` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self % other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:65:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Mod {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:175:13: error: member access into undefined facet type `Mul` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self * other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:45:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Mul {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:182:13: error: member access into undefined facet type `RightShift` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self >> other;
// CHECK:STDERR:             ^~~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:55:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface RightShift {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:189:13: error: member access into undefined facet type `Sub` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self - other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:30:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Sub {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:198:5: error: member access into undefined facet type `SubAssign` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self -= 1;
// CHECK:STDERR:     ^~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:35:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface SubAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:205:5: error: member access into undefined facet type `AddAssign` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self += 1;
// CHECK:STDERR:     ^~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:15:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface AddAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:

// --- i32.carbon

library "[[@TEST_NAME]]";

class Expect(N:! i32) {}
fn Test(N:! i32) -> Expect(N) { return {}; }

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";

fn F() {
  // CHECK:STDERR: i32.carbon:[[@LINE+30]]:19: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(0, 31)) as Expect(0);
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-9]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+20]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(0, 31)) as Expect(0);
  // CHECK:STDERR:                      ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-19]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+10]]:30: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(0, 31)) as Expect(0);
  // CHECK:STDERR:                              ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-32]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! i32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(RightShift(0, 31)) as Expect(0);
  // CHECK:STDERR: i32.carbon:[[@LINE+30]]:19: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 31)) as Expect(0);
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-40]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+20]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 31)) as Expect(0);
  // CHECK:STDERR:                      ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-50]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+10]]:30: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 31)) as Expect(0);
  // CHECK:STDERR:                              ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-63]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! i32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(RightShift(1, 31)) as Expect(0);
  // CHECK:STDERR: i32.carbon:[[@LINE+30]]:19: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 0)) as Expect(1);
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-71]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+20]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 0)) as Expect(1);
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-81]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+10]]:29: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 0)) as Expect(1);
  // CHECK:STDERR:                             ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-94]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! i32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(RightShift(1, 0)) as Expect(1);
  // CHECK:STDERR: i32.carbon:[[@LINE+30]]:19: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 2)) as Expect(0);
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-102]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+20]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 2)) as Expect(0);
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-112]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+10]]:29: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 2)) as Expect(0);
  // CHECK:STDERR:                             ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-125]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! i32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(RightShift(1, 2)) as Expect(0);
  // CHECK:STDERR: i32.carbon:[[@LINE+30]]:19: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(22, 2)) as Expect(5);
  // CHECK:STDERR:                   ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-133]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+20]]:23: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(22, 2)) as Expect(5);
  // CHECK:STDERR:                       ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-143]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+10]]:30: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(22, 2)) as Expect(5);
  // CHECK:STDERR:                              ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-156]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! i32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(RightShift(22, 2)) as Expect(5);
  // CHECK:STDERR: i32.carbon:[[@LINE+24]]:19: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-1, 1)) as Expect(-1);
  // CHECK:STDERR:                   ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+17]]:23: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-1, 1)) as Expect(-1);
  // CHECK:STDERR:                       ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-171]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-1, 1)) as Expect(-1);
  // CHECK:STDERR:                                     ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(RightShift(-1, 1)) as Expect(-1);
  // CHECK:STDERR: i32.carbon:[[@LINE+24]]:19: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-2, 1)) as Expect(-1);
  // CHECK:STDERR:                   ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+17]]:23: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-2, 1)) as Expect(-1);
  // CHECK:STDERR:                       ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-196]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-2, 1)) as Expect(-1);
  // CHECK:STDERR:                                     ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(RightShift(-2, 1)) as Expect(-1);
  // CHECK:STDERR: i32.carbon:[[@LINE+24]]:19: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-10, 2)) as Expect(-3);
  // CHECK:STDERR:                   ^~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+17]]:24: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-10, 2)) as Expect(-3);
  // CHECK:STDERR:                        ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-221]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+7]]:38: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-10, 2)) as Expect(-3);
  // CHECK:STDERR:                                      ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(RightShift(-10, 2)) as Expect(-3);
}

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  return RightShift(a, b);
}

// --- u32.carbon

library "[[@TEST_NAME]]";

class Expect(N:! u32) {}
fn Test(N:! u32) -> Expect(N) { return {}; }

fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";

fn F() {
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:19: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(0, 31)) as Expect(0);
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-9]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(0, 31)) as Expect(0);
  // CHECK:STDERR:                      ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-19]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:30: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(0, 31)) as Expect(0);
  // CHECK:STDERR:                              ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-32]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(RightShift(0, 31)) as Expect(0);
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:19: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 31)) as Expect(0);
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-40]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 31)) as Expect(0);
  // CHECK:STDERR:                      ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-50]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:30: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 31)) as Expect(0);
  // CHECK:STDERR:                              ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-63]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(RightShift(1, 31)) as Expect(0);
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:19: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 0)) as Expect(1);
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-71]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 0)) as Expect(1);
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-81]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:29: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 0)) as Expect(1);
  // CHECK:STDERR:                             ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-94]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(RightShift(1, 0)) as Expect(1);
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:19: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 2)) as Expect(0);
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-102]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 2)) as Expect(0);
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-112]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:29: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(1, 2)) as Expect(0);
  // CHECK:STDERR:                             ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-125]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(RightShift(1, 2)) as Expect(0);
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:19: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(22, 2)) as Expect(5);
  // CHECK:STDERR:                   ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-133]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:23: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(22, 2)) as Expect(5);
  // CHECK:STDERR:                       ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-143]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:30: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(22, 2)) as Expect(5);
  // CHECK:STDERR:                              ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-156]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(RightShift(22, 2)) as Expect(5);
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:19: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(0xFFFF_FFFF, 1)) as Expect(0x7FFF_FFFF);
  // CHECK:STDERR:                   ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-164]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:32: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(0xFFFF_FFFF, 1)) as Expect(0x7FFF_FFFF);
  // CHECK:STDERR:                                ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-174]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:39: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(0xFFFF_FFFF, 1)) as Expect(0x7FFF_FFFF);
  // CHECK:STDERR:                                       ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-187]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(RightShift(0xFFFF_FFFF, 1)) as Expect(0x7FFF_FFFF);
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:19: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(0xABCD_EF01, 8)) as Expect(0xAB_CDEF);
  // CHECK:STDERR:                   ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-195]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:32: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(0xABCD_EF01, 8)) as Expect(0xAB_CDEF);
  // CHECK:STDERR:                                ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-205]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:39: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(0xABCD_EF01, 8)) as Expect(0xAB_CDEF);
  // CHECK:STDERR:                                       ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-218]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(RightShift(0xABCD_EF01, 8)) as Expect(0xAB_CDEF);
}

fn RuntimeCall(a: u32, b: i32) -> u32 {
  return RightShift(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn RightShift(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.right_shift";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(RightShift(0, 31)) as Expect(0);
  Test(RightShift(1, 31)) as Expect(0);
  Test(RightShift(1, 0)) as Expect(1);
  Test(RightShift(1, 2)) as Expect(0);
  Test(RightShift(22, 2)) as Expect(5);
  // CHECK:STDERR: literal.carbon:[[@LINE+14]]:19: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-1, 1)) as Expect(-1);
  // CHECK:STDERR:                   ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: literal.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-1, 1)) as Expect(-1);
  // CHECK:STDERR:                                     ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(RightShift(-1, 1)) as Expect(-1);
  // CHECK:STDERR: literal.carbon:[[@LINE+14]]:19: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-2, 1)) as Expect(-1);
  // CHECK:STDERR:                   ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: literal.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-2, 1)) as Expect(-1);
  // CHECK:STDERR:                                     ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(RightShift(-2, 1)) as Expect(-1);
  // CHECK:STDERR: literal.carbon:[[@LINE+14]]:19: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-10, 2)) as Expect(-3);
  // CHECK:STDERR:                   ^~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: literal.carbon:[[@LINE+7]]:38: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-10, 2)) as Expect(-3);
  // CHECK:STDERR:                                      ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(RightShift(-10, 2)) as Expect(-3);
  Test(RightShift(0xFFFF_FFFF, 1)) as Expect(0x7FFF_FFFF);
  Test(RightShift(0xABCD_EF01, 8)) as Expect(0xAB_CDEF);

  Test(RightShift(0x1234_5678, 1_000_000_000)) as Expect(0);
  // CHECK:STDERR: literal.carbon:[[@LINE+14]]:19: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-0x1234_5678, 1_000_000_000)) as Expect(-1);
  // CHECK:STDERR:                   ^~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: literal.carbon:[[@LINE+7]]:59: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-0x1234_5678, 1_000_000_000)) as Expect(-1);
  // CHECK:STDERR:                                                           ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(RightShift(-0x1234_5678, 1_000_000_000)) as Expect(-1);
  Test(RightShift(0xFFFF_FFFF_FFFF_FFFF, 1_000_000_000)) as Expect(0);
  Test(RightShift(0x7FFF_FFFF_FFFF_FFFF, 1_000_000_000)) as Expect(0);
  // CHECK:STDERR: literal.carbon:[[@LINE+14]]:19: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-0x7FFF_FFFF_FFFF_FFFF, 1_000_000_000)) as Expect(-1);
  // CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: literal.carbon:[[@LINE+7]]:69: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-0x7FFF_FFFF_FFFF_FFFF, 1_000_000_000)) as Expect(-1);
  // CHECK:STDERR:                                                                     ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(RightShift(-0x7FFF_FFFF_FFFF_FFFF, 1_000_000_000)) as Expect(-1);
  // CHECK:STDERR: literal.carbon:[[@LINE+14]]:19: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-0x8000_0000_0000_0000, 1_000_000_000)) as Expect(-1);
  // CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: literal.carbon:[[@LINE+7]]:69: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(RightShift(-0x8000_0000_0000_0000, 1_000_000_000)) as Expect(-1);
  // CHECK:STDERR:                                                                     ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(RightShift(-0x8000_0000_0000_0000, 1_000_000_000)) as Expect(-1);
}

// --- fail_bad_shift.carbon

library "[[@TEST_NAME]]";

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
fn RightShiftLit(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.right_shift";

// Shift greater than size is disallowed for sized types.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+20]]:30: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let size_1: i32 = RightShift(1, 31);
// CHECK:STDERR:                              ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-10]]:15: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
// CHECK:STDERR:               ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+10]]:33: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let size_1: i32 = RightShift(1, 31);
// CHECK:STDERR:                                 ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-20]]:23: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
// CHECK:STDERR:                       ^~~~~~
// CHECK:STDERR:
let size_1: i32 = RightShift(1, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+20]]:30: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let size_2: i32 = RightShift(1, 32);
// CHECK:STDERR:                              ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-31]]:15: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
// CHECK:STDERR:               ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+10]]:33: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let size_2: i32 = RightShift(1, 32);
// CHECK:STDERR:                                 ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-41]]:23: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
// CHECK:STDERR:                       ^~~~~~
// CHECK:STDERR:
let size_2: i32 = RightShift(1, 32);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+20]]:30: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let size_3: i32 = RightShift(1, 33);
// CHECK:STDERR:                              ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-52]]:15: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
// CHECK:STDERR:               ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+10]]:33: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let size_3: i32 = RightShift(1, 33);
// CHECK:STDERR:                                 ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-62]]:23: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
// CHECK:STDERR:                       ^~~~~~
// CHECK:STDERR:
let size_3: i32 = RightShift(1, 33);

// Negative shifts aren't allowed either, even for literals, even if the lhs is zero.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+17]]:35: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let negative: i32 = RightShift(1, -1);
// CHECK:STDERR:                                   ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Negate {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+10]]:32: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let negative: i32 = RightShift(1, -1);
// CHECK:STDERR:                                ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-82]]:15: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
// CHECK:STDERR:               ^~~~~~
// CHECK:STDERR:
let negative: i32 = RightShift(1, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+17]]:40: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let negative_zero: i32 = RightShift(0, -1);
// CHECK:STDERR:                                        ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Negate {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+10]]:37: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let negative_zero: i32 = RightShift(0, -1);
// CHECK:STDERR:                                     ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-100]]:15: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
// CHECK:STDERR:               ^~~~~~
// CHECK:STDERR:
let negative_zero: i32 = RightShift(0, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+7]]:56: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let negative_lit: Core.IntLiteral() = RightShiftLit(1, -1);
// CHECK:STDERR:                                                        ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Negate {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative_lit: Core.IntLiteral() = RightShiftLit(1, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+7]]:61: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let negative_lit_zero: Core.IntLiteral() = RightShiftLit(0, -1);
// CHECK:STDERR:                                                             ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Negate {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative_lit_zero: Core.IntLiteral() = RightShiftLit(0, -1);
