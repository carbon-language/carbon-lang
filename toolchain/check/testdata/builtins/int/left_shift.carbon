// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/left_shift.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/left_shift.carbon
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

fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";

fn F() {
  // CHECK:STDERR: i32.carbon:[[@LINE+30]]:18: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 0)) as Expect(0);
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-9]]:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+20]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 0)) as Expect(0);
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-19]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+10]]:28: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 0)) as Expect(0);
  // CHECK:STDERR:                            ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-32]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! i32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(LeftShift(0, 0)) as Expect(0);
  // CHECK:STDERR: i32.carbon:[[@LINE+30]]:18: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 1)) as Expect(0);
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-40]]:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+20]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 1)) as Expect(0);
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-50]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+10]]:28: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 1)) as Expect(0);
  // CHECK:STDERR:                            ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-63]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! i32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(LeftShift(0, 1)) as Expect(0);
  // CHECK:STDERR: i32.carbon:[[@LINE+30]]:18: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 30)) as Expect(0);
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-71]]:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+20]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 30)) as Expect(0);
  // CHECK:STDERR:                     ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-81]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+10]]:29: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 30)) as Expect(0);
  // CHECK:STDERR:                             ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-94]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! i32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(LeftShift(0, 30)) as Expect(0);
  // CHECK:STDERR: i32.carbon:[[@LINE+30]]:18: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(1, 30)) as Expect(0x4000_0000);
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-102]]:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+20]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(1, 30)) as Expect(0x4000_0000);
  // CHECK:STDERR:                     ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-112]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+10]]:29: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(1, 30)) as Expect(0x4000_0000);
  // CHECK:STDERR:                             ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-125]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! i32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(LeftShift(1, 30)) as Expect(0x4000_0000);
  // CHECK:STDERR: i32.carbon:[[@LINE+30]]:18: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(5, 2)) as Expect(20);
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-133]]:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+20]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(5, 2)) as Expect(20);
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-143]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+10]]:28: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(5, 2)) as Expect(20);
  // CHECK:STDERR:                            ^~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-156]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! i32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(LeftShift(5, 2)) as Expect(20);
  // CHECK:STDERR: i32.carbon:[[@LINE+24]]:18: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-1, 1)) as Expect(-2);
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+17]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-1, 1)) as Expect(-2);
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-171]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-1, 1)) as Expect(-2);
  // CHECK:STDERR:                                    ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(LeftShift(-1, 1)) as Expect(-2);
  // CHECK:STDERR: i32.carbon:[[@LINE+24]]:18: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-2, 1)) as Expect(-4);
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+17]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-2, 1)) as Expect(-4);
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-196]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-2, 1)) as Expect(-4);
  // CHECK:STDERR:                                    ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(LeftShift(-2, 1)) as Expect(-4);
  // CHECK:STDERR: i32.carbon:[[@LINE+24]]:18: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-3, 1)) as Expect(-6);
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+17]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-3, 1)) as Expect(-6);
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: i32.carbon:[[@LINE-221]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: i32.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-3, 1)) as Expect(-6);
  // CHECK:STDERR:                                    ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(LeftShift(-3, 1)) as Expect(-6);
}

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  return LeftShift(a, b);
}

// --- u32.carbon

library "[[@TEST_NAME]]";

class Expect(N:! u32) {}
fn Test(N:! u32) -> Expect(N) { return {}; }

fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";

fn F() {
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:18: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 0)) as Expect(0);
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-9]]:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 0)) as Expect(0);
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-19]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:28: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 0)) as Expect(0);
  // CHECK:STDERR:                            ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-32]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(LeftShift(0, 0)) as Expect(0);
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:18: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 1)) as Expect(0);
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-40]]:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 1)) as Expect(0);
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-50]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:28: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 1)) as Expect(0);
  // CHECK:STDERR:                            ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-63]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(LeftShift(0, 1)) as Expect(0);
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:18: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 30)) as Expect(0);
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-71]]:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 30)) as Expect(0);
  // CHECK:STDERR:                     ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-81]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:29: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0, 30)) as Expect(0);
  // CHECK:STDERR:                             ^~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-94]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(LeftShift(0, 30)) as Expect(0);
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:18: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(1, 30)) as Expect(0x4000_0000);
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-102]]:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(1, 30)) as Expect(0x4000_0000);
  // CHECK:STDERR:                     ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-112]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:29: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(1, 30)) as Expect(0x4000_0000);
  // CHECK:STDERR:                             ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-125]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(LeftShift(1, 30)) as Expect(0x4000_0000);
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:18: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(5, 2)) as Expect(20);
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-133]]:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(5, 2)) as Expect(20);
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-143]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:28: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(5, 2)) as Expect(20);
  // CHECK:STDERR:                            ^~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-156]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(LeftShift(5, 2)) as Expect(20);
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:18: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0xFFFF_FFFF, 1)) as Expect(0xFFFF_FFFE);
  // CHECK:STDERR:                  ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-164]]:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:31: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0xFFFF_FFFF, 1)) as Expect(0xFFFF_FFFE);
  // CHECK:STDERR:                               ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-174]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:38: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0xFFFF_FFFF, 1)) as Expect(0xFFFF_FFFE);
  // CHECK:STDERR:                                      ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-187]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(LeftShift(0xFFFF_FFFF, 1)) as Expect(0xFFFF_FFFE);
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:18: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0xFFFF_FFFE, 1)) as Expect(0xFFFF_FFFC);
  // CHECK:STDERR:                  ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-195]]:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:31: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0xFFFF_FFFE, 1)) as Expect(0xFFFF_FFFC);
  // CHECK:STDERR:                               ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-205]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:38: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0xFFFF_FFFE, 1)) as Expect(0xFFFF_FFFC);
  // CHECK:STDERR:                                      ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-218]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(LeftShift(0xFFFF_FFFE, 1)) as Expect(0xFFFF_FFFC);
  // CHECK:STDERR: u32.carbon:[[@LINE+30]]:18: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0xABCD_EF01, 8)) as Expect(0xCDEF_0100);
  // CHECK:STDERR:                  ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-226]]:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+20]]:31: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0xABCD_EF01, 8)) as Expect(0xCDEF_0100);
  // CHECK:STDERR:                               ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-236]]:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: u32.carbon:[[@LINE+10]]:38: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(0xABCD_EF01, 8)) as Expect(0xCDEF_0100);
  // CHECK:STDERR:                                      ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: u32.carbon:[[@LINE-249]]:14: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class Expect(N:! u32) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Test(LeftShift(0xABCD_EF01, 8)) as Expect(0xCDEF_0100);
}

fn RuntimeCall(a: u32, b: i32) -> u32 {
  return LeftShift(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn LeftShift(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.left_shift";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  // Zero can be shifted by any amount.
  Test(LeftShift(0, 0)) as Expect(0);
  Test(LeftShift(0, 0)) as Expect(0);
  Test(LeftShift(0, 1)) as Expect(0);
  Test(LeftShift(0, 30)) as Expect(0);
  Test(LeftShift(0, 1_000_000_000)) as Expect(0);

  // Positive numbers can be shifted.
  Test(LeftShift(1, 0)) as Expect(1);
  Test(LeftShift(1, 1)) as Expect(2);
  Test(LeftShift(2, 1)) as Expect(4);
  Test(LeftShift(1, 2)) as Expect(4);
  Test(LeftShift(3, 2)) as Expect(12);
  Test(LeftShift(1, 30)) as Expect(0x4000_0000);
  Test(LeftShift(5, 2)) as Expect(20);

  // Negative numbers can be shifted too.
  // CHECK:STDERR: literal.carbon:[[@LINE+14]]:18: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-1, 0)) as Expect(-1);
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: literal.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-1, 0)) as Expect(-1);
  // CHECK:STDERR:                                    ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(LeftShift(-1, 0)) as Expect(-1);
  // CHECK:STDERR: literal.carbon:[[@LINE+14]]:18: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-1, 1)) as Expect(-2);
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: literal.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-1, 1)) as Expect(-2);
  // CHECK:STDERR:                                    ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(LeftShift(-1, 1)) as Expect(-2);
  // CHECK:STDERR: literal.carbon:[[@LINE+14]]:18: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-2, 1)) as Expect(-4);
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: literal.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-2, 1)) as Expect(-4);
  // CHECK:STDERR:                                    ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(LeftShift(-2, 1)) as Expect(-4);
  // CHECK:STDERR: literal.carbon:[[@LINE+14]]:18: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-3, 1)) as Expect(-6);
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: literal.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(LeftShift(-3, 1)) as Expect(-6);
  // CHECK:STDERR:                                    ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(LeftShift(-3, 1)) as Expect(-6);

  // Large numbers can be shifted losslessly.
  Test(LeftShift(0xFFFF_FFFF, 1)) as Expect(0x1_FFFF_FFFE);
  Test(LeftShift(0xFFFF_FFFE, 1)) as Expect(0x1_FFFF_FFFC);
  Test(LeftShift(0xABCD_EF01, 8)) as Expect(0xAB_CDEF_0100);
  Test(LeftShift(0x7FFF_FFFF_FFFF_FFFF, 1)) as Expect(0xFFFF_FFFF_FFFF_FFFE);
  Test(LeftShift(0xFFFF_FFFF_FFFF_FFFF, 1)) as Expect(0x1_FFFF_FFFF_FFFF_FFFE);
}

// --- fail_bad_shift.carbon

library "[[@TEST_NAME]]";

fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
fn LeftShiftLit(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.left_shift";

// Shift greater than size is disallowed for sized types.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+20]]:29: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let size_1: i32 = LeftShift(1, 31);
// CHECK:STDERR:                             ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-10]]:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+10]]:32: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let size_1: i32 = LeftShift(1, 31);
// CHECK:STDERR:                                ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-20]]:22: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:                      ^~~~~~
// CHECK:STDERR:
let size_1: i32 = LeftShift(1, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+20]]:29: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let size_2: i32 = LeftShift(1, 32);
// CHECK:STDERR:                             ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-31]]:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+10]]:32: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let size_2: i32 = LeftShift(1, 32);
// CHECK:STDERR:                                ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-41]]:22: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:                      ^~~~~~
// CHECK:STDERR:
let size_2: i32 = LeftShift(1, 32);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+20]]:29: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let size_3: i32 = LeftShift(1, 33);
// CHECK:STDERR:                             ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-52]]:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+10]]:32: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let size_3: i32 = LeftShift(1, 33);
// CHECK:STDERR:                                ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-62]]:22: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:                      ^~~~~~
// CHECK:STDERR:
let size_3: i32 = LeftShift(1, 33);

// Overflow is allowed if the shift distance is in bounds.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+20]]:33: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let overflow_1: i32 = LeftShift(1000, 31);
// CHECK:STDERR:                                 ^~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-75]]:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+10]]:39: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let overflow_1: i32 = LeftShift(1000, 31);
// CHECK:STDERR:                                       ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-85]]:22: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:                      ^~~~~~
// CHECK:STDERR:
let overflow_1: i32 = LeftShift(1000, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+20]]:33: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let overflow_2: i32 = LeftShift(1000, 32);
// CHECK:STDERR:                                 ^~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-96]]:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+10]]:39: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let overflow_2: i32 = LeftShift(1000, 32);
// CHECK:STDERR:                                       ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-106]]:22: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:                      ^~~~~~
// CHECK:STDERR:
let overflow_2: i32 = LeftShift(1000, 32);

// Oversize shifts aren't allowed even if there's no overflow.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+20]]:36: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let no_overflow_1: i32 = LeftShift(0, 31);
// CHECK:STDERR:                                    ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-119]]:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+10]]:39: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let no_overflow_1: i32 = LeftShift(0, 31);
// CHECK:STDERR:                                       ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-129]]:22: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:                      ^~~~~~
// CHECK:STDERR:
let no_overflow_1: i32 = LeftShift(0, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+20]]:36: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let no_overflow_2: i32 = LeftShift(0, 32);
// CHECK:STDERR:                                    ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-140]]:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+10]]:39: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let no_overflow_2: i32 = LeftShift(0, 32);
// CHECK:STDERR:                                       ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-150]]:22: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:                      ^~~~~~
// CHECK:STDERR:
let no_overflow_2: i32 = LeftShift(0, 32);

// Negative shifts aren't allowed either, even for literals, even if the lhs is zero.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+17]]:34: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let negative: i32 = LeftShift(1, -1);
// CHECK:STDERR:                                  ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Negate {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+10]]:31: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let negative: i32 = LeftShift(1, -1);
// CHECK:STDERR:                               ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-170]]:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
let negative: i32 = LeftShift(1, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+17]]:39: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let negative_zero: i32 = LeftShift(0, -1);
// CHECK:STDERR:                                       ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Negate {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+10]]:36: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let negative_zero: i32 = LeftShift(0, -1);
// CHECK:STDERR:                                    ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE-188]]:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
let negative_zero: i32 = LeftShift(0, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+7]]:55: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let negative_lit: Core.IntLiteral() = LeftShiftLit(1, -1);
// CHECK:STDERR:                                                       ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Negate {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative_lit: Core.IntLiteral() = LeftShiftLit(1, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+7]]:60: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let negative_lit_zero: Core.IntLiteral() = LeftShiftLit(0, -1);
// CHECK:STDERR:                                                            ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Negate {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative_lit_zero: Core.IntLiteral() = LeftShiftLit(0, -1);

// --- fail_literal_overflow.carbon

library "[[@TEST_NAME]]";

fn LeftShift(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.left_shift";

// CHECK:STDERR: fail_literal_overflow.carbon:[[@LINE+11]]:16: error: shift distance of 1000000000 would result in an integer whose width is greater than the maximum supported width of 8388608 [CompileTimeUnsizedShiftOutOfRange]
// CHECK:STDERR: let bad: i32 = LeftShift(1, 1_000_000_000);
// CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_literal_overflow.carbon:[[@LINE+7]]:16: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let bad: i32 = LeftShift(1, 1_000_000_000);
// CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let bad: i32 = LeftShift(1, 1_000_000_000);

// CHECK:STDERR: fail_literal_overflow.carbon:[[@LINE+14]]:35: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let bad_negative: i32 = LeftShift(-1, 1_000_000_000);
// CHECK:STDERR:                                   ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Negate {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_literal_overflow.carbon:[[@LINE+7]]:25: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let bad_negative: i32 = LeftShift(-1, 1_000_000_000);
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let bad_negative: i32 = LeftShift(-1, 1_000_000_000);

// --- fail_comp_time_only_shift.carbon

library "[[@TEST_NAME]]";

fn LeftShiftByLit(a: i32, b: Core.IntLiteral()) -> i32 = "int.left_shift";
fn LeftShiftOfLit(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.left_shift";

var a_lit: Core.IntLiteral() = 12;
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE+7]]:1: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: var an_i32: i32 = 34;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var an_i32: i32 = 34;

// This can't be valid: we don't have a compile-time or runtime integer value for `a_lit`.
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE+7]]:17: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let bad1: i32 = LeftShiftByLit(an_i32, a_lit);
// CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE-17]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn LeftShiftByLit(a: i32, b: Core.IntLiteral()) -> i32 = "int.left_shift";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let bad1: i32 = LeftShiftByLit(an_i32, a_lit);

// TODO: This could be valid because we don't actually need the return value at runtime.
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE+7]]:31: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let bad2: Core.IntLiteral() = LeftShiftOfLit(a_lit, an_i32);
// CHECK:STDERR:                               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE-26]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn LeftShiftOfLit(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.left_shift";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let bad2: Core.IntLiteral() = LeftShiftOfLit(a_lit, an_i32);

// TODO: This could be valid because the literal argument has a constant value.
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE+7]]:17: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let bad3: i32 = LeftShiftByLit(an_i32, 12);
// CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE-37]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn LeftShiftByLit(a: i32, b: Core.IntLiteral()) -> i32 = "int.left_shift";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let bad3: i32 = LeftShiftByLit(an_i32, 12);

// TODO: This could be valid because we don't actually need the return value at runtime.
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE+7]]:31: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let bad4: Core.IntLiteral() = LeftShiftOfLit(12, an_i32);
// CHECK:STDERR:                               ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE-46]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn LeftShiftOfLit(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.left_shift";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let bad4: Core.IntLiteral() = LeftShiftOfLit(12, an_i32);
