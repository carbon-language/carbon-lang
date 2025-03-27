// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/reorder_qualified.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/reorder_qualified.carbon
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

class A {
  class B {
    class C;

    fn BF();
    var b: i32;
  }

  class B.C {
    class D {
      fn F();

      fn DF();
      var d: i32;
    }

    fn D.DF() {
      // A, B, C, and D are complete here.
      // CHECK:STDERR: reorder_qualified.carbon:[[@LINE+7]]:18: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
      // CHECK:STDERR:       var a: A = {.a = 1};
      // CHECK:STDERR:                  ^~~~~~~~
      // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
      // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
      // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // CHECK:STDERR:
      var a: A = {.a = 1};
      // CHECK:STDERR: reorder_qualified.carbon:[[@LINE+7]]:18: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
      // CHECK:STDERR:       var b: B = {.b = 2};
      // CHECK:STDERR:                  ^~~~~~~~
      // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
      // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
      // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // CHECK:STDERR:
      var b: B = {.b = 2};
      // CHECK:STDERR: reorder_qualified.carbon:[[@LINE+7]]:18: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
      // CHECK:STDERR:       var c: C = {.c = 3};
      // CHECK:STDERR:                  ^~~~~~~~
      // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
      // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
      // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // CHECK:STDERR:
      var c: C = {.c = 3};
      // CHECK:STDERR: reorder_qualified.carbon:[[@LINE+7]]:18: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
      // CHECK:STDERR:       var d: D = {.d = 4};
      // CHECK:STDERR:                  ^~~~~~~~
      // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
      // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
      // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // CHECK:STDERR:
      var d: D = {.d = 4};

      // Unqualified lookup looks in all of them.
      AF();
      BF();
      CF();
      DF();
    }

    fn CF();
    var c: i32;
  }

  fn AF();
  var a: i32;
}

// CHECK:STDOUT: --- reorder_qualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %BF.type: type = fn_type @BF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BF: %BF.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.b.0a3: type = struct_type {.b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ba8: <witness> = complete_type_witness %struct_type.b.0a3 [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %DF.type: type = fn_type @DF [concrete]
// CHECK:STDOUT:   %DF: %DF.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.d.b7b: type = struct_type {.d: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.860: <witness> = complete_type_witness %struct_type.d.b7b [concrete]
// CHECK:STDOUT:   %CF.type: type = fn_type @CF [concrete]
// CHECK:STDOUT:   %CF: %CF.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.c.b66: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.836: <witness> = complete_type_witness %struct_type.c.b66 [concrete]
// CHECK:STDOUT:   %AF.type: type = fn_type @AF [concrete]
// CHECK:STDOUT:   %AF: %AF.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.ba9: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fd7: <witness> = complete_type_witness %struct_type.a.ba9 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.a.a6c: type = struct_type {.a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.b.a15: type = struct_type {.b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.c.5b8: type = struct_type {.c: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %struct_type.d.3ea: type = struct_type {.d: Core.IntLiteral} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %AF.decl: %AF.type = fn_decl @AF [concrete = constants.%AF] {} {}
// CHECK:STDOUT:   %.loc274_8: %A.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc274_3: %A.elem = var_pattern %.loc274_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %A.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.ba9 [concrete = constants.%complete_type.fd7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT:   .AF = %AF.decl
// CHECK:STDOUT:   .a = %.loc274_8
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %BF.decl: %BF.type = fn_decl @BF [concrete = constants.%BF] {} {}
// CHECK:STDOUT:   %.loc216_10: %B.elem = field_decl b, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc216_5: %B.elem = var_pattern %.loc216_10
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %B.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.b.0a3 [concrete = constants.%complete_type.ba8]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .C = %C.decl
// CHECK:STDOUT:   .BF = %BF.decl
// CHECK:STDOUT:   .b = %.loc216_10
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .AF = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %DF.decl: %DF.type = fn_decl @DF [concrete = constants.%DF] {} {}
// CHECK:STDOUT:   %CF.decl: %CF.type = fn_decl @CF [concrete = constants.%CF] {} {}
// CHECK:STDOUT:   %.loc270_10: %C.elem = field_decl c, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc270_5: %C.elem = var_pattern %.loc270_10
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %C.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.c.b66 [concrete = constants.%complete_type.836]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .D = %D.decl
// CHECK:STDOUT:   .CF = %CF.decl
// CHECK:STDOUT:   .c = %.loc270_10
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .AF = <poisoned>
// CHECK:STDOUT:   .BF = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %DF.decl: %DF.type = fn_decl @DF [concrete = constants.%DF] {} {}
// CHECK:STDOUT:   %.loc224_12: %D.elem = field_decl d, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc224_7: %D.elem = var_pattern %.loc224_12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %D.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.d.b7b [concrete = constants.%complete_type.860]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .DF = %DF.decl
// CHECK:STDOUT:   .d = %.loc224_12
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .AF = <poisoned>
// CHECK:STDOUT:   .BF = <poisoned>
// CHECK:STDOUT:   .CF = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %.loc236_7: %A = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var a
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc236_25.1: %struct_type.a.a6c = struct_literal (%int_1)
// CHECK:STDOUT:   %.loc236_25.2: %i32 = converted %int_1, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %B = binding_pattern b
// CHECK:STDOUT:     %.loc244_7: %B = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var b
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc244_25.1: %struct_type.b.a15 = struct_literal (%int_2)
// CHECK:STDOUT:   %.loc244_25.2: %i32 = converted %int_2, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %B.ref: type = name_ref B, @A.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b: ref %B = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc252_7: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc252_25.1: %struct_type.c.5b8 = struct_literal (%int_3)
// CHECK:STDOUT:   %.loc252_25.2: %i32 = converted %int_3, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %c.var, <error>
// CHECK:STDOUT:   %C.ref: type = name_ref C, @B.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %D = binding_pattern d
// CHECK:STDOUT:     %.loc260_7: %D = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %D = var d
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc260_25.1: %struct_type.d.3ea = struct_literal (%int_4)
// CHECK:STDOUT:   %.loc260_25.2: %i32 = converted %int_4, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %d.var, <error>
// CHECK:STDOUT:   %D.ref: type = name_ref D, @C.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %d: ref %D = bind_name d, %d.var
// CHECK:STDOUT:   %AF.ref: %AF.type = name_ref AF, @A.%AF.decl [concrete = constants.%AF]
// CHECK:STDOUT:   %AF.call: init %empty_tuple.type = call %AF.ref()
// CHECK:STDOUT:   %BF.ref: %BF.type = name_ref BF, @B.%BF.decl [concrete = constants.%BF]
// CHECK:STDOUT:   %BF.call: init %empty_tuple.type = call %BF.ref()
// CHECK:STDOUT:   %CF.ref: %CF.type = name_ref CF, @C.%CF.decl [concrete = constants.%CF]
// CHECK:STDOUT:   %CF.call: init %empty_tuple.type = call %CF.ref()
// CHECK:STDOUT:   %DF.ref: %DF.type = name_ref DF, @D.%DF.decl [concrete = constants.%DF]
// CHECK:STDOUT:   %DF.call: init %empty_tuple.type = call %DF.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AF();
// CHECK:STDOUT:
