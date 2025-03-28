// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/equal_rewrite.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/equal_rewrite.carbon
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/io.carbon:18:26: error: member access into undefined facet type `Negate` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn EOF() -> i32 { return -1; }
// CHECK:STDERR:                          ^~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/io.carbon:9:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Negate {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:115:13: error: member access into undefined facet type `Add` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self + other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:10:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Add {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:122:13: error: member access into undefined facet type `BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self & other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:15:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:129:13: error: member access into undefined facet type `BitOr` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self | other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitOr {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:136:13: error: member access into undefined facet type `BitXor` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self ^ other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:35:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitXor {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:143:13: error: member access into undefined facet type `Div` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self / other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:55:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Div {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:150:13: error: member access into undefined facet type `LeftShift` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self << other;
// CHECK:STDERR:             ^~~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:45:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface LeftShift {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:157:13: error: member access into undefined facet type `Mod` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self % other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:65:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Mod {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:164:13: error: member access into undefined facet type `Mul` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self * other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:45:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Mul {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:171:13: error: member access into undefined facet type `RightShift` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self >> other;
// CHECK:STDERR:             ^~~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:55:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface RightShift {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:178:13: error: member access into undefined facet type `Sub` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self - other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:30:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Sub {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:187:5: error: member access into undefined facet type `SubAssign` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self -= 1;
// CHECK:STDERR:     ^~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:35:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface SubAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:194:5: error: member access into undefined facet type `AddAssign` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self += 1;
// CHECK:STDERR:     ^~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:15:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface AddAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:126:13: error: member access into undefined facet type `Add` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self + other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:10:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Add {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:133:13: error: member access into undefined facet type `BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self & other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:15:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:140:13: error: member access into undefined facet type `BitOr` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self | other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitOr {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:147:13: error: member access into undefined facet type `BitXor` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self ^ other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:35:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitXor {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:154:13: error: member access into undefined facet type `Div` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self / other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:55:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Div {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:161:13: error: member access into undefined facet type `LeftShift` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self << other;
// CHECK:STDERR:             ^~~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:45:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface LeftShift {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:168:13: error: member access into undefined facet type `Mod` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self % other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:65:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Mod {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:175:13: error: member access into undefined facet type `Mul` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self * other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:45:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Mul {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:182:13: error: member access into undefined facet type `RightShift` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self >> other;
// CHECK:STDERR:             ^~~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon:55:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface RightShift {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:189:13: error: member access into undefined facet type `Sub` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self = *self - other;
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:30:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Sub {
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:198:5: error: member access into undefined facet type `SubAssign` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self -= 1;
// CHECK:STDERR:     ^~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:35:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface SubAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:205:5: error: member access into undefined facet type `AddAssign` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR:     *self += 1;
// CHECK:STDERR:     ^~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:15:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface AddAssign {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:

// --- equal_constraint.carbon

library "[[@TEST_NAME]]";

interface N {
  let P:! type;
}

// CHECK:STDERR: equal_constraint.carbon:[[@LINE+7]]:22: error: member access into undefined facet type `N` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn Equal(T:! N where .P = {});
// CHECK:STDERR:                      ^~
// CHECK:STDERR: equal_constraint.carbon:[[@LINE-7]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface N {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn Equal(T:! N where .P = {});

// --- nested_rewrites.carbon

library "[[@TEST_NAME]]";

interface A {
  let B:! type;
  let C:! type;
}

// CHECK:STDERR: nested_rewrites.carbon:[[@LINE+7]]:31: error: member access into undefined facet type `A` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn NestedRewrite(D:! (A where .B = bool) where .C = ());
// CHECK:STDERR:                               ^~
// CHECK:STDERR: nested_rewrites.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface A {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn NestedRewrite(D:! (A where .B = bool) where .C = ());

// --- repeated_rewrite.carbon

library "[[@TEST_NAME]]";

interface E {
  let F:! type;
}

// CHECK:STDERR: repeated_rewrite.carbon:[[@LINE+7]]:27: error: member access into undefined facet type `E` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn OneRewrite(G:! E where .F = i32) {}
// CHECK:STDERR:                           ^~
// CHECK:STDERR: repeated_rewrite.carbon:[[@LINE-7]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface E {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn OneRewrite(G:! E where .F = i32) {}

// CHECK:STDERR: repeated_rewrite.carbon:[[@LINE+14]]:32: error: member access into undefined facet type `E` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn RepeatedRewrite(H:! E where .F = i32 and .F = i32) {
// CHECK:STDERR:                                ^~
// CHECK:STDERR: repeated_rewrite.carbon:[[@LINE-16]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface E {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: repeated_rewrite.carbon:[[@LINE+7]]:45: error: member access into undefined facet type `E` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn RepeatedRewrite(H:! E where .F = i32 and .F = i32) {
// CHECK:STDERR:                                             ^~
// CHECK:STDERR: repeated_rewrite.carbon:[[@LINE-23]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface E {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn RepeatedRewrite(H:! E where .F = i32 and .F = i32) {
  OneRewrite(H);
}

// CHECK:STDERR: repeated_rewrite.carbon:[[@LINE+7]]:32: error: member access into undefined facet type `E` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn OneRewriteAgain(I:! E where .F = i32) {
// CHECK:STDERR:                                ^~
// CHECK:STDERR: repeated_rewrite.carbon:[[@LINE-34]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface E {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn OneRewriteAgain(I:! E where .F = i32) {
  RepeatedRewrite(I);
}

// --- rewrites_reordered.carbon

library "[[@TEST_NAME]]";

interface J {
  let K:! type;
  let L:! type;
}

// CHECK:STDERR: rewrites_reordered.carbon:[[@LINE+14]]:29: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn Alphabetical(M:! J where .K = () and .L = bool) {}
// CHECK:STDERR:                             ^~
// CHECK:STDERR: rewrites_reordered.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: rewrites_reordered.carbon:[[@LINE+7]]:41: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn Alphabetical(M:! J where .K = () and .L = bool) {}
// CHECK:STDERR:                                         ^~
// CHECK:STDERR: rewrites_reordered.carbon:[[@LINE-15]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn Alphabetical(M:! J where .K = () and .L = bool) {}

// CHECK:STDERR: rewrites_reordered.carbon:[[@LINE+14]]:25: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn Reversed(N:! J where .L = bool and .K = ()) {
// CHECK:STDERR:                         ^~
// CHECK:STDERR: rewrites_reordered.carbon:[[@LINE-24]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: rewrites_reordered.carbon:[[@LINE+7]]:39: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn Reversed(N:! J where .L = bool and .K = ()) {
// CHECK:STDERR:                                       ^~
// CHECK:STDERR: rewrites_reordered.carbon:[[@LINE-31]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn Reversed(N:! J where .L = bool and .K = ()) {
  Alphabetical(N);
}

// --- todo_fail_rewrites_mismatch_right.carbon

library "[[@TEST_NAME]]";

interface O {
  let P:! type;
}

// CHECK:STDERR: todo_fail_rewrites_mismatch_right.carbon:[[@LINE+7]]:28: error: member access into undefined facet type `O` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn WithInteger(Q:! O where .P = i32) {}
// CHECK:STDERR:                            ^~
// CHECK:STDERR: todo_fail_rewrites_mismatch_right.carbon:[[@LINE-7]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface O {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn WithInteger(Q:! O where .P = i32) {}

// CHECK:STDERR: todo_fail_rewrites_mismatch_right.carbon:[[@LINE+7]]:25: error: member access into undefined facet type `O` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn WithBool(R:! O where .P = bool) {
// CHECK:STDERR:                         ^~
// CHECK:STDERR: todo_fail_rewrites_mismatch_right.carbon:[[@LINE-16]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface O {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn WithBool(R:! O where .P = bool) {
  // TODO: This should fail.
  WithInteger(R);
}

// --- todo_fail_rewrites_mismatch_left.carbon

library "[[@TEST_NAME]]";

interface S {
  let T:! type;
  let U:! type;
}

// CHECK:STDERR: todo_fail_rewrites_mismatch_left.carbon:[[@LINE+7]]:22: error: member access into undefined facet type `S` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn WithT(V:! S where .T = ()) {}
// CHECK:STDERR:                      ^~
// CHECK:STDERR: todo_fail_rewrites_mismatch_left.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface S {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn WithT(V:! S where .T = ()) {}

// CHECK:STDERR: todo_fail_rewrites_mismatch_left.carbon:[[@LINE+7]]:22: error: member access into undefined facet type `S` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn WithU(W:! S where .U = ()) {
// CHECK:STDERR:                      ^~
// CHECK:STDERR: todo_fail_rewrites_mismatch_left.carbon:[[@LINE-17]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface S {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn WithU(W:! S where .U = ()) {
  // TODO: This should fail.
  WithT(W);
}

// --- fail_import_rewrites.carbon

library "[[@TEST_NAME]]";

import library "equal_constraint";
import library "nested_rewrites";

fn Calls() {
  Equal(bool);

  NestedRewrite(i32);
}

// --- fail_check_rewrite_constraints.carbon

library "[[@TEST_NAME]]";

interface I {
  let Member:! type;
}

// `2` can't be converted to the type of `I.Member`
// CHECK:STDERR: fail_check_rewrite_constraints.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn RewriteTypeMismatch(X:! I where .Member = 2);
// CHECK:STDERR:                                    ^~~~~~~
// CHECK:STDERR: fail_check_rewrite_constraints.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn RewriteTypeMismatch(X:! I where .Member = 2);

// --- fail_todo_let.carbon

library "[[@TEST_NAME]]";

interface A {}
class D {}
impl D as A {}
// TODO: This should be a compile-time binding, once that is supported.
// CHECK:STDERR: fail_todo_let.carbon:[[@LINE+7]]:35: error: member access into undefined facet type `Core.ImplicitAs(type where...)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let B: type where .Self impls A = D;
// CHECK:STDERR:                                   ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let B: type where .Self impls A = D;

// --- fail_type_does_not_implement_where.carbon

library "[[@TEST_NAME]]";

interface E {
  let F:! type;
  let G:! type;
}
// Testing how these types get stringified in error messages.

// TODO: This should be a compile-time binding, once that is supported.
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+14]]:17: error: member access into undefined facet type `E` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let H: (E where .F = bool and .G = ()) = f64;
// CHECK:STDERR:                 ^~
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE-10]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface E {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+7]]:31: error: member access into undefined facet type `E` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let H: (E where .F = bool and .G = ()) = f64;
// CHECK:STDERR:                               ^~
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE-17]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface E {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
let H: (E where .F = bool and .G = ()) = f64;

// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+7]]:18: error: member access into undefined facet type `E` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let J: ((E where .F = {}) where .G = i32) = bool;
// CHECK:STDERR:                  ^~
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE-26]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface E {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
let J: ((E where .F = {}) where .G = i32) = bool;

// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+14]]:17: error: member access into undefined facet type `E` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let K: (E where .F = .Self.G) = bool;
// CHECK:STDERR:                 ^~
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE-35]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface E {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+7]]:22: error: member access into undefined facet type `E` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let K: (E where .F = .Self.G) = bool;
// CHECK:STDERR:                      ^~~~~~~
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE-42]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface E {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
let K: (E where .F = .Self.G) = bool;

// CHECK:STDOUT: --- equal_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %N.type: type = facet_type <@N> [concrete]
// CHECK:STDOUT:   %Self: %N.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %N.assoc_type: type = assoc_entity_type %N.type [concrete]
// CHECK:STDOUT:   %assoc0: %N.assoc_type = assoc_entity element0, @N.%P [concrete]
// CHECK:STDOUT:   %.Self: %N.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %T.patt: <error> = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:     .Equal = %Equal.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %N.decl: type = interface_decl @N [concrete = constants.%N.type] {} {}
// CHECK:STDOUT:   %Equal.decl: %Equal.type = fn_decl @Equal [concrete = constants.%Equal] {
// CHECK:STDOUT:     %T.patt.loc15_10.1: <error> = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_16.1: type = splice_block %.loc15_16.2 [concrete = <error>] {
// CHECK:STDOUT:       %N.ref: type = name_ref N, file.%N.decl [concrete = constants.%N.type]
// CHECK:STDOUT:       %.Self: %N.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %N.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %P.ref: <error> = name_ref P, <error> [concrete = <error>]
// CHECK:STDOUT:       %.loc15_28: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc15_16.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %P.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T: <error> = bind_symbolic_name T, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @N {
// CHECK:STDOUT:   %Self: %N.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %P: type = assoc_const_decl @P [concrete] {
// CHECK:STDOUT:     %assoc0: %N.assoc_type = assoc_entity element0, @N.%P [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .P = @P.%assoc0
// CHECK:STDOUT:   witness = (%P)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @P(@N.%Self: %N.type) {
// CHECK:STDOUT:   assoc_const P:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal(%T: <error>) {
// CHECK:STDOUT:   %T.patt.loc15_10.2: <error> = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_10.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc15_10.1: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal(<error>) {
// CHECK:STDOUT:   %T.patt.loc15_10.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type %A.type [concrete]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.%B [concrete]
// CHECK:STDOUT:   %assoc1: %A.assoc_type = assoc_entity element1, @A.%C [concrete]
// CHECK:STDOUT:   %.Self: %A.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.patt: <error> = symbolic_binding_pattern D, 0 [symbolic]
// CHECK:STDOUT:   %NestedRewrite.type: type = fn_type @NestedRewrite [concrete]
// CHECK:STDOUT:   %NestedRewrite: %NestedRewrite.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .NestedRewrite = %NestedRewrite.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %NestedRewrite.decl: %NestedRewrite.type = fn_decl @NestedRewrite [concrete = constants.%NestedRewrite] {
// CHECK:STDOUT:     %D.patt.loc16_18.1: <error> = symbolic_binding_pattern D, 0 [symbolic = %D.patt.loc16_18.2 (constants.%D.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_42.1: type = splice_block %.loc16_42.2 [concrete = <error>] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %.Self.1: %A.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref.loc16_31: %A.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %B.ref: <error> = name_ref B, <error> [concrete = <error>]
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc16_25: type = where_expr %.Self.1 [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %B.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %.Self.2: <error> = bind_symbolic_name .Self [concrete = <error>]
// CHECK:STDOUT:       %.Self.ref.loc16_48: <error> = name_ref .Self, %.Self.2 [concrete = <error>]
// CHECK:STDOUT:       %C.ref: <error> = name_ref C, <error> [concrete = <error>]
// CHECK:STDOUT:       %.loc16_54: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc16_42.2: type = where_expr %.Self.2 [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %C.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %D: <error> = bind_symbolic_name D, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %B: type = assoc_const_decl @B [concrete] {
// CHECK:STDOUT:     %assoc0: %A.assoc_type = assoc_entity element0, @A.%B [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: type = assoc_const_decl @C [concrete] {
// CHECK:STDOUT:     %assoc1: %A.assoc_type = assoc_entity element1, @A.%C [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .B = @B.%assoc0
// CHECK:STDOUT:   .C = @C.%assoc1
// CHECK:STDOUT:   witness = (%B, %C)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @B(@A.%Self: %A.type) {
// CHECK:STDOUT:   assoc_const B:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @C(@A.%Self: %A.type) {
// CHECK:STDOUT:   assoc_const C:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NestedRewrite(%D: <error>) {
// CHECK:STDOUT:   %D.patt.loc16_18.2: <error> = symbolic_binding_pattern D, 0 [symbolic = %D.patt.loc16_18.2 (constants.%D.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%D.patt.loc16_18.1: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NestedRewrite(<error>) {
// CHECK:STDOUT:   %D.patt.loc16_18.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- repeated_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %E.type: type = facet_type <@E> [concrete]
// CHECK:STDOUT:   %Self: %E.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %E.assoc_type: type = assoc_entity_type %E.type [concrete]
// CHECK:STDOUT:   %assoc0: %E.assoc_type = assoc_entity element0, @E.%F [concrete]
// CHECK:STDOUT:   %.Self: %E.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %G.patt: <error> = symbolic_binding_pattern G, 0 [symbolic]
// CHECK:STDOUT:   %OneRewrite.type: type = fn_type @OneRewrite [concrete]
// CHECK:STDOUT:   %OneRewrite: %OneRewrite.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.patt: <error> = symbolic_binding_pattern H, 0 [symbolic]
// CHECK:STDOUT:   %RepeatedRewrite.type: type = fn_type @RepeatedRewrite [concrete]
// CHECK:STDOUT:   %RepeatedRewrite: %RepeatedRewrite.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.patt: <error> = symbolic_binding_pattern I, 0 [symbolic]
// CHECK:STDOUT:   %OneRewriteAgain.type: type = fn_type @OneRewriteAgain [concrete]
// CHECK:STDOUT:   %OneRewriteAgain: %OneRewriteAgain.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .OneRewrite = %OneRewrite.decl
// CHECK:STDOUT:     .RepeatedRewrite = %RepeatedRewrite.decl
// CHECK:STDOUT:     .OneRewriteAgain = %OneRewriteAgain.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %E.decl: type = interface_decl @E [concrete = constants.%E.type] {} {}
// CHECK:STDOUT:   %OneRewrite.decl: %OneRewrite.type = fn_decl @OneRewrite [concrete = constants.%OneRewrite] {
// CHECK:STDOUT:     %G.patt.loc15_15.1: <error> = symbolic_binding_pattern G, 0 [symbolic = %G.patt.loc15_15.2 (constants.%G.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_21.1: type = splice_block %.loc15_21.2 [concrete = <error>] {
// CHECK:STDOUT:       %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:       %.Self: %E.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %E.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc15_21.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %F.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G: <error> = bind_symbolic_name G, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RepeatedRewrite.decl: %RepeatedRewrite.type = fn_decl @RepeatedRewrite [concrete = constants.%RepeatedRewrite] {
// CHECK:STDOUT:     %H.patt.loc31_20.1: <error> = symbolic_binding_pattern H, 0 [symbolic = %H.patt.loc31_20.2 (constants.%H.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc31_26.1: type = splice_block %.loc31_26.2 [concrete = <error>] {
// CHECK:STDOUT:       %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:       %.Self: %E.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref.loc31_32: %E.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %F.ref.loc31_32: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:       %int_32.loc31_37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc31_37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.Self.ref.loc31_45: %E.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %F.ref.loc31_45: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:       %int_32.loc31_50: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc31_50: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc31_26.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %F.ref.loc31_32, <error>
// CHECK:STDOUT:         requirement_rewrite %F.ref.loc31_45, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %H: <error> = bind_symbolic_name H, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OneRewriteAgain.decl: %OneRewriteAgain.type = fn_decl @OneRewriteAgain [concrete = constants.%OneRewriteAgain] {
// CHECK:STDOUT:     %I.patt.loc42_20.1: <error> = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc42_20.2 (constants.%I.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc42_26.1: type = splice_block %.loc42_26.2 [concrete = <error>] {
// CHECK:STDOUT:       %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:       %.Self: %E.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %E.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc42_26.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %F.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I: <error> = bind_symbolic_name I, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @E {
// CHECK:STDOUT:   %Self: %E.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F: type = assoc_const_decl @F [concrete] {
// CHECK:STDOUT:     %assoc0: %E.assoc_type = assoc_entity element0, @E.%F [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @F.%assoc0
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @F(@E.%Self: %E.type) {
// CHECK:STDOUT:   assoc_const F:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OneRewrite(%G: <error>) {
// CHECK:STDOUT:   %G.patt.loc15_15.2: <error> = symbolic_binding_pattern G, 0 [symbolic = %G.patt.loc15_15.2 (constants.%G.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%G.patt.loc15_15.1: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RepeatedRewrite(%H: <error>) {
// CHECK:STDOUT:   %H.patt.loc31_20.2: <error> = symbolic_binding_pattern H, 0 [symbolic = %H.patt.loc31_20.2 (constants.%H.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%H.patt.loc31_20.1: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %OneRewrite.ref: %OneRewrite.type = name_ref OneRewrite, file.%OneRewrite.decl [concrete = constants.%OneRewrite]
// CHECK:STDOUT:     %H.ref: <error> = name_ref H, %H [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OneRewriteAgain(%I: <error>) {
// CHECK:STDOUT:   %I.patt.loc42_20.2: <error> = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc42_20.2 (constants.%I.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%I.patt.loc42_20.1: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %RepeatedRewrite.ref: %RepeatedRewrite.type = name_ref RepeatedRewrite, file.%RepeatedRewrite.decl [concrete = constants.%RepeatedRewrite]
// CHECK:STDOUT:     %I.ref: <error> = name_ref I, %I [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OneRewrite(<error>) {
// CHECK:STDOUT:   %G.patt.loc15_15.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RepeatedRewrite(<error>) {
// CHECK:STDOUT:   %H.patt.loc31_20.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OneRewriteAgain(<error>) {
// CHECK:STDOUT:   %I.patt.loc42_20.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- rewrites_reordered.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%K [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%L [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %M.patt: <error> = symbolic_binding_pattern M, 0 [symbolic]
// CHECK:STDOUT:   %Alphabetical.type: type = fn_type @Alphabetical [concrete]
// CHECK:STDOUT:   %Alphabetical: %Alphabetical.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.patt: <error> = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Reversed.type: type = fn_type @Reversed [concrete]
// CHECK:STDOUT:   %Reversed: %Reversed.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .Alphabetical = %Alphabetical.decl
// CHECK:STDOUT:     .Reversed = %Reversed.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %Alphabetical.decl: %Alphabetical.type = fn_decl @Alphabetical [concrete = constants.%Alphabetical] {
// CHECK:STDOUT:     %M.patt.loc23_17.1: <error> = symbolic_binding_pattern M, 0 [symbolic = %M.patt.loc23_17.2 (constants.%M.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc23_23.1: type = splice_block %.loc23_23.2 [concrete = <error>] {
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref.loc23_29: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K.ref: <error> = name_ref K, <error> [concrete = <error>]
// CHECK:STDOUT:       %.loc23_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.Self.ref.loc23_41: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L.ref: <error> = name_ref L, <error> [concrete = <error>]
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc23_23.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %K.ref, <error>
// CHECK:STDOUT:         requirement_rewrite %L.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %M: <error> = bind_symbolic_name M, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Reversed.decl: %Reversed.type = fn_decl @Reversed [concrete = constants.%Reversed] {
// CHECK:STDOUT:     %N.patt.loc39_13.1: <error> = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc39_13.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc39_19.1: type = splice_block %.loc39_19.2 [concrete = <error>] {
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref.loc39_25: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L.ref: <error> = name_ref L, <error> [concrete = <error>]
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.Self.ref.loc39_39: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K.ref: <error> = name_ref K, <error> [concrete = <error>]
// CHECK:STDOUT:       %.loc39_45: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc39_19.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %L.ref, <error>
// CHECK:STDOUT:         requirement_rewrite %K.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: <error> = bind_symbolic_name N, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %K: type = assoc_const_decl @K [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%K [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L: type = assoc_const_decl @L [concrete] {
// CHECK:STDOUT:     %assoc1: %J.assoc_type = assoc_entity element1, @J.%L [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .K = @K.%assoc0
// CHECK:STDOUT:   .L = @L.%assoc1
// CHECK:STDOUT:   witness = (%K, %L)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @K(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const K:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @L(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const L:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Alphabetical(%M: <error>) {
// CHECK:STDOUT:   %M.patt.loc23_17.2: <error> = symbolic_binding_pattern M, 0 [symbolic = %M.patt.loc23_17.2 (constants.%M.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%M.patt.loc23_17.1: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Reversed(%N: <error>) {
// CHECK:STDOUT:   %N.patt.loc39_13.2: <error> = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc39_13.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.patt.loc39_13.1: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Alphabetical.ref: %Alphabetical.type = name_ref Alphabetical, file.%Alphabetical.decl [concrete = constants.%Alphabetical]
// CHECK:STDOUT:     %N.ref: <error> = name_ref N, %N [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Alphabetical(<error>) {
// CHECK:STDOUT:   %M.patt.loc23_17.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Reversed(<error>) {
// CHECK:STDOUT:   %N.patt.loc39_13.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_rewrites_mismatch_right.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %O.type: type = facet_type <@O> [concrete]
// CHECK:STDOUT:   %Self: %O.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %O.assoc_type: type = assoc_entity_type %O.type [concrete]
// CHECK:STDOUT:   %assoc0: %O.assoc_type = assoc_entity element0, @O.%P [concrete]
// CHECK:STDOUT:   %.Self: %O.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Q.patt: <error> = symbolic_binding_pattern Q, 0 [symbolic]
// CHECK:STDOUT:   %WithInteger.type: type = fn_type @WithInteger [concrete]
// CHECK:STDOUT:   %WithInteger: %WithInteger.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %R.patt: <error> = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %WithBool.type: type = fn_type @WithBool [concrete]
// CHECK:STDOUT:   %WithBool: %WithBool.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .O = %O.decl
// CHECK:STDOUT:     .WithInteger = %WithInteger.decl
// CHECK:STDOUT:     .WithBool = %WithBool.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %O.decl: type = interface_decl @O [concrete = constants.%O.type] {} {}
// CHECK:STDOUT:   %WithInteger.decl: %WithInteger.type = fn_decl @WithInteger [concrete = constants.%WithInteger] {
// CHECK:STDOUT:     %Q.patt.loc15_16.1: <error> = symbolic_binding_pattern Q, 0 [symbolic = %Q.patt.loc15_16.2 (constants.%Q.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_22.1: type = splice_block %.loc15_22.2 [concrete = <error>] {
// CHECK:STDOUT:       %O.ref: type = name_ref O, file.%O.decl [concrete = constants.%O.type]
// CHECK:STDOUT:       %.Self: %O.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %O.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %P.ref: <error> = name_ref P, <error> [concrete = <error>]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc15_22.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %P.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Q: <error> = bind_symbolic_name Q, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithBool.decl: %WithBool.type = fn_decl @WithBool [concrete = constants.%WithBool] {
// CHECK:STDOUT:     %R.patt.loc24_13.1: <error> = symbolic_binding_pattern R, 0 [symbolic = %R.patt.loc24_13.2 (constants.%R.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc24_19.1: type = splice_block %.loc24_19.2 [concrete = <error>] {
// CHECK:STDOUT:       %O.ref: type = name_ref O, file.%O.decl [concrete = constants.%O.type]
// CHECK:STDOUT:       %.Self: %O.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %O.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %P.ref: <error> = name_ref P, <error> [concrete = <error>]
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc24_19.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %P.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %R: <error> = bind_symbolic_name R, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @O {
// CHECK:STDOUT:   %Self: %O.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %P: type = assoc_const_decl @P [concrete] {
// CHECK:STDOUT:     %assoc0: %O.assoc_type = assoc_entity element0, @O.%P [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .P = @P.%assoc0
// CHECK:STDOUT:   witness = (%P)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @P(@O.%Self: %O.type) {
// CHECK:STDOUT:   assoc_const P:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithInteger(%Q: <error>) {
// CHECK:STDOUT:   %Q.patt.loc15_16.2: <error> = symbolic_binding_pattern Q, 0 [symbolic = %Q.patt.loc15_16.2 (constants.%Q.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%Q.patt.loc15_16.1: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithBool(%R: <error>) {
// CHECK:STDOUT:   %R.patt.loc24_13.2: <error> = symbolic_binding_pattern R, 0 [symbolic = %R.patt.loc24_13.2 (constants.%R.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%R.patt.loc24_13.1: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %WithInteger.ref: %WithInteger.type = name_ref WithInteger, file.%WithInteger.decl [concrete = constants.%WithInteger]
// CHECK:STDOUT:     %R.ref: <error> = name_ref R, %R [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithInteger(<error>) {
// CHECK:STDOUT:   %Q.patt.loc15_16.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithBool(<error>) {
// CHECK:STDOUT:   %R.patt.loc24_13.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_rewrites_mismatch_left.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S.type: type = facet_type <@S> [concrete]
// CHECK:STDOUT:   %Self: %S.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %S.assoc_type: type = assoc_entity_type %S.type [concrete]
// CHECK:STDOUT:   %assoc0: %S.assoc_type = assoc_entity element0, @S.%T [concrete]
// CHECK:STDOUT:   %assoc1: %S.assoc_type = assoc_entity element1, @S.%U [concrete]
// CHECK:STDOUT:   %.Self: %S.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %V.patt: <error> = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %WithT.type: type = fn_type @WithT [concrete]
// CHECK:STDOUT:   %WithT: %WithT.type = struct_value () [concrete]
// CHECK:STDOUT:   %W.patt: <error> = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %WithU.type: type = fn_type @WithU [concrete]
// CHECK:STDOUT:   %WithU: %WithU.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .WithT = %WithT.decl
// CHECK:STDOUT:     .WithU = %WithU.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %S.decl: type = interface_decl @S [concrete = constants.%S.type] {} {}
// CHECK:STDOUT:   %WithT.decl: %WithT.type = fn_decl @WithT [concrete = constants.%WithT] {
// CHECK:STDOUT:     %V.patt.loc16_10.1: <error> = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc16_10.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_16.1: type = splice_block %.loc16_16.2 [concrete = <error>] {
// CHECK:STDOUT:       %S.ref: type = name_ref S, file.%S.decl [concrete = constants.%S.type]
// CHECK:STDOUT:       %.Self: %S.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %S.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:       %.loc16_28: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc16_16.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %T.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V: <error> = bind_symbolic_name V, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithU.decl: %WithU.type = fn_decl @WithU [concrete = constants.%WithU] {
// CHECK:STDOUT:     %W.patt.loc25_10.1: <error> = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc25_10.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc25_16.1: type = splice_block %.loc25_16.2 [concrete = <error>] {
// CHECK:STDOUT:       %S.ref: type = name_ref S, file.%S.decl [concrete = constants.%S.type]
// CHECK:STDOUT:       %.Self: %S.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %S.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.ref: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:       %.loc25_28: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc25_16.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %U.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W: <error> = bind_symbolic_name W, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @S {
// CHECK:STDOUT:   %Self: %S.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %S.assoc_type = assoc_entity element0, @S.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc1: %S.assoc_type = assoc_entity element1, @S.%U [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   .U = @U.%assoc1
// CHECK:STDOUT:   witness = (%T, %U)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@S.%Self: %S.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@S.%Self: %S.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithT(%V: <error>) {
// CHECK:STDOUT:   %V.patt.loc16_10.2: <error> = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc16_10.2 (constants.%V.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.patt.loc16_10.1: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithU(%W: <error>) {
// CHECK:STDOUT:   %W.patt.loc25_10.2: <error> = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc25_10.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%W.patt.loc25_10.1: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %WithT.ref: %WithT.type = name_ref WithT, file.%WithT.decl [concrete = constants.%WithT]
// CHECK:STDOUT:     %W.ref: <error> = name_ref W, %W [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithT(<error>) {
// CHECK:STDOUT:   %V.patt.loc16_10.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithU(<error>) {
// CHECK:STDOUT:   %W.patt.loc25_10.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Calls.type: type = fn_type @Calls [concrete]
// CHECK:STDOUT:   %Calls: %Calls.type = struct_value () [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.patt: <error> = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %NestedRewrite.type: type = fn_type @NestedRewrite [concrete]
// CHECK:STDOUT:   %NestedRewrite: %NestedRewrite.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.patt: <error> = symbolic_binding_pattern D, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.N = import_ref Main//equal_constraint, N, unloaded
// CHECK:STDOUT:   %Main.Equal: %Equal.type = import_ref Main//equal_constraint, Equal, loaded [concrete = constants.%Equal]
// CHECK:STDOUT:   %Main.A = import_ref Main//nested_rewrites, A, unloaded
// CHECK:STDOUT:   %Main.NestedRewrite: %NestedRewrite.type = import_ref Main//nested_rewrites, NestedRewrite, loaded [concrete = constants.%NestedRewrite]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: <error> = import_ref Main//equal_constraint, loc15_10, loaded [concrete = <error>]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: <error> = import_ref Main//nested_rewrites, loc16_18, loaded [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N = imports.%Main.N
// CHECK:STDOUT:     .Equal = imports.%Main.Equal
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .NestedRewrite = imports.%Main.NestedRewrite
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Calls = %Calls.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Calls.decl: %Calls.type = fn_decl @Calls [concrete = constants.%Calls] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Calls() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Equal.ref: %Equal.type = name_ref Equal, imports.%Main.Equal [concrete = constants.%Equal]
// CHECK:STDOUT:   %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %NestedRewrite.ref: %NestedRewrite.type = name_ref NestedRewrite, imports.%Main.NestedRewrite [concrete = constants.%NestedRewrite]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal(imports.%Main.import_ref.5ab3ec.1: <error>) [from "equal_constraint.carbon"] {
// CHECK:STDOUT:   %T.patt.2: <error> = symbolic_binding_pattern T, 0 [symbolic = %T.patt.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.1: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NestedRewrite(imports.%Main.import_ref.5ab3ec.2: <error>) [from "nested_rewrites.carbon"] {
// CHECK:STDOUT:   %D.patt.2: <error> = symbolic_binding_pattern D, 0 [symbolic = %D.patt.2 (constants.%D.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%D.patt.1: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal(<error>) {
// CHECK:STDOUT:   %T.patt.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NestedRewrite(<error>) {
// CHECK:STDOUT:   %D.patt.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_check_rewrite_constraints.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Member [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %X.patt: <error> = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %RewriteTypeMismatch.type: type = fn_type @RewriteTypeMismatch [concrete]
// CHECK:STDOUT:   %RewriteTypeMismatch: %RewriteTypeMismatch.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .RewriteTypeMismatch = %RewriteTypeMismatch.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %RewriteTypeMismatch.decl: %RewriteTypeMismatch.type = fn_decl @RewriteTypeMismatch [concrete = constants.%RewriteTypeMismatch] {
// CHECK:STDOUT:     %X.patt.loc16_24.1: <error> = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc16_24.2 (constants.%X.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_30.1: type = splice_block %.loc16_30.2 [concrete = <error>] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Member.ref: <error> = name_ref Member, <error> [concrete = <error>]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:       %.loc16_30.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %Member.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X: <error> = bind_symbolic_name X, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Member: type = assoc_const_decl @Member [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%Member [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Member = @Member.%assoc0
// CHECK:STDOUT:   witness = (%Member)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Member(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const Member:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RewriteTypeMismatch(%X: <error>) {
// CHECK:STDOUT:   %X.patt.loc16_24.2: <error> = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc16_24.2 (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%X.patt.loc16_24.1: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Member(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RewriteTypeMismatch(<error>) {
// CHECK:STDOUT:   %X.patt.loc16_24.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.31d: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
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
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %B.patt: %type_where = binding_pattern B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_13.1: type = splice_block %.loc15_13.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %A.ref: type = name_ref A, %A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     %.loc15_13.2: type = where_expr %.Self [concrete = constants.%type_where] {
// CHECK:STDOUT:       requirement_impls %.Self.ref, %A.ref
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_35: %type_where = converted @__global_init.%D.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %B: %type_where = bind_name B, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.31d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %D.ref as %A.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_type_does_not_implement_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %E.type: type = facet_type <@E> [concrete]
// CHECK:STDOUT:   %Self: %E.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %E.assoc_type: type = assoc_entity_type %E.type [concrete]
// CHECK:STDOUT:   %assoc0: %E.assoc_type = assoc_entity element0, @E.%F [concrete]
// CHECK:STDOUT:   %assoc1: %E.assoc_type = assoc_entity element1, @E.%G [concrete]
// CHECK:STDOUT:   %.Self: %E.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .H = %H
// CHECK:STDOUT:     .J = %J
// CHECK:STDOUT:     .K = %K
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %E.decl: type = interface_decl @E [concrete = constants.%E.type] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %H.patt: <error> = binding_pattern H
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25_11.1: type = splice_block %.loc25_11.2 [concrete = <error>] {
// CHECK:STDOUT:     %E.ref.loc25: type = name_ref E, %E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:     %.Self.1: %E.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc25_17: %E.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %F.ref.loc25: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.Self.ref.loc25_31: %E.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %G.ref.loc25: <error> = name_ref G, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc25_37: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc25_11.2: type = where_expr %.Self.1 [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %F.ref.loc25, <error>
// CHECK:STDOUT:       requirement_rewrite %G.ref.loc25, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H: <error> = bind_name H, <error>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %J.patt: <error> = binding_pattern J
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc34_27.1: type = splice_block %.loc34_27.2 [concrete = <error>] {
// CHECK:STDOUT:     %E.ref.loc34: type = name_ref E, %E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:     %.Self.2: %E.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc34_18: %E.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %F.ref.loc34: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc34_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc34_12: type = where_expr %.Self.2 [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %F.ref.loc34, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.3: <error> = bind_symbolic_name .Self [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref.loc34_33: <error> = name_ref .Self, %.Self.3 [concrete = <error>]
// CHECK:STDOUT:     %G.ref.loc34: <error> = name_ref G, <error> [concrete = <error>]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc34_27.2: type = where_expr %.Self.3 [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %G.ref.loc34, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J: <error> = bind_name J, <error>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %K.patt: <error> = binding_pattern K
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc50_11.1: type = splice_block %.loc50_11.2 [concrete = <error>] {
// CHECK:STDOUT:     %E.ref.loc50: type = name_ref E, %E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:     %.Self.4: %E.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc50_17: %E.type = name_ref .Self, %.Self.4 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %F.ref.loc50: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref.loc50_22: %E.type = name_ref .Self, %.Self.4 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %G.ref.loc50: <error> = name_ref G, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc50_11.2: type = where_expr %.Self.4 [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %F.ref.loc50, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K: <error> = bind_name K, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @E {
// CHECK:STDOUT:   %Self: %E.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F: type = assoc_const_decl @F [concrete] {
// CHECK:STDOUT:     %assoc0: %E.assoc_type = assoc_entity element0, @E.%F [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: type = assoc_const_decl @G [concrete] {
// CHECK:STDOUT:     %assoc1: %E.assoc_type = assoc_entity element1, @E.%G [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @F.%assoc0
// CHECK:STDOUT:   .G = @G.%assoc1
// CHECK:STDOUT:   witness = (%F, %G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @F(@E.%Self: %E.type) {
// CHECK:STDOUT:   assoc_const F:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @G(@E.%Self: %E.type) {
// CHECK:STDOUT:   assoc_const G:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:   %bool.make_type.loc34: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %bool.make_type.loc50: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self) {}
// CHECK:STDOUT:
