// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/use_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/use_assoc_const.carbon
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

// --- associated_type_in_method_signature.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn F[self: Self](u: U) -> U;
}

// CHECK:STDERR: associated_type_in_method_signature.carbon:[[@LINE+7]]:20: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as J where .U = i32 {
// CHECK:STDERR:                    ^~
// CHECK:STDERR: associated_type_in_method_signature.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as J where .U = i32 {
  // CHECK:STDERR: associated_type_in_method_signature.carbon:[[@LINE+7]]:44: error: member access into undefined facet type `Core.Add` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   fn F[self: Self](u: i32) -> i32 { return u + 2; }
  // CHECK:STDERR:                                            ^~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:10:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Add {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F[self: Self](u: i32) -> i32 { return u + 2; }
}

fn CallMethod(x: ()) -> i32 {
  // CHECK:STDERR: associated_type_in_method_signature.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return x.(J.F)(40);
  // CHECK:STDERR:             ^~~
  // CHECK:STDERR: associated_type_in_method_signature.carbon:[[@LINE-27]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface J {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  return x.(J.F)(40);
}

class C {
  // This allows the type to be copyable so it can be returned.
  adapt {};
}

// CHECK:STDERR: associated_type_in_method_signature.carbon:[[@LINE+7]]:19: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C as J where .U = C {
// CHECK:STDERR:                   ^~
// CHECK:STDERR: associated_type_in_method_signature.carbon:[[@LINE-42]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C as J where .U = C {
  fn F[self: Self](u: C) -> C { return self; }
}

// --- associated_type_in_function_signature.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn F(u: U) -> U;
}

class D { }

// CHECK:STDERR: associated_type_in_function_signature.carbon:[[@LINE+7]]:19: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl D as J where .U = i32 {
// CHECK:STDERR:                   ^~
// CHECK:STDERR: associated_type_in_function_signature.carbon:[[@LINE-10]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl D as J where .U = i32 {
  // CHECK:STDERR: associated_type_in_function_signature.carbon:[[@LINE+7]]:32: error: member access into undefined facet type `Core.Add` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   fn F(u: i32) -> i32 { return u + 3; }
  // CHECK:STDERR:                                ^~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:10:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Add {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F(u: i32) -> i32 { return u + 3; }
}

fn CallFunction() -> i32 {
  // CHECK:STDERR: associated_type_in_function_signature.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return D.(J.F)(4);
  // CHECK:STDERR:             ^~~
  // CHECK:STDERR: associated_type_in_function_signature.carbon:[[@LINE-29]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface J {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  return D.(J.F)(4);
}

// --- extend_impl_with_associated_type_in_signature.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn F(u: U) -> U;
  fn G[self: Self](v: U) -> U;
}

class E {
  // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE+11]]:26: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   extend impl as J where .U = i32 {
  // CHECK:STDERR:                          ^~
  // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE-10]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface J {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE+4]]:3: error: semantics TODO: `extending non-facet-type constraint` [SemanticsTodo]
  // CHECK:STDERR:   extend impl as J where .U = i32 {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend impl as J where .U = i32 {
    fn F(u: i32) -> i32 {
      // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE+7]]:14: error: member access into undefined facet type `Core.Add` [QualifiedExprInUndefinedFacetTypeScope]
      // CHECK:STDERR:       return u + 1;
      // CHECK:STDERR:              ^~~~~
      // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:10:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
      // CHECK:STDERR: interface Add {
      // CHECK:STDERR: ^~~~~~~~~~~~~~~
      // CHECK:STDERR:
      return u + 1;
    }
    fn G[self: Self](v: i32) -> i32 {
      // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE+7]]:14: error: member access into undefined facet type `Core.Add` [QualifiedExprInUndefinedFacetTypeScope]
      // CHECK:STDERR:       return v + 2;
      // CHECK:STDERR:              ^~~~~
      // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:10:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
      // CHECK:STDERR: interface Add {
      // CHECK:STDERR: ^~~~~~~~~~~~~~~
      // CHECK:STDERR:
      return v + 2;
    }
  }
}

fn CallBoth(e: E) -> i32 {
  // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE+35]]:10: error: member access into undefined facet type `Core.Add` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return e.F(2) + e.G(3) + E.F(4) + e.(E.G)(5) + e.(J.G)(6);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:10:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Add {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE+28]]:10: error: member access into undefined facet type `Core.Add` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return e.F(2) + e.G(3) + E.F(4) + e.(E.G)(5) + e.(J.G)(6);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:10:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Add {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE+21]]:10: error: member access into undefined facet type `Core.Add` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return e.F(2) + e.G(3) + E.F(4) + e.(E.G)(5) + e.(J.G)(6);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:10:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Add {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE+14]]:53: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return e.F(2) + e.G(3) + E.F(4) + e.(E.G)(5) + e.(J.G)(6);
  // CHECK:STDERR:                                                     ^~~
  // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE-67]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface J {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE+7]]:10: error: member access into undefined facet type `Core.Add` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return e.F(2) + e.G(3) + E.F(4) + e.(E.G)(5) + e.(J.G)(6);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:10:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Add {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return e.F(2) + e.G(3) + E.F(4) + e.(E.G)(5) + e.(J.G)(6);
}

// CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE+14]]:33: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn GenericCallF[T:! J](t: T, u: T.U) -> T.U {
// CHECK:STDERR:                                 ^~~
// CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE-84]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE+7]]:41: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn GenericCallF[T:! J](t: T, u: T.U) -> T.U {
// CHECK:STDERR:                                         ^~~
// CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE-91]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn GenericCallF[T:! J](t: T, u: T.U) -> T.U {
  // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE+7]]:10: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return t.F(u);
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE-99]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface J {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  return t.F(u);
}

fn CallGeneric(e: E) -> i32 {
  // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE+10]]:10: error: member access into undefined facet type `Core.ImplicitAs(J)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return GenericCallF(e, 2);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: extend_impl_with_associated_type_in_signature.carbon:[[@LINE-18]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn GenericCallF[T:! J](t: T, u: T.U) -> T.U {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return GenericCallF(e, 2);
}

// --- use_constraint_on_associated_type.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! Core.Add;
  fn F(u: U) -> U;
}

// CHECK:STDERR: use_constraint_on_associated_type.carbon:[[@LINE+14]]:37: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn GenericAddResult[T:! J](t: T, u: T.U) -> T.U {
// CHECK:STDERR:                                     ^~~
// CHECK:STDERR: use_constraint_on_associated_type.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: use_constraint_on_associated_type.carbon:[[@LINE+7]]:45: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn GenericAddResult[T:! J](t: T, u: T.U) -> T.U {
// CHECK:STDERR:                                             ^~~
// CHECK:STDERR: use_constraint_on_associated_type.carbon:[[@LINE-15]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn GenericAddResult[T:! J](t: T, u: T.U) -> T.U {
  // CHECK:STDERR: use_constraint_on_associated_type.carbon:[[@LINE+21]]:10: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return t.F(u) + T.F(u);
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: use_constraint_on_associated_type.carbon:[[@LINE-23]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface J {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: use_constraint_on_associated_type.carbon:[[@LINE+14]]:19: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return t.F(u) + T.F(u);
  // CHECK:STDERR:                   ^~~
  // CHECK:STDERR: use_constraint_on_associated_type.carbon:[[@LINE-30]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface J {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: use_constraint_on_associated_type.carbon:[[@LINE+7]]:10: error: member access into undefined facet type `Core.Add` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return t.F(u) + T.F(u);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:10:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Add {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return t.F(u) + T.F(u);
}

// --- interface_qualified.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn G[self: Self](v: U) -> U;
}

// CHECK:STDERR: interface_qualified.carbon:[[@LINE+14]]:50: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn GenericCallInterfaceQualified[T:! J](t: T, u: T.U) -> T.U {
// CHECK:STDERR:                                                  ^~~
// CHECK:STDERR: interface_qualified.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: interface_qualified.carbon:[[@LINE+7]]:58: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn GenericCallInterfaceQualified[T:! J](t: T, u: T.U) -> T.U {
// CHECK:STDERR:                                                          ^~~
// CHECK:STDERR: interface_qualified.carbon:[[@LINE-15]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn GenericCallInterfaceQualified[T:! J](t: T, u: T.U) -> T.U {
  // CHECK:STDERR: interface_qualified.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return t.(J.G)(u);
  // CHECK:STDERR:             ^~~
  // CHECK:STDERR: interface_qualified.carbon:[[@LINE-23]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface J {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  return t.(J.G)(u);
}

// --- fail_todo_use_where.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn F(u: U) -> U;
}

// CHECK:STDERR: fail_todo_use_where.carbon:[[@LINE+7]]:32: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn GenericCallFI32[T:! J where .U = i32](t: T) -> i32 {
// CHECK:STDERR:                                ^~
// CHECK:STDERR: fail_todo_use_where.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn GenericCallFI32[T:! J where .U = i32](t: T) -> i32 {
  return t.F(2);
}

// --- fail_todo_use_associated_constant_in_impl.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn F(u: U) -> U;
}

class E {
  // CHECK:STDERR: fail_todo_use_associated_constant_in_impl.carbon:[[@LINE+11]]:26: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   extend impl as J where .U = i32 {
  // CHECK:STDERR:                          ^~
  // CHECK:STDERR: fail_todo_use_associated_constant_in_impl.carbon:[[@LINE-9]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface J {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_use_associated_constant_in_impl.carbon:[[@LINE+4]]:3: error: semantics TODO: `extending non-facet-type constraint` [SemanticsTodo]
  // CHECK:STDERR:   extend impl as J where .U = i32 {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend impl as J where .U = i32 {
    // CHECK:STDERR: fail_todo_use_associated_constant_in_impl.carbon:[[@LINE+8]]:13: error: name `U` not found [NameNotFound]
    // CHECK:STDERR:     fn F(u: U) -> U {
    // CHECK:STDERR:             ^
    // CHECK:STDERR:
    // CHECK:STDERR: fail_todo_use_associated_constant_in_impl.carbon:[[@LINE+4]]:19: error: name `U` not found [NameNotFound]
    // CHECK:STDERR:     fn F(u: U) -> U {
    // CHECK:STDERR:                   ^
    // CHECK:STDERR:
    fn F(u: U) -> U {
      // CHECK:STDERR: fail_todo_use_associated_constant_in_impl.carbon:[[@LINE+7]]:14: error: member access into undefined facet type `Core.Add` [QualifiedExprInUndefinedFacetTypeScope]
      // CHECK:STDERR:       return u + 1;
      // CHECK:STDERR:              ^~~~~
      // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:10:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
      // CHECK:STDERR: interface Add {
      // CHECK:STDERR: ^~~~~~~~~~~~~~~
      // CHECK:STDERR:
      return u + 1;
    }
  }
}

// --- fail_todo_self_period_associated_type.carbon
library "[[@TEST_NAME]]";

interface J2 {
  let U2:! type;
  // CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE+14]]:23: error: member access into object of incomplete type `J2` [IncompleteTypeInMemberAccess]
  // CHECK:STDERR:   fn F[self: Self](z: Self.U2) -> Self.U2;
  // CHECK:STDERR:                       ^~~~
  // CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE-5]]:1: note: interface is currently being defined [InterfaceIncompleteWithinDefinition]
  // CHECK:STDERR: interface J2 {
  // CHECK:STDERR: ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE+7]]:35: error: member access into object of incomplete type `J2` [IncompleteTypeInMemberAccess]
  // CHECK:STDERR:   fn F[self: Self](z: Self.U2) -> Self.U2;
  // CHECK:STDERR:                                   ^~~~
  // CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE-12]]:1: note: interface is currently being defined [InterfaceIncompleteWithinDefinition]
  // CHECK:STDERR: interface J2 {
  // CHECK:STDERR: ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F[self: Self](z: Self.U2) -> Self.U2;
}

// CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE+7]]:21: error: member access into undefined facet type `J2` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as J2 where .U2 = {} {
// CHECK:STDERR:                     ^~~
// CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE-22]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J2 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as J2 where .U2 = {} {
  fn F[self: Self](z: {}) -> {} { return z; }
}

class C2 {
  // This allows the type to be copyable so it can be returned.
  adapt {};
}

// CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE+7]]:21: error: member access into undefined facet type `J2` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C2 as J2 where .U2 = C2 {
// CHECK:STDERR:                     ^~~
// CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE-38]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J2 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl C2 as J2 where .U2 = C2 {
  fn F[self: Self](z: C2) -> C2 { return self; }
}

// --- fail_associated_type_in_signature_mismatch.carbon
library "[[@TEST_NAME]]";

interface K {
  let V:! type;
  fn F[self: Self](v: V) -> V;
}

// CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE+7]]:20: error: member access into undefined facet type `K` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as K where .V = {.a: ()} {
// CHECK:STDERR:                    ^~
// CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface K {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as K where .V = {.a: ()} {
  fn F[self: Self](v: {.x: ()}) -> {.x: ()} { return v; }
}

// --- use_non-type_in_function.carbon
library "[[@TEST_NAME]]";

interface M {
  let Z:! {.b: {}};
  fn G() -> {};
}

// CHECK:STDERR: use_non-type_in_function.carbon:[[@LINE+7]]:20: error: member access into undefined facet type `M` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as M where .Z = {.b = {}} {
// CHECK:STDERR:                    ^~
// CHECK:STDERR: use_non-type_in_function.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface M {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as M where .Z = {.b = {}} {
  fn G() -> {} {
    // CHECK:STDERR: use_non-type_in_function.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `Core.As(M)` [QualifiedExprInUndefinedFacetTypeScope]
    // CHECK:STDERR:     return (Self as M).Z.b;
    // CHECK:STDERR:             ^~~~~~~~~
    // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
    // CHECK:STDERR: interface As(Dest:! type) {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    return (Self as M).Z.b;
  }
}

// --- self_as_uses_correct_rewrite_constraint.carbon
library "[[@TEST_NAME]]";

class C(T:! type) {}

interface M {
  let Z:! {.b: type};
  fn G() -> type;
}

// CHECK:STDERR: self_as_uses_correct_rewrite_constraint.carbon:[[@LINE+7]]:23: error: member access into undefined facet type `M` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C({}) as M where .Z = {.b = {}} {
// CHECK:STDERR:                       ^~
// CHECK:STDERR: self_as_uses_correct_rewrite_constraint.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface M {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C({}) as M where .Z = {.b = {}} {
  fn G() -> type {
    // CHECK:STDERR: self_as_uses_correct_rewrite_constraint.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `Core.As(M)` [QualifiedExprInUndefinedFacetTypeScope]
    // CHECK:STDERR:     return (Self as M).Z.b;
    // CHECK:STDERR:             ^~~~~~~~~
    // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
    // CHECK:STDERR: interface As(Dest:! type) {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    return (Self as M).Z.b;
  }
}

// CHECK:STDERR: self_as_uses_correct_rewrite_constraint.carbon:[[@LINE+7]]:23: error: member access into undefined facet type `M` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C(()) as M where .Z = {.b = ()} {
// CHECK:STDERR:                       ^~
// CHECK:STDERR: self_as_uses_correct_rewrite_constraint.carbon:[[@LINE-28]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface M {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C(()) as M where .Z = {.b = ()} {
  fn G() -> type {
    // CHECK:STDERR: self_as_uses_correct_rewrite_constraint.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `Core.As(M)` [QualifiedExprInUndefinedFacetTypeScope]
    // CHECK:STDERR:     return (Self as M).Z.b;
    // CHECK:STDERR:             ^~~~~~~~~
    // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
    // CHECK:STDERR: interface As(Dest:! type) {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    return (Self as M).Z.b;
  }
}

// --- associated_int_in_array.carbon
library "[[@TEST_NAME]]";

interface I {
  let N:! i32;
  // CHECK:STDERR: associated_int_in_array.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Core.ImplicitAs(Core.IntLiteral)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   fn F[self: Self]() -> array(bool, N);
  // CHECK:STDERR:                                     ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F[self: Self]() -> array(bool, N);
}

// CHECK:STDERR: associated_int_in_array.carbon:[[@LINE+7]]:20: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as I where .N = 2 {
// CHECK:STDERR:                    ^~
// CHECK:STDERR: associated_int_in_array.carbon:[[@LINE-15]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as I where .N = 2 {
  fn F[self: Self]() -> array(bool, 2) { return (true, false); }
}

// --- fail_todo_symbolic_associated_type_in_concrete_context.carbon

interface Z {
  let X:! type;
}

class C(T:! type) {}
class D {}

// CHECK:STDERR: fail_todo_symbolic_associated_type_in_concrete_context.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] T as Z where .X = C(T) {}
// CHECK:STDERR:                                     ^~
// CHECK:STDERR: fail_todo_symbolic_associated_type_in_concrete_context.carbon:[[@LINE-10]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z where .X = C(T) {}

fn F() {
  // TODO: `D.(Z.X)` is `C(D)`, but is incorrecty treated only as the symbolic
  // `C(T)` here which fails to convert.

  // CHECK:STDERR: fail_todo_symbolic_associated_type_in_concrete_context.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: D.(Z.X) = {} as C(D);
  // CHECK:STDERR:             ^~~
  // CHECK:STDERR: fail_todo_symbolic_associated_type_in_concrete_context.carbon:[[@LINE-23]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: D.(Z.X) = {} as C(D);
}

// CHECK:STDOUT: --- associated_type_in_method_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self.ccd, element0 [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %Self.as_type, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type.fc8: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.f7b: %F.type.fc8 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %CallMethod.type: type = fn_type @CallMethod [concrete]
// CHECK:STDOUT:   %CallMethod: %CallMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type.78e: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.5fe: %F.type.78e = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Add = %Core.Add
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .CallMethod = %CallMethod.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.de5 [concrete] {} {
// CHECK:STDOUT:     %.loc15_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_7.2: type = converted %.loc15_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %U.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallMethod.decl: %CallMethod.type = fn_decl @CallMethod [concrete = constants.%CallMethod] {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_tuple.type = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc26_19.1: type = splice_block %.loc26_19.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc26_19.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc26_19.3: type = converted %.loc26_19.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.e4d [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc49_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     %C.ref.loc49_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc49: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %U.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c14 = fn_decl @F.1 [concrete = constants.%F.b71] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %u.patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = value_param_pattern %u.patt, call_param1
// CHECK:STDOUT:     %return.patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_29: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_29: type = impl_witness_access %Self.as_wit.iface0.loc5_29, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %U.ref.loc5_29: type = name_ref U, %impl.elem0.loc5_29 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = value_param call_param1
// CHECK:STDOUT:     %.loc5_23: type = splice_block %U.ref.loc5_23 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_23.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_23.2: type = impl_witness_access %Self.as_wit.iface0.loc5_23.2, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %U.ref.loc5_23: type = name_ref U, %impl.elem0.loc5_23.2 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = out_param call_param2
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.de5: %.loc15_7.2 as %.loc15_14 {
// CHECK:STDOUT:   %F.decl: %F.type.fc8 = fn_decl @F.2 [concrete = constants.%F.f7b] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %u.patt: %i32 = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %i32 = value_param_pattern %u.patt, call_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc23_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.de5.%.loc15_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc23: type = splice_block %i32.loc23_23 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc23_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc23_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %i32 = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.e4d: %C.ref.loc49_6 as %.loc49 {
// CHECK:STDOUT:   %F.decl: %F.type.78e = fn_decl @F.3 [concrete = constants.%F.5fe] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %u.patt: %C = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %C = value_param_pattern %u.patt, call_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc50_29: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.e4d.%C.ref.loc49_6 [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc50_23: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %u: %C = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc39_10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc39_11: type = converted %.loc39_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc39_11 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_23.1: type = impl_witness_access %Self.as_wit.iface0.loc5_23.1, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type)](%u.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0)) -> @F.1.%impl.elem0.loc5_23.1 (%impl.elem0);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type](%u.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallMethod(%x.param_patt: %empty_tuple.type) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, %x
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete = constants.%int_40]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[%self.param_patt: %C](%u.param_patt: %C) -> %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   return %self.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_type_in_function_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self.ccd, element0 [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %Self.as_type, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type.5a1: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.830: %F.type.5a1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %CallFunction.type: type = fn_type @CallFunction [concrete]
// CHECK:STDOUT:   %CallFunction: %CallFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Add = %Core.Add
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .CallFunction = %CallFunction.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc17: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %U.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFunction.decl: %CallFunction.type = fn_decl @CallFunction [concrete = constants.%CallFunction] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c14 = fn_decl @F.1 [concrete = constants.%F.b71] {
// CHECK:STDOUT:     %u.patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_17: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access %Self.as_wit.iface0.loc5_17, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %u.param: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_11.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access %Self.as_wit.iface0.loc5_11.2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %D.ref as %.loc17 {
// CHECK:STDOUT:   %F.decl: %F.type.5a1 = fn_decl @F.2 [concrete = constants.%F.830] {
// CHECK:STDOUT:     %u.patt: %i32 = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %i32 = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc25_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc25_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc25: type = splice_block %i32.loc25_11 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc25_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc25_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %i32 = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %Self.as_wit.iface0.loc5_11.1, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0)) -> @F.1.%impl.elem0.loc5_11.1 (%impl.elem0);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%u.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFunction() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_impl_with_associated_type_in_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0.1e6: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self.ccd, element0 [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %Self.as_type.3df, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %G.type.285: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.e4b: %G.type.285 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc2: %J.assoc_type = assoc_entity element2, @J.%G.decl [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type.ae6: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.920: %F.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type.823: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.1be: %G.type.823 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %CallBoth.type: type = fn_type @CallBoth [concrete]
// CHECK:STDOUT:   %CallBoth: %CallBoth.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %T: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %J.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %GenericCallF.type: type = fn_type @GenericCallF [concrete]
// CHECK:STDOUT:   %GenericCallF: %GenericCallF.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.2a5: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Add = %Core.Add
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .CallBoth = %CallBoth.decl
// CHECK:STDOUT:     .GenericCallF = %GenericCallF.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   %CallBoth.decl: %CallBoth.type = fn_decl @CallBoth [concrete = constants.%CallBoth] {
// CHECK:STDOUT:     %e.patt: %E = binding_pattern e
// CHECK:STDOUT:     %e.param_patt: %E = value_param_pattern %e.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %e.param: %E = value_param call_param0
// CHECK:STDOUT:     %E.ref.loc45: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %e: %E = bind_name e, %e.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericCallF.decl: %GenericCallF.type = fn_decl @GenericCallF [concrete = constants.%GenericCallF] {
// CHECK:STDOUT:     %T.patt.loc98_17.1: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc98_17.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.patt: @GenericCallF.%T.as_type.loc98_27.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @GenericCallF.%T.as_type.loc98_27.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:     %u.patt: <error> = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: <error> = value_param_pattern %u.patt, call_param1 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc98_41: %J.type = name_ref T, %T.loc98_17.1 [symbolic = %T.loc98_17.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc98_42: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %T.loc98_17.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc98_17.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: @GenericCallF.%T.as_type.loc98_27.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc98_27.1: type = splice_block %.loc98_27.2 [symbolic = %T.as_type.loc98_27.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc98_27: %J.type = name_ref T, %T.loc98_17.1 [symbolic = %T.loc98_17.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc98_27.1: type = facet_access_type %T.ref.loc98_27 [symbolic = %T.as_type.loc98_27.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc98_27.2: type = converted %T.ref.loc98_27, %T.as_type.loc98_27.1 [symbolic = %T.as_type.loc98_27.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericCallF.%T.as_type.loc98_27.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:     %u.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc98_33: %J.type = name_ref T, %T.loc98_17.1 [symbolic = %T.loc98_17.2 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc98_34: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: <error> = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param2
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %e.patt: %E = binding_pattern e
// CHECK:STDOUT:     %e.param_patt: %E = value_param_pattern %e.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %e.param: %E = value_param call_param0
// CHECK:STDOUT:     %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %e: %E = bind_name e, %e.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.1e6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c14 = fn_decl @F.1 [concrete = constants.%F.b71] {
// CHECK:STDOUT:     %u.patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_17: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access %Self.as_wit.iface0.loc5_17, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %u.param: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_11.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access %Self.as_wit.iface0.loc5_11.2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:   %G.decl: %G.type.285 = fn_decl @G.1 [concrete = constants.%G.e4b] {
// CHECK:STDOUT:     %self.patt: @G.1.%Self.as_type.loc6_14.1 (%Self.as_type.3df) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @G.1.%Self.as_type.loc6_14.1 (%Self.as_type.3df) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %v.patt: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0) = binding_pattern v
// CHECK:STDOUT:     %v.param_patt: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0) = value_param_pattern %v.patt, call_param1
// CHECK:STDOUT:     %return.patt: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc6_29: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc6_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc6_29: type = impl_witness_access %Self.as_wit.iface0.loc6_29, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %U.ref.loc6_29: type = name_ref U, %impl.elem0.loc6_29 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %self.param: @G.1.%Self.as_type.loc6_14.1 (%Self.as_type.3df) = value_param call_param0
// CHECK:STDOUT:     %.loc6_14.1: type = splice_block %.loc6_14.2 [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.3df)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:       %Self.as_type.loc6_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:       %.loc6_14.2: type = converted %Self.ref, %Self.as_type.loc6_14.2 [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @G.1.%Self.as_type.loc6_14.1 (%Self.as_type.3df) = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0) = value_param call_param1
// CHECK:STDOUT:     %.loc6_23: type = splice_block %U.ref.loc6_23 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc6_23.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc6_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc6_23.2: type = impl_witness_access %Self.as_wit.iface0.loc6_23.2, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %U.ref.loc6_23: type = name_ref U, %impl.elem0.loc6_23.2 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0) = bind_name v, %v.param
// CHECK:STDOUT:     %return.param: ref @G.1.%impl.elem0.loc6_23.1 (%impl.elem0) = out_param call_param2
// CHECK:STDOUT:     %return: ref @G.1.%impl.elem0.loc6_23.1 (%impl.elem0) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc2: %J.assoc_type = assoc_entity element2, %G.decl [concrete = constants.%assoc2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   .G = %assoc2
// CHECK:STDOUT:   witness = (%U, %F.decl, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %.loc21 {
// CHECK:STDOUT:   %F.decl: %F.type.ae6 = fn_decl @F.2 [concrete = constants.%F.920] {
// CHECK:STDOUT:     %u.patt: %i32 = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %i32 = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc22_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22_21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc22: type = splice_block %i32.loc22_13 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc22_13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc22_13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %i32 = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.823 = fn_decl @G.2 [concrete = constants.%G.1be] {
// CHECK:STDOUT:     %self.patt: %E = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %E = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %v.patt: %i32 = binding_pattern v
// CHECK:STDOUT:     %v.param_patt: %i32 = value_param_pattern %v.patt, call_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc32_33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc32_33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %E = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %self: %E = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc32: type = splice_block %i32.loc32_25 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc32_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc32_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: %i32 = bind_name v, %v.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc21: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %U.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %Self.as_wit.iface0.loc5_11.1, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0)) -> @F.1.%impl.elem0.loc5_11.1 (%impl.elem0);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_type.loc6_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc6_23.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc6_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc6_23.1: type = impl_witness_access %Self.as_wit.iface0.loc6_23.1, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.1.%Self.as_type.loc6_14.1 (%Self.as_type.3df)](%v.param_patt: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0)) -> @G.1.%impl.elem0.loc6_23.1 (%impl.elem0);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%u.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[%self.param_patt: %E](%v.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: %i32 = name_ref v, %v
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallBoth(%e.param_patt: %E) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %e.ref.loc81_10: %E = name_ref e, %e
// CHECK:STDOUT:   %F.ref.loc81_11: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %e.ref.loc81_19: %E = name_ref e, %e
// CHECK:STDOUT:   %G.ref.loc81_20: <error> = name_ref G, <error> [concrete = <error>]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %E.ref.loc81_28: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %F.ref.loc81_29: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %e.ref.loc81_37: %E = name_ref e, %e
// CHECK:STDOUT:   %E.ref.loc81_40: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %G.ref.loc81_41: <error> = name_ref G, <error> [concrete = <error>]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %e.ref.loc81_50: %E = name_ref e, %e
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %G.ref.loc81_54: <error> = name_ref G, <error> [concrete = <error>]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallF(%T.loc98_17.1: %J.type) {
// CHECK:STDOUT:   %T.loc98_17.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc98_17.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc98_17.2: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc98_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc98_27.2: type = facet_access_type %T.loc98_17.2 [symbolic = %T.as_type.loc98_27.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @GenericCallF.%T.as_type.loc98_27.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete.2a5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc98_17.1: %J.type](%t.param_patt: @GenericCallF.%T.as_type.loc98_27.2 (%T.as_type), %u.param_patt: <error>) -> <error> {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericCallF.%T.as_type.loc98_27.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:     %u.ref: <error> = name_ref u, %u [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGeneric(%e.param_patt: %E) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %GenericCallF.ref: %GenericCallF.type = name_ref GenericCallF, file.%GenericCallF.decl [concrete = constants.%GenericCallF]
// CHECK:STDOUT:   %e.ref: %E = name_ref e, %e
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc120: %J.type = converted constants.%E, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_type.loc6_14.1 => constants.%Self.as_type.3df
// CHECK:STDOUT:   %Self.as_wit.iface0.loc6_23.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc6_23.1 => constants.%impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallF(constants.%T) {
// CHECK:STDOUT:   %T.loc98_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc98_17.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc98_27.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_constraint_on_associated_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self.ccd, element0 [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %Self.as_type, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0: %Add.type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %T: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %J.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %GenericAddResult.type: type = fn_type @GenericAddResult [concrete]
// CHECK:STDOUT:   %GenericAddResult: %GenericAddResult.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Add = %Core.Add
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .GenericAddResult = %GenericAddResult.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %GenericAddResult.decl: %GenericAddResult.type = fn_decl @GenericAddResult [concrete = constants.%GenericAddResult] {
// CHECK:STDOUT:     %T.patt.loc22_21.1: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_21.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.patt: @GenericAddResult.%T.as_type.loc22_31.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @GenericAddResult.%T.as_type.loc22_31.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:     %u.patt: <error> = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: <error> = value_param_pattern %u.patt, call_param1 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc22_45: %J.type = name_ref T, %T.loc22_21.1 [symbolic = %T.loc22_21.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc22_46: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %T.loc22_21.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc22_21.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: @GenericAddResult.%T.as_type.loc22_31.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc22_31.1: type = splice_block %.loc22_31.2 [symbolic = %T.as_type.loc22_31.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc22_31: %J.type = name_ref T, %T.loc22_21.1 [symbolic = %T.loc22_21.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc22_31.1: type = facet_access_type %T.ref.loc22_31 [symbolic = %T.as_type.loc22_31.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc22_31.2: type = converted %T.ref.loc22_31, %T.as_type.loc22_31.1 [symbolic = %T.as_type.loc22_31.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericAddResult.%T.as_type.loc22_31.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:     %u.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc22_37: %J.type = name_ref T, %T.loc22_21.1 [symbolic = %T.loc22_21.2 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc22_38: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: <error> = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param2
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: %Add.type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %u.patt: @F.%as_type (%as_type) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @F.%as_type (%as_type) = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%as_type (%as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%as_type (%as_type) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_17: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_17: %Add.type = impl_witness_access %Self.as_wit.iface0.loc5_17, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %U.ref.loc5_17: %Add.type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %U.as_type.loc5_17: type = facet_access_type %U.ref.loc5_17 [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:     %.loc5_17: type = converted %U.ref.loc5_17, %U.as_type.loc5_17 [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:     %u.param: @F.%as_type (%as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_11.1: type = splice_block %.loc5_11.2 [symbolic = %as_type (constants.%as_type)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_11.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: %Add.type = impl_witness_access %Self.as_wit.iface0.loc5_11.2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %U.ref.loc5_11: %Add.type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %U.as_type.loc5_11: type = facet_access_type %U.ref.loc5_11 [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:       %.loc5_11.2: type = converted %U.ref.loc5_11, %U.as_type.loc5_11 [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.%as_type (%as_type) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.%as_type (%as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%as_type (%as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! %Add.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: %Add.type = impl_witness_access %Self.as_wit.iface0.loc5_11.1, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.loc5_11.1 [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param_patt: @F.%as_type (%as_type)) -> @F.%as_type (%as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericAddResult(%T.loc22_21.1: %J.type) {
// CHECK:STDOUT:   %T.loc22_21.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc22_21.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc22_21.2: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc22_31.2: type = facet_access_type %T.loc22_21.2 [symbolic = %T.as_type.loc22_31.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @GenericAddResult.%T.as_type.loc22_31.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc22_21.1: %J.type](%t.param_patt: @GenericAddResult.%T.as_type.loc22_31.2 (%T.as_type), %u.param_patt: <error>) -> <error> {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericAddResult.%T.as_type.loc22_31.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %F.ref.loc44_11: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:     %u.ref.loc44_14: <error> = name_ref u, %u [concrete = <error>]
// CHECK:STDOUT:     %T.ref.loc44: %J.type = name_ref T, %T.loc22_21.1 [symbolic = %T.loc22_21.2 (constants.%T)]
// CHECK:STDOUT:     %F.ref.loc44_20: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:     %u.ref.loc44_23: <error> = name_ref u, %u [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0
// CHECK:STDOUT:   %as_type => constants.%as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAddResult(constants.%T) {
// CHECK:STDOUT:   %T.loc22_21.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc22_21.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc22_31.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface_qualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self, element0 [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %Self.as_type, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%G.decl [concrete]
// CHECK:STDOUT:   %T: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %J.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %GenericCallInterfaceQualified.type: type = fn_type @GenericCallInterfaceQualified [concrete]
// CHECK:STDOUT:   %GenericCallInterfaceQualified: %GenericCallInterfaceQualified.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
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
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .GenericCallInterfaceQualified = %GenericCallInterfaceQualified.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %GenericCallInterfaceQualified.decl: %GenericCallInterfaceQualified.type = fn_decl @GenericCallInterfaceQualified [concrete = constants.%GenericCallInterfaceQualified] {
// CHECK:STDOUT:     %T.patt.loc22_34.1: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_34.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.patt: @GenericCallInterfaceQualified.%T.as_type.loc22_44.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @GenericCallInterfaceQualified.%T.as_type.loc22_44.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:     %u.patt: <error> = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: <error> = value_param_pattern %u.patt, call_param1 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc22_58: %J.type = name_ref T, %T.loc22_34.1 [symbolic = %T.loc22_34.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc22_59: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     %J.ref.loc22: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %T.loc22_34.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc22_34.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: @GenericCallInterfaceQualified.%T.as_type.loc22_44.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc22_44.1: type = splice_block %.loc22_44.2 [symbolic = %T.as_type.loc22_44.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc22_44: %J.type = name_ref T, %T.loc22_34.1 [symbolic = %T.loc22_34.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc22_44.1: type = facet_access_type %T.ref.loc22_44 [symbolic = %T.as_type.loc22_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc22_44.2: type = converted %T.ref.loc22_44, %T.as_type.loc22_44.1 [symbolic = %T.as_type.loc22_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericCallInterfaceQualified.%T.as_type.loc22_44.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:     %u.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc22_50: %J.type = name_ref T, %T.loc22_34.1 [symbolic = %T.loc22_34.2 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc22_51: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: <error> = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param2
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %self.patt: @G.%Self.as_type.loc5_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @G.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %v.patt: @G.%impl.elem0.loc5_23.1 (%impl.elem0) = binding_pattern v
// CHECK:STDOUT:     %v.param_patt: @G.%impl.elem0.loc5_23.1 (%impl.elem0) = value_param_pattern %v.patt, call_param1
// CHECK:STDOUT:     %return.patt: @G.%impl.elem0.loc5_23.1 (%impl.elem0) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @G.%impl.elem0.loc5_23.1 (%impl.elem0) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_29: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_29: type = impl_witness_access %Self.as_wit.iface0.loc5_29, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %U.ref.loc5_29: type = name_ref U, %impl.elem0.loc5_29 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %self.param: @G.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @G.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: @G.%impl.elem0.loc5_23.1 (%impl.elem0) = value_param call_param1
// CHECK:STDOUT:     %.loc5_23: type = splice_block %U.ref.loc5_23 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_23.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_23.2: type = impl_witness_access %Self.as_wit.iface0.loc5_23.2, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %U.ref.loc5_23: type = name_ref U, %impl.elem0.loc5_23.2 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @G.%impl.elem0.loc5_23.1 (%impl.elem0) = bind_name v, %v.param
// CHECK:STDOUT:     %return.param: ref @G.%impl.elem0.loc5_23.1 (%impl.elem0) = out_param call_param2
// CHECK:STDOUT:     %return: ref @G.%impl.elem0.loc5_23.1 (%impl.elem0) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%U, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_23.1: type = impl_witness_access %Self.as_wit.iface0.loc5_23.1, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.%Self.as_type.loc5_14.1 (%Self.as_type)](%v.param_patt: @G.%impl.elem0.loc5_23.1 (%impl.elem0)) -> @G.%impl.elem0.loc5_23.1 (%impl.elem0);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallInterfaceQualified(%T.loc22_34.1: %J.type) {
// CHECK:STDOUT:   %T.loc22_34.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc22_34.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc22_34.2: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_34.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc22_44.2: type = facet_access_type %T.loc22_34.2 [symbolic = %T.as_type.loc22_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @GenericCallInterfaceQualified.%T.as_type.loc22_44.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc22_34.1: %J.type](%t.param_patt: @GenericCallInterfaceQualified.%T.as_type.loc22_44.2 (%T.as_type), %u.param_patt: <error>) -> <error> {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericCallInterfaceQualified.%T.as_type.loc22_44.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %J.ref.loc30: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %G.ref: <error> = name_ref G, <error> [concrete = <error>]
// CHECK:STDOUT:     %u.ref: <error> = name_ref u, %u [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallInterfaceQualified(constants.%T) {
// CHECK:STDOUT:   %T.loc22_34.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc22_34.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc22_44.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self, element0 [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %Self.as_type, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %T.patt: <error> = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %GenericCallFI32.type: type = fn_type @GenericCallFI32 [concrete]
// CHECK:STDOUT:   %GenericCallFI32: %GenericCallFI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
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
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .GenericCallFI32 = %GenericCallFI32.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %GenericCallFI32.decl: %GenericCallFI32.type = fn_decl @GenericCallFI32 [concrete = constants.%GenericCallFI32] {
// CHECK:STDOUT:     %T.patt.loc15_20.1: <error> = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_20.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.patt: <error> = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: <error> = value_param_pattern %t.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc15_51: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_51: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15_26.1: type = splice_block %.loc15_26.2 [concrete = <error>] {
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.ref: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:       %int_32.loc15_37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc15_37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc15_26.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %U.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T: <error> = bind_symbolic_name T, 0 [concrete = <error>]
// CHECK:STDOUT:     %t.param: <error> = value_param call_param0
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, %T [concrete = <error>]
// CHECK:STDOUT:     %t: <error> = bind_name t, %t.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %u.patt: @F.%impl.elem0.loc5_11.1 (%impl.elem0) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @F.%impl.elem0.loc5_11.1 (%impl.elem0) = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%impl.elem0.loc5_11.1 (%impl.elem0) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%impl.elem0.loc5_11.1 (%impl.elem0) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_17: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access %Self.as_wit.iface0.loc5_17, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %u.param: @F.%impl.elem0.loc5_11.1 (%impl.elem0) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_11.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access %Self.as_wit.iface0.loc5_11.2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.%impl.elem0.loc5_11.1 (%impl.elem0) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.%impl.elem0.loc5_11.1 (%impl.elem0) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%impl.elem0.loc5_11.1 (%impl.elem0) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %Self.as_wit.iface0.loc5_11.1, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param_patt: @F.%impl.elem0.loc5_11.1 (%impl.elem0)) -> @F.%impl.elem0.loc5_11.1 (%impl.elem0);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallFI32(%T: <error>) {
// CHECK:STDOUT:   %T.patt.loc15_20.2: <error> = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_20.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc15_20.1: <error>](%t.param_patt: <error>) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: <error> = name_ref t, %t [concrete = <error>]
// CHECK:STDOUT:     %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallFI32(<error>) {
// CHECK:STDOUT:   %T.patt.loc15_20.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_associated_constant_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self.ccd, element0 [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %Self.as_type, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type.ae6: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.920: %F.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Add = %Core.Add
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c14 = fn_decl @F.1 [concrete = constants.%F.b71] {
// CHECK:STDOUT:     %u.patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_17: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access %Self.as_wit.iface0.loc5_17, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %u.param: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_11.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access %Self.as_wit.iface0.loc5_11.2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %.loc20 {
// CHECK:STDOUT:   %F.decl: %F.type.ae6 = fn_decl @F.2 [concrete = constants.%F.920] {
// CHECK:STDOUT:     %u.patt: <error> = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: <error> = value_param_pattern %u.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc29_19: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     %u.param: <error> = value_param call_param0
// CHECK:STDOUT:     %U.ref.loc29_13: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     %u: <error> = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param1
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .U = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc20: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %U.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .U = <poisoned>
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %Self.as_wit.iface0.loc5_11.1, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0)) -> @F.1.%impl.elem0.loc5_11.1 (%impl.elem0);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%u.param_patt: <error>) -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: <error> = name_ref u, %u [concrete = <error>]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_self_period_associated_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J2.type: type = facet_type <@J2> [concrete]
// CHECK:STDOUT:   %Self: %J2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J2.assoc_type: type = assoc_entity_type %J2.type [concrete]
// CHECK:STDOUT:   %assoc0: %J2.assoc_type = assoc_entity element0, @J2.%U2 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type.c41: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.932: %F.type.c41 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J2.assoc_type = assoc_entity element1, @J2.%F.decl [concrete]
// CHECK:STDOUT:   %.Self: %J2.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %F.type.fc8: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.f7b: %F.type.fc8 = struct_value () [concrete]
// CHECK:STDOUT:   %C2: type = class_type @C2 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type.c02: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.e2b: %F.type.c02 = struct_value () [concrete]
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
// CHECK:STDOUT:     .J2 = %J2.decl
// CHECK:STDOUT:     .C2 = %C2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J2.decl: type = interface_decl @J2 [concrete = constants.%J2.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.de5 [concrete] {} {
// CHECK:STDOUT:     %.loc29_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc29_7.2: type = converted %.loc29_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J2.ref: type = name_ref J2, file.%J2.decl [concrete = constants.%J2.type]
// CHECK:STDOUT:     %.Self: %J2.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U2.ref: <error> = name_ref U2, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc29_28: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc29_15: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %U2.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2 [concrete = constants.%C2] {} {}
// CHECK:STDOUT:   impl_decl @impl.fcc [concrete] {} {
// CHECK:STDOUT:     %C2.ref.loc45_6: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %J2.ref: type = name_ref J2, file.%J2.decl [concrete = constants.%J2.type]
// CHECK:STDOUT:     %.Self: %J2.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U2.ref: <error> = name_ref U2, <error> [concrete = <error>]
// CHECK:STDOUT:     %C2.ref.loc45_27: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %.loc45: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %U2.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J2 {
// CHECK:STDOUT:   %Self: %J2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %U2: type = assoc_const_decl @U2 [concrete] {
// CHECK:STDOUT:     %assoc0: %J2.assoc_type = assoc_entity element0, @J2.%U2 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c41 = fn_decl @F.1 [concrete = constants.%F.932] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %z.patt: <error> = binding_pattern z
// CHECK:STDOUT:     %z.param_patt: <error> = value_param_pattern %z.patt, call_param1 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc19_35: %J2.type = name_ref Self, @J2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc19_14.1: type = splice_block %.loc19_14.2 [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc19_14: %J2.type = name_ref Self, @J2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc19_14.2: type = facet_access_type %Self.ref.loc19_14 [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc19_14.2: type = converted %Self.ref.loc19_14, %Self.as_type.loc19_14.2 [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %z.param: <error> = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc19_23: %J2.type = name_ref Self, @J2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %z: <error> = bind_name z, %z.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param2
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J2.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U2 = @U2.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U2, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U2(@J2.%Self: %J2.type) {
// CHECK:STDOUT:   assoc_const U2:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.de5: %.loc29_7.2 as %.loc29_15 {
// CHECK:STDOUT:   %F.decl: %F.type.fc8 = fn_decl @F.2 [concrete = constants.%F.f7b] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %z.patt: %empty_struct_type = binding_pattern z
// CHECK:STDOUT:     %z.param_patt: %empty_struct_type = value_param_pattern %z.patt, call_param1
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc30_31.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc30_31.2: type = converted %.loc30_31.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.de5.%.loc29_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %z.param: %empty_struct_type = value_param call_param1
// CHECK:STDOUT:     %.loc30_24.1: type = splice_block %.loc30_24.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc30_24.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc30_24.3: type = converted %.loc30_24.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %z: %empty_struct_type = bind_name z, %z.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param2
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.fcc: %C2.ref.loc45_6 as %.loc45 {
// CHECK:STDOUT:   %F.decl: %F.type.c02 = fn_decl @F.3 [concrete = constants.%F.e2b] {
// CHECK:STDOUT:     %self.patt: %C2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C2 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %z.patt: %C2 = binding_pattern z
// CHECK:STDOUT:     %z.param_patt: %C2 = value_param_pattern %z.patt, call_param1
// CHECK:STDOUT:     %return.patt: %C2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C2 = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C2.ref.loc46_30: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %self.param: %C2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.fcc.%C2.ref.loc45_6 [concrete = constants.%C2]
// CHECK:STDOUT:     %self: %C2 = bind_name self, %self.param
// CHECK:STDOUT:     %z.param: %C2 = value_param call_param1
// CHECK:STDOUT:     %C2.ref.loc46_23: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %z: %C2 = bind_name z, %z.param
// CHECK:STDOUT:     %return.param: ref %C2 = out_param call_param2
// CHECK:STDOUT:     %return: ref %C2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C2 = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %.loc35_10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc35_11: type = converted %.loc35_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc35_11 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J2.%Self: %J2.type) {
// CHECK:STDOUT:   %Self: %J2.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc19_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type)](%z.param_patt: <error>) -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type](%z.param_patt: %empty_struct_type) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %z.ref: %empty_struct_type = name_ref z, %z
// CHECK:STDOUT:   return %z.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[%self.param_patt: %C2](%z.param_patt: %C2) -> %C2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C2 = name_ref self, %self
// CHECK:STDOUT:   return %self.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U2(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc19_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_associated_type_in_signature_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type %K.type [concrete]
// CHECK:STDOUT:   %assoc0: %K.assoc_type = assoc_entity element0, @K.%V [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self, element0 [symbolic]
// CHECK:STDOUT:   %K.facet: %K.type = facet_value %Self.as_type, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %F.type.cf5: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.06b: %F.type.cf5 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %K.assoc_type = assoc_entity element1, @K.%F.decl [concrete]
// CHECK:STDOUT:   %.Self: %K.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %F.type.fc8: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.f7b: %F.type.fc8 = struct_value () [concrete]
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
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %K.decl: type = interface_decl @K [concrete = constants.%K.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc15_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_7.2: type = converted %.loc15_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:     %.Self: %K.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %K.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.ref: <error> = name_ref V, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc15_31.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_31.2: type = converted %.loc15_31.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %.loc15_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %V.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %V: type = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %K.assoc_type = assoc_entity element0, @K.%V [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.cf5 = fn_decl @F.1 [concrete = constants.%F.06b] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %v.patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = binding_pattern v
// CHECK:STDOUT:     %v.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = value_param_pattern %v.patt, call_param1
// CHECK:STDOUT:     %return.patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_29: <witness> = facet_access_witness @K.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_29: type = impl_witness_access %Self.as_wit.iface0.loc5_29, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %V.ref.loc5_29: type = name_ref V, %impl.elem0.loc5_29 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %K.type = name_ref Self, @K.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = value_param call_param1
// CHECK:STDOUT:     %.loc5_23: type = splice_block %V.ref.loc5_23 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_23.2: <witness> = facet_access_witness @K.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_23.2: type = impl_witness_access %Self.as_wit.iface0.loc5_23.2, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %V.ref.loc5_23: type = name_ref V, %impl.elem0.loc5_23.2 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = bind_name v, %v.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = out_param call_param2
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_23.1 (%impl.elem0) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %K.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%V, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@K.%Self: %K.type) {
// CHECK:STDOUT:   assoc_const V:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc15_7.2 as %.loc15_14 {
// CHECK:STDOUT:   %F.decl: %F.type.fc8 = fn_decl @F.2 [concrete = constants.%F.f7b] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %v.patt: %struct_type.x = binding_pattern v
// CHECK:STDOUT:     %v.param_patt: %struct_type.x = value_param_pattern %v.patt, call_param1
// CHECK:STDOUT:     %return.patt: %struct_type.x = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %struct_type.x = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_42.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_42.2: type = converted %.loc16_42.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.x.loc16_43: type = struct_type {.x: %empty_tuple.type} [concrete = constants.%struct_type.x]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%.loc15_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: %struct_type.x = value_param call_param1
// CHECK:STDOUT:     %.loc16_30: type = splice_block %struct_type.x.loc16_30 [concrete = constants.%struct_type.x] {
// CHECK:STDOUT:       %.loc16_29.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc16_29.2: type = converted %.loc16_29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.loc16_30: type = struct_type {.x: %empty_tuple.type} [concrete = constants.%struct_type.x]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: %struct_type.x = bind_name v, %v.param
// CHECK:STDOUT:     %return.param: ref %struct_type.x = out_param call_param2
// CHECK:STDOUT:     %return: ref %struct_type.x = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@K.%Self: %K.type) {
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_23.1: type = impl_witness_access %Self.as_wit.iface0.loc5_23.1, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type)](%v.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0)) -> @F.1.%impl.elem0.loc5_23.1 (%impl.elem0);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type](%v.param_patt: %struct_type.x) -> %struct_type.x {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: %struct_type.x = name_ref v, %v
// CHECK:STDOUT:   return %v.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%K.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_non-type_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %Self.bcc: %M.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type %M.type [concrete]
// CHECK:STDOUT:   %assoc0.e3d: %M.assoc_type = assoc_entity element0, @M.%Z [concrete]
// CHECK:STDOUT:   %G.type.020: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.91c: %G.type.020 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, @M.%G.decl [concrete]
// CHECK:STDOUT:   %.Self: %M.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %G.type.321: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.a0c: %G.type.321 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [concrete = constants.%M.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc15_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_7.2: type = converted %.loc15_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Z.ref: <error> = name_ref Z, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc15_32: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc15_33: %struct_type.b.347 = struct_literal (%.loc15_32)
// CHECK:STDOUT:     %.loc15_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %Z.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.bcc]
// CHECK:STDOUT:   %Z: %struct_type.b.347 = assoc_const_decl @Z [concrete] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete = constants.%assoc0.e3d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.020 = fn_decl @G.1 [concrete = constants.%G.91c] {
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, %G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = @Z.%assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%Z, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Z(@M.%Self: %M.type) {
// CHECK:STDOUT:   assoc_const Z:! %struct_type.b.347;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc15_7.2 as %.loc15_14 {
// CHECK:STDOUT:   %G.decl: %G.type.321 = fn_decl @G.2 [concrete = constants.%G.a0c] {
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc16_14.2: type = converted %.loc16_14.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@M.%Self: %M.type) {
// CHECK:STDOUT:   fn() -> %empty_struct_type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @impl.%.loc15_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %.loc24: %M.type = converted %Self.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %Z.ref: <error> = name_ref Z, <error> [concrete = <error>]
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.bcc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.bcc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- self_as_uses_correct_rewrite_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %Self.bcc: %M.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.b.86f: type = struct_type {.b: type} [concrete]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type %M.type [concrete]
// CHECK:STDOUT:   %assoc0.fbc: %M.assoc_type = assoc_entity element0, @M.%Z [concrete]
// CHECK:STDOUT:   %G.type.020: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.91c: %G.type.020 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, @M.%G.decl [concrete]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %.Self: %M.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %G.type.d95: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.acc: %G.type.d95 = struct_value () [concrete]
// CHECK:STDOUT:   %C.3a0: type = class_type @C, @C(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %struct_type.b.161: type = struct_type {.b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %G.type.ebd: type = fn_type @G.3 [concrete]
// CHECK:STDOUT:   %G.5c7: %G.type.ebd = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc3_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [concrete = constants.%M.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.d2a [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.loc17_9: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_10: type = converted %.loc17_9, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%empty_struct_type) [concrete = constants.%C.7a7]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Z.ref: <error> = name_ref Z, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc17_35: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_36: %struct_type.b.347 = struct_literal (%.loc17_35)
// CHECK:STDOUT:     %.loc17_17: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %Z.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.c95 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.loc37_9: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc37_10: type = converted %.loc37_9, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%empty_tuple.type) [concrete = constants.%C.3a0]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Z.ref: <error> = name_ref Z, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc37_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc37_36: %struct_type.b.161 = struct_literal (%.loc37_35)
// CHECK:STDOUT:     %.loc37_17: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %Z.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.bcc]
// CHECK:STDOUT:   %Z: %struct_type.b.86f = assoc_const_decl @Z [concrete] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete = constants.%assoc0.fbc]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.020 = fn_decl @G.1 [concrete = constants.%G.91c] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, %G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = @Z.%assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%Z, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Z(@M.%Self: %M.type) {
// CHECK:STDOUT:   assoc_const Z:! %struct_type.b.86f;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.d2a: %C as %.loc17_17 {
// CHECK:STDOUT:   %G.decl: %G.type.d95 = fn_decl @G.2 [concrete = constants.%G.acc] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c95: %C as %.loc37_17 {
// CHECK:STDOUT:   %G.decl: %G.type.ebd = fn_decl @G.3 [concrete = constants.%G.5c7] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc3_9.1: type) {
// CHECK:STDOUT:   %T.loc3_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc3_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@M.%Self: %M.type) {
// CHECK:STDOUT:   fn() -> type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @impl.d2a.%C [concrete = constants.%C.7a7]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %.loc26: %M.type = converted %Self.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %Z.ref: <error> = name_ref Z, <error> [concrete = <error>]
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.3() -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @impl.c95.%C [concrete = constants.%C.3a0]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %.loc46: %M.type = converted %Self.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %Z.ref: <error> = name_ref Z, <error> [concrete = <error>]
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc3_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc3_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.bcc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.bcc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc3_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc3_9.2 => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.loc3_9.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %T.patt.loc3_9.2 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_int_in_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0.73c: %I.assoc_type = assoc_entity element0, @I.%N [concrete]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self.826, element0 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %Self.as_type.b70, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0: %i32 = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%F.decl [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, bool [concrete]
// CHECK:STDOUT:   %F.type.fc8: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.f7b: %F.type.fc8 = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (bool, bool) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%true, %false) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc22_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc22_7.2: type = converted %.loc22_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %N.ref: <error> = name_ref N, <error> [concrete = <error>]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %.loc22_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %N.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %N: %i32 = assoc_const_decl @N [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%N [concrete = constants.%assoc0.73c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [concrete = constants.%F.bc6] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.b70) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.b70) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %Self.as_wit.iface0.loc12_37.2: <witness> = facet_access_witness @I.%Self, element0 [symbolic = %Self.as_wit.iface0.loc12_37.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc12_37.2: %i32 = impl_witness_access %Self.as_wit.iface0.loc12_37.2, element0 [symbolic = %impl.elem0.loc12_37.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %impl.elem0.loc12_37.2 [symbolic = %impl.elem0.loc12_37.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %.loc12_31.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc12_31.2: type = converted %bool.make_type, %.loc12_31.1 [concrete = bool]
// CHECK:STDOUT:     %.loc12_37: Core.IntLiteral = converted %N.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     %array_type: type = array_type <error>, bool [concrete = <error>]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.b70) = value_param call_param0
// CHECK:STDOUT:     %.loc12_14.1: type = splice_block %.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc12_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc12_14.2: type = converted %Self.ref, %Self.as_type.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.b70) = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param1
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = @N.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%N, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @N(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const N:! %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc22_7.2 as %.loc22_14 {
// CHECK:STDOUT:   %F.decl: %F.type.fc8 = fn_decl @F.2 [concrete = constants.%F.f7b] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %array_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %array_type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %.loc23_31.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc23_31.2: type = converted %bool.make_type, %.loc23_31.1 [concrete = bool]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, bool [concrete = constants.%array_type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%.loc22_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %array_type = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type.loc12_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc12_37.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc12_37.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc12_37.1: %i32 = impl_witness_access %Self.as_wit.iface0.loc12_37.1, element0 [symbolic = %impl.elem0.loc12_37.1 (constants.%impl.elem0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.b70)]() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type]() -> %return.param_patt: %array_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc23_61.1: %tuple.type = tuple_literal (%true, %false)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc23_61.2: ref bool = array_index %return, %int_0
// CHECK:STDOUT:   %.loc23_61.3: init bool = initialize_from %true to %.loc23_61.2 [concrete = constants.%true]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc23_61.4: ref bool = array_index %return, %int_1
// CHECK:STDOUT:   %.loc23_61.5: init bool = initialize_from %false to %.loc23_61.4 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc23_61.6: init %array_type = array_init (%.loc23_61.3, %.loc23_61.5) to %return [concrete = constants.%array]
// CHECK:STDOUT:   %.loc23_62: init %array_type = converted %.loc23_61.1, %.loc23_61.6 [concrete = constants.%array]
// CHECK:STDOUT:   return %.loc23_62 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%Self.as_type.b70
// CHECK:STDOUT:   %Self.as_wit.iface0.loc12_37.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc12_37.1 => constants.%impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_symbolic_associated_type_in_concrete_context.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type %Z.type [concrete]
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %.Self: %Z.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.131: type = class_type @C, @C(%D) [concrete]
// CHECK:STDOUT:   %C.val: %C.131 = struct_value () [concrete]
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
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc6_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc16_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc16_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc16_24: type = name_ref T, %T.loc16_14.1 [symbolic = %T.loc16_14.2 (constants.%T)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.Self: %Z.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %Z.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref: <error> = name_ref X, <error> [concrete = <error>]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref.loc16_44: type = name_ref T, %T.loc16_14.1 [symbolic = %T.loc16_14.2 (constants.%T)]
// CHECK:STDOUT:     %C.loc16_45.1: type = class_type @C, @C(constants.%T) [symbolic = %C.loc16_45.2 (constants.%C.f2e)]
// CHECK:STDOUT:     %.loc16: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %X.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc16_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %Z.assoc_type = assoc_entity element0, @Z.%X [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .X = @X.%assoc0
// CHECK:STDOUT:   witness = (%X)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc16_14.1: type) {
// CHECK:STDOUT:   %T.loc16_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc16_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc16_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc16_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %C.loc16_45.2: type = class_type @C, @C(%T.loc16_14.2) [symbolic = %C.loc16_45.2 (constants.%C.f2e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc16_24 as %.loc16 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc6_9.1: type) {
// CHECK:STDOUT:   %T.loc6_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc29_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %D.ref.loc29_28: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%D) [concrete = constants.%C.131]
// CHECK:STDOUT:   %.loc29_21.2: ref %C.131 = temporary_storage
// CHECK:STDOUT:   %.loc29_21.3: init %C.131 = class_init (), %.loc29_21.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc29_21.4: ref %C.131 = temporary %.loc29_21.2, %.loc29_21.3
// CHECK:STDOUT:   %.loc29_23: ref %C.131 = converted %.loc29_21.1, %.loc29_21.4
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %D.ref.loc29_10: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %X.ref: <error> = name_ref X, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: <error> = bind_name a, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc6_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc16_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc16_14.2 => constants.%T
// CHECK:STDOUT:   %C.loc16_45.2 => constants.%C.f2e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@impl.%T.loc16_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%D) {
// CHECK:STDOUT:   %T.loc6_9.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc6_9.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
