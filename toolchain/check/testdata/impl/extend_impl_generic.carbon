// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/extend_impl_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/extend_impl_generic.carbon
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

// --- extend_impl_generic_interface.carbon

library "[[@TEST_NAME]]";

interface HasF(T:! type) {
  fn F() -> T;
}

class Param {
  var x: i32;
}

class C {
  extend impl as HasF(Param) {
    // CHECK:STDERR: extend_impl_generic_interface.carbon:[[@LINE+10]]:15: error: member access into undefined facet type `HasF(Param)` [QualifiedExprInUndefinedFacetTypeScope]
    // CHECK:STDERR:     fn F() -> Param {
    // CHECK:STDERR:               ^~~~~
    // CHECK:STDERR: extend_impl_generic_interface.carbon:[[@LINE-13]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
    // CHECK:STDERR: interface HasF(T:! type) {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: extend_impl_generic_interface.carbon:[[@LINE-7]]:18: note: declared as an extended scope here [FromExtendHere]
    // CHECK:STDERR:   extend impl as HasF(Param) {
    // CHECK:STDERR:                  ^~~~~~~~~~~
    // CHECK:STDERR:
    fn F() -> Param {
      // CHECK:STDERR: extend_impl_generic_interface.carbon:[[@LINE+7]]:14: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
      // CHECK:STDERR:       return {.x = 2};
      // CHECK:STDERR:              ^~~~~~~~
      // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-fastbuild/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
      // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
      // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // CHECK:STDERR:
      return {.x = 2};
    }
  }
}

fn G(c: C) {
  // CHECK:STDERR: extend_impl_generic_interface.carbon:[[@LINE+10]]:16: error: member access into undefined facet type `HasF(Param)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: i32 = C.F().x;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR: extend_impl_generic_interface.carbon:[[@LINE-37]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface HasF(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: extend_impl_generic_interface.carbon:[[@LINE-31]]:18: note: declared as an extended scope here [FromExtendHere]
  // CHECK:STDERR:   extend impl as HasF(Param) {
  // CHECK:STDERR:                  ^~~~~~~~~~~
  // CHECK:STDERR:
  let a: i32 = C.F().x;
  // CHECK:STDERR: extend_impl_generic_interface.carbon:[[@LINE+10]]:16: error: member access into undefined facet type `HasF(Param)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   var b: i32 = c.F().x;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR: extend_impl_generic_interface.carbon:[[@LINE-48]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface HasF(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: extend_impl_generic_interface.carbon:[[@LINE-42]]:18: note: declared as an extended scope here [FromExtendHere]
  // CHECK:STDERR:   extend impl as HasF(Param) {
  // CHECK:STDERR:                  ^~~~~~~~~~~
  // CHECK:STDERR:
  var b: i32 = c.F().x;
}

// --- extend_impl_generic_class.carbon

library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn F[self: Self](t: T);
}

class X(U:! type) {
  extend impl as I(U) {
    // CHECK:STDERR: extend_impl_generic_class.carbon:[[@LINE+10]]:25: error: member access into undefined facet type `I(U)` [QualifiedExprInUndefinedFacetTypeScope]
    // CHECK:STDERR:     fn F[self: Self](t: U) { }
    // CHECK:STDERR:                         ^
    // CHECK:STDERR: extend_impl_generic_class.carbon:[[@LINE-9]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
    // CHECK:STDERR: interface I(T:! type) {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: extend_impl_generic_class.carbon:[[@LINE-7]]:18: note: declared as an extended scope here [FromExtendHere]
    // CHECK:STDERR:   extend impl as I(U) {
    // CHECK:STDERR:                  ^~~~
    // CHECK:STDERR:
    fn F[self: Self](t: U) { }
  }
}

// CHECK:STDOUT: --- extend_impl_generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.type.fe3: type = generic_interface_type @HasF [concrete]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.fe3 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.901: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self.322: %HasF.type.901 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.46c: type = fn_type @F.1, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %F.823: %F.type.46c = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.595: type = assoc_entity_type %HasF.type.901 [symbolic]
// CHECK:STDOUT:   %assoc0.35e: %HasF.assoc_type.595 = assoc_entity element0, @HasF.%F.decl [symbolic]
// CHECK:STDOUT:   %Param: type = class_type @Param [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasF.type.b18: type = facet_type <@HasF, @HasF(%Param)> [concrete]
// CHECK:STDOUT:   %F.type.7f1: type = fn_type @F.1, @HasF(%Param) [concrete]
// CHECK:STDOUT:   %F.eff: %F.type.7f1 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type.dc4: type = assoc_entity_type %HasF.type.b18 [concrete]
// CHECK:STDOUT:   %assoc0.a6b: %HasF.assoc_type.dc4 = assoc_entity element0, @HasF.%F.decl [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.94c: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.901: %F.type.94c = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type.b18 = facet_value %C, (%impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.c96: type = struct_type {.x: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .Param = %Param.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: %HasF.type.fe3 = interface_decl @HasF [concrete = constants.%HasF.generic] {
// CHECK:STDOUT:     %T.patt.loc4_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_16.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Param.decl: type = class_decl @Param [concrete = constants.%Param] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc37: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_16.2)> [symbolic = %HasF.type (constants.%HasF.type.901)]
// CHECK:STDOUT:   %Self.2: %HasF.type.901 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.322)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @HasF(%T.loc4_16.2) [symbolic = %F.type (constants.%F.type.46c)]
// CHECK:STDOUT:   %F: @HasF.%F.type (%F.type.46c) = struct_value () [symbolic = %F (constants.%F.823)]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF.%HasF.type (%HasF.type.901) [symbolic = %HasF.assoc_type (constants.%HasF.assoc_type.595)]
// CHECK:STDOUT:   %assoc0.loc5_14.2: @HasF.%HasF.assoc_type (%HasF.assoc_type.595) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_14.2 (constants.%assoc0.35e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @HasF.%HasF.type (%HasF.type.901) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.322)]
// CHECK:STDOUT:     %F.decl: @HasF.%F.type (%F.type.46c) = fn_decl @F.1 [symbolic = @HasF.%F (constants.%F.823)] {
// CHECK:STDOUT:       %return.patt: @F.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.1.%T (%T) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @HasF.%T.loc4_16.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @F.1.%T (%T) = out_param call_param0
// CHECK:STDOUT:       %return: ref @F.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_14.1: @HasF.%HasF.assoc_type (%HasF.assoc_type.595) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_14.2 (constants.%assoc0.35e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc5_14.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %HasF.type {
// CHECK:STDOUT:   %F.decl: %F.type.94c = fn_decl @F.2 [concrete = constants.%F.901] {
// CHECK:STDOUT:     %return.patt: %Param = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Param = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [concrete = constants.%Param]
// CHECK:STDOUT:     %return.param: ref %Param = out_param call_param0
// CHECK:STDOUT:     %return: ref %Param = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Param = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @C.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param {
// CHECK:STDOUT:   %.loc9_8: %Param.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc9_3: %Param.elem = var_pattern %.loc9_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %Param.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.ed6 [concrete = constants.%complete_type.1ec]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Param
// CHECK:STDOUT:   .x = %.loc9_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.fe3 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [concrete = constants.%Param]
// CHECK:STDOUT:     %HasF.type: type = facet_type <@HasF, @HasF(constants.%Param)> [concrete = constants.%HasF.type.b18]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .HasF = <poisoned>
// CHECK:STDOUT:   .Param = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend @impl.%HasF.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%T.loc4_16.1: type, @HasF.%Self.1: @HasF.%HasF.type (%HasF.type.901)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @F.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %return.param_patt: %Param {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc32_21.1: %struct_type.x.c96 = struct_literal (%int_2)
// CHECK:STDOUT:   %.loc32_21.2: %i32 = converted %int_2, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref.loc48: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref.loc48: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:   %x.ref.loc48: <error> = name_ref x, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc48: type = splice_block %i32.loc48 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc48: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc48: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %i32 = bind_name a, <error>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %.loc59_3: %i32 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc59: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:   %x.ref.loc59: <error> = name_ref x, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %.loc59_10: type = splice_block %i32.loc59 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc59: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc59: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self.322) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(%T.loc4_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%Param) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%Param
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%Param
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.b18
// CHECK:STDOUT:   %Self.2 => constants.%Self.322
// CHECK:STDOUT:   %F.type => constants.%F.type.7f1
// CHECK:STDOUT:   %F => constants.%F.eff
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.dc4
// CHECK:STDOUT:   %assoc0.loc5_14.2 => constants.%assoc0.a6b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Param, constants.%HasF.facet) {
// CHECK:STDOUT:   %T => constants.%Param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_impl_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325e65.1: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.325e65.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type.2aef59.1: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.bb2dd4.1: %F.type.2aef59.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.955255.1: type = assoc_entity_type %I.type.325e65.1 [symbolic]
// CHECK:STDOUT:   %assoc0.fef501.1: %I.assoc_type.955255.1 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %X.type: type = generic_class_type @X [concrete]
// CHECK:STDOUT:   %X.generic: %X.type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic]
// CHECK:STDOUT:   %I.type.325e65.2: type = facet_type <@I, @I(%U)> [symbolic]
// CHECK:STDOUT:   %F.type.2aef59.2: type = fn_type @F.1, @I(%U) [symbolic]
// CHECK:STDOUT:   %F.bb2dd4.2: %F.type.2aef59.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.955255.2: type = assoc_entity_type %I.type.325e65.2 [symbolic]
// CHECK:STDOUT:   %assoc0.fef501.2: %I.assoc_type.955255.2 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.cfe: <witness> = require_complete_type %I.type.325e65.2 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(%U) [symbolic]
// CHECK:STDOUT:   %F.type.e88: type = fn_type @F.2, @impl(%U) [symbolic]
// CHECK:STDOUT:   %F.b02: %F.type.e88 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.325e65.2 = facet_value %X, (%impl_witness) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.441: <witness> = require_complete_type %X [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %U [symbolic]
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
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: %X.type = class_decl @X [concrete = constants.%X.generic] {
// CHECK:STDOUT:     %U.patt.loc8_9.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_9.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc8_9.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc8_9.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_13.2)> [symbolic = %I.type (constants.%I.type.325e65.1)]
// CHECK:STDOUT:   %Self.2: %I.type.325e65.1 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T.loc4_13.2) [symbolic = %F.type (constants.%F.type.2aef59.1)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.2aef59.1) = struct_value () [symbolic = %F (constants.%F.bb2dd4.1)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.325e65.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.955255.1)]
// CHECK:STDOUT:   %assoc0.loc5_25.2: @I.%I.assoc_type (%I.assoc_type.955255.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_25.2 (constants.%assoc0.fef501.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.325e65.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type.2aef59.1) = fn_decl @F.1 [symbolic = @I.%F (constants.%F.bb2dd4.1)] {
// CHECK:STDOUT:       %self.patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %t.patt: @F.1.%T (%T) = binding_pattern t
// CHECK:STDOUT:       %t.param_patt: @F.1.%T (%T) = value_param_pattern %t.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_14.1: type = splice_block %.loc5_14.3 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_14.2: @F.1.%I.type (%I.type.325e65.1) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @F.1.%I.type (%I.type.325e65.1) = name_ref Self, %.loc5_14.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_14.3: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.1.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @I.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @F.1.%T (%T) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_25.1: @I.%I.assoc_type (%I.assoc_type.955255.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_25.2 (constants.%assoc0.fef501.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc5_25.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(@X.%U.loc8_9.1: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %I.type.loc9_21.2: type = facet_type <@I, @I(%U)> [symbolic = %I.type.loc9_21.2 (constants.%I.type.325e65.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%I.type.loc9_21.2 (%I.type.325e65.2) [symbolic = %require_complete (constants.%require_complete.cfe)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%F.decl), @impl(%U) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%U) [symbolic = %F.type (constants.%F.type.e88)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.e88) = struct_value () [symbolic = %F (constants.%F.b02)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %I.type.loc9_21.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.e88) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.b02)] {
// CHECK:STDOUT:       %self.patt: @F.2.%X (%X) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.2.%X (%X) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %t.patt: @F.2.%U (%U) = binding_pattern t
// CHECK:STDOUT:       %t.param_patt: @F.2.%U (%U) = value_param_pattern %t.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.2.%X (%X) = value_param call_param0
// CHECK:STDOUT:       %.loc20_16.1: type = splice_block %Self.ref [symbolic = %X (constants.%X)] {
// CHECK:STDOUT:         %.loc20_16.2: type = specific_constant constants.%X, @X(constants.%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc20_16.2 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.2.%X (%X) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.2.%U (%U) = value_param call_param1
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc8_9.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %t: @F.2.%U (%U) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = @X.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @X(%U.loc8_9.1: type) {
// CHECK:STDOUT:   %U.loc8_9.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc8_9.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc8_9.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_9.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U.loc8_9.2)> [symbolic = %I.type (constants.%I.type.325e65.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%X [symbolic = %X (constants.%X)]
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc8_9.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %I.type.loc9_21.1: type = facet_type <@I, @I(constants.%U)> [symbolic = %I.type.loc9_21.2 (constants.%I.type.325e65.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(constants.%U) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:     %.loc9: type = specific_constant @impl.%I.type.loc9_21.1, @impl(constants.%U) [symbolic = %I.type (constants.%I.type.325e65.2)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%X
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     extend %.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%T.loc4_13.1: type, @I.%Self.1: @I.%I.type (%I.type.325e65.1)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325e65.1)]
// CHECK:STDOUT:   %Self: %I.type.325e65.1 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type)](%t.param_patt: @F.1.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@X.%U.loc8_9.1: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_14: <witness> = require_complete_type @F.2.%X (%X) [symbolic = %require_complete.loc20_14 (constants.%require_complete.441)]
// CHECK:STDOUT:   %require_complete.loc20_23: <witness> = require_complete_type @F.2.%U (%U) [symbolic = %require_complete.loc20_23 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.2.%X (%X)](%t.param_patt: @F.2.%U (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.1
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T.loc4_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%U) {
// CHECK:STDOUT:   %U.loc8_9.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc8_9.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%U) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%U
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.2aef59.2
// CHECK:STDOUT:   %F => constants.%F.bb2dd4.2
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.955255.2
// CHECK:STDOUT:   %assoc0.loc5_25.2 => constants.%assoc0.fef501.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %I.type.loc9_21.2 => constants.%I.type.325e65.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cfe
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.e88
// CHECK:STDOUT:   %F => constants.%F.b02
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(@impl.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(@F.2.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%U, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%U
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@X.%U.loc8_9.2) {}
// CHECK:STDOUT:
