// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/member_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/member_lookup.carbon
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

// --- member_access.carbon

library "[[@TEST_NAME]]";

interface Interface(T:! type) {
  let X:! T;
}

fn AccessGeneric[T:! type](I:! Interface(T)) -> T {
  // CHECK:STDERR: member_access.carbon:[[@LINE+7]]:10: error: member access into undefined facet type `Interface(T)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return I.X;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: member_access.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Interface(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return I.X;
}

fn AccessConcrete(I:! Interface(i32)) -> i32 {
  // CHECK:STDERR: member_access.carbon:[[@LINE+7]]:10: error: member access into undefined facet type `Interface(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return I.X;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: member_access.carbon:[[@LINE-19]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Interface(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return I.X;
}

// --- fail_no_member.carbon

library "[[@TEST_NAME]]";

interface Interface(T:! type) {
  let X:! T;
}

fn AccessMissingGeneric[T:! type](I:! Interface(T)) -> T {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+7]]:10: error: member access into undefined facet type `Interface(T)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return I.nonesuch;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Interface(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return I.nonesuch;
}

fn AccessMissingConcrete(I:! Interface(i32)) -> i32 {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+7]]:10: error: member access into undefined facet type `Interface(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return I.nonesuch;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE-19]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Interface(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return I.nonesuch;
}

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.e32: type = generic_interface_type @Interface [concrete]
// CHECK:STDOUT:   %Interface.generic: %Interface.type.e32 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.type.d32: type = facet_type <@Interface, @Interface(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Interface.type.d32 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.3bd: type = assoc_entity_type %Interface.type.d32 [symbolic]
// CHECK:STDOUT:   %assoc0.ed7: %Interface.assoc_type.3bd = assoc_entity element0, @Interface.%X [symbolic]
// CHECK:STDOUT:   %I.a5f: %Interface.type.d32 = bind_symbolic_name I, 1 [symbolic]
// CHECK:STDOUT:   %I.patt.47c: %Interface.type.d32 = symbolic_binding_pattern I, 1 [symbolic]
// CHECK:STDOUT:   %AccessGeneric.type: type = fn_type @AccessGeneric [concrete]
// CHECK:STDOUT:   %AccessGeneric: %AccessGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6f3: <witness> = require_complete_type %Interface.type.d32 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Interface.type.981: type = facet_type <@Interface, @Interface(%i32)> [concrete]
// CHECK:STDOUT:   %I.d08: %Interface.type.981 = bind_symbolic_name I, 0 [symbolic]
// CHECK:STDOUT:   %I.patt.1f8: %Interface.type.981 = symbolic_binding_pattern I, 0 [symbolic]
// CHECK:STDOUT:   %AccessConcrete.type: type = fn_type @AccessConcrete [concrete]
// CHECK:STDOUT:   %AccessConcrete: %AccessConcrete.type = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.assoc_type.44d: type = assoc_entity_type %Interface.type.981 [concrete]
// CHECK:STDOUT:   %assoc0.4ff: %Interface.assoc_type.44d = assoc_entity element0, @Interface.%X [concrete]
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
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .AccessGeneric = %AccessGeneric.decl
// CHECK:STDOUT:     .AccessConcrete = %AccessConcrete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: %Interface.type.e32 = interface_decl @Interface [concrete = constants.%Interface.generic] {
// CHECK:STDOUT:     %T.patt.loc4_21.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessGeneric.decl: %AccessGeneric.type = fn_decl @AccessGeneric [concrete = constants.%AccessGeneric] {
// CHECK:STDOUT:     %T.patt.loc8_18.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_18.2 (constants.%T.patt)]
// CHECK:STDOUT:     %I.patt.loc8_28.1: @AccessGeneric.%Interface.type.loc8_43.2 (%Interface.type.d32) = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_28.2 (constants.%I.patt.47c)]
// CHECK:STDOUT:     %return.patt: @AccessGeneric.%T.loc8_18.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @AccessGeneric.%T.loc8_18.2 (%T) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_49: type = name_ref T, %T.loc8_18.1 [symbolic = %T.loc8_18.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc8_18.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_18.2 (constants.%T)]
// CHECK:STDOUT:     %.loc8: type = splice_block %Interface.type.loc8_43.1 [symbolic = %Interface.type.loc8_43.2 (constants.%Interface.type.d32)] {
// CHECK:STDOUT:       %Interface.ref: %Interface.type.e32 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %T.ref.loc8_42: type = name_ref T, %T.loc8_18.1 [symbolic = %T.loc8_18.2 (constants.%T)]
// CHECK:STDOUT:       %Interface.type.loc8_43.1: type = facet_type <@Interface, @Interface(constants.%T)> [symbolic = %Interface.type.loc8_43.2 (constants.%Interface.type.d32)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc8_28.1: @AccessGeneric.%Interface.type.loc8_43.2 (%Interface.type.d32) = bind_symbolic_name I, 1 [symbolic = %I.loc8_28.2 (constants.%I.a5f)]
// CHECK:STDOUT:     %return.param: ref @AccessGeneric.%T.loc8_18.2 (%T) = out_param call_param0
// CHECK:STDOUT:     %return: ref @AccessGeneric.%T.loc8_18.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessConcrete.decl: %AccessConcrete.type = fn_decl @AccessConcrete [concrete = constants.%AccessConcrete] {
// CHECK:STDOUT:     %I.patt.loc19_19.1: %Interface.type.981 = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc19_19.2 (constants.%I.patt.1f8)]
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc19_42: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc19_42: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc19: type = splice_block %Interface.type [concrete = constants.%Interface.type.981] {
// CHECK:STDOUT:       %Interface.ref: %Interface.type.e32 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %int_32.loc19_33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc19_33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Interface.type: type = facet_type <@Interface, @Interface(constants.%i32)> [concrete = constants.%Interface.type.981]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc19_19.1: %Interface.type.981 = bind_symbolic_name I, 0 [symbolic = %I.loc19_19.2 (constants.%I.d08)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Interface(%T.loc4_21.1: type) {
// CHECK:STDOUT:   %T.loc4_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_21.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_21.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface, @Interface(%T.loc4_21.2)> [symbolic = %Interface.type (constants.%Interface.type.d32)]
// CHECK:STDOUT:   %Self.2: %Interface.type.d32 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface.%Interface.type (%Interface.type.d32) [symbolic = %Interface.assoc_type (constants.%Interface.assoc_type.3bd)]
// CHECK:STDOUT:   %assoc0: @Interface.%Interface.assoc_type (%Interface.assoc_type.3bd) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.ed7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Interface.%Interface.type (%Interface.type.d32) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %X: @X.%T (%T) = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Interface.%Interface.assoc_type (%Interface.assoc_type.3bd) = assoc_entity element0, @Interface.%X [symbolic = @Interface.%assoc0 (constants.%assoc0.ed7)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (%X)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Interface.%T.loc4_21.1: type, @Interface.%Self.1: @Interface.%Interface.type (%Interface.type.d32)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @X.%T (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const X:! @X.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessGeneric(%T.loc8_18.1: type, %I.loc8_28.1: @AccessGeneric.%Interface.type.loc8_43.2 (%Interface.type.d32)) {
// CHECK:STDOUT:   %T.loc8_18.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_18.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_18.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Interface.type.loc8_43.2: type = facet_type <@Interface, @Interface(%T.loc8_18.2)> [symbolic = %Interface.type.loc8_43.2 (constants.%Interface.type.d32)]
// CHECK:STDOUT:   %I.loc8_28.2: %Interface.type.d32 = bind_symbolic_name I, 1 [symbolic = %I.loc8_28.2 (constants.%I.a5f)]
// CHECK:STDOUT:   %I.patt.loc8_28.2: %Interface.type.d32 = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_28.2 (constants.%I.patt.47c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @AccessGeneric.%Interface.type.loc8_43.2 (%Interface.type.d32) [symbolic = %require_complete (constants.%require_complete.6f3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc8_18.1: type](%I.patt.loc8_28.1: @AccessGeneric.%Interface.type.loc8_43.2 (%Interface.type.d32)) -> @AccessGeneric.%T.loc8_18.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: @AccessGeneric.%Interface.type.loc8_43.2 (%Interface.type.d32) = name_ref I, %I.loc8_28.1 [symbolic = %I.loc8_28.2 (constants.%I.a5f)]
// CHECK:STDOUT:     %X.ref: <error> = name_ref X, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessConcrete(%I.loc19_19.1: %Interface.type.981) {
// CHECK:STDOUT:   %I.loc19_19.2: %Interface.type.981 = bind_symbolic_name I, 0 [symbolic = %I.loc19_19.2 (constants.%I.d08)]
// CHECK:STDOUT:   %I.patt.loc19_19.2: %Interface.type.981 = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc19_19.2 (constants.%I.patt.1f8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%I.patt.loc19_19.1: %Interface.type.981) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %Interface.type.981 = name_ref I, %I.loc19_19.1 [symbolic = %I.loc19_19.2 (constants.%I.d08)]
// CHECK:STDOUT:     %X.ref: <error> = name_ref X, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%T) {
// CHECK:STDOUT:   %T.loc4_21.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_21.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.d32
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.3bd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ed7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(%T.loc4_21.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessGeneric(constants.%T, constants.%I.a5f) {
// CHECK:STDOUT:   %T.loc8_18.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_18.2 => constants.%T
// CHECK:STDOUT:   %Interface.type.loc8_43.2 => constants.%Interface.type.d32
// CHECK:STDOUT:   %I.loc8_28.2 => constants.%I.a5f
// CHECK:STDOUT:   %I.patt.loc8_28.2 => constants.%I.a5f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(@AccessGeneric.%T.loc8_18.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_21.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_21.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.981
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.44d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4ff
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessConcrete(constants.%I.d08) {
// CHECK:STDOUT:   %I.loc19_19.2 => constants.%I.d08
// CHECK:STDOUT:   %I.patt.loc19_19.2 => constants.%I.d08
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.e32: type = generic_interface_type @Interface [concrete]
// CHECK:STDOUT:   %Interface.generic: %Interface.type.e32 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.type.d32: type = facet_type <@Interface, @Interface(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Interface.type.d32 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type.3bd: type = assoc_entity_type %Interface.type.d32 [symbolic]
// CHECK:STDOUT:   %assoc0.ed7: %Interface.assoc_type.3bd = assoc_entity element0, @Interface.%X [symbolic]
// CHECK:STDOUT:   %I.a5f: %Interface.type.d32 = bind_symbolic_name I, 1 [symbolic]
// CHECK:STDOUT:   %I.patt.47c: %Interface.type.d32 = symbolic_binding_pattern I, 1 [symbolic]
// CHECK:STDOUT:   %AccessMissingGeneric.type: type = fn_type @AccessMissingGeneric [concrete]
// CHECK:STDOUT:   %AccessMissingGeneric: %AccessMissingGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6f3: <witness> = require_complete_type %Interface.type.d32 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Interface.type.981: type = facet_type <@Interface, @Interface(%i32)> [concrete]
// CHECK:STDOUT:   %I.d08: %Interface.type.981 = bind_symbolic_name I, 0 [symbolic]
// CHECK:STDOUT:   %I.patt.1f8: %Interface.type.981 = symbolic_binding_pattern I, 0 [symbolic]
// CHECK:STDOUT:   %AccessMissingConcrete.type: type = fn_type @AccessMissingConcrete [concrete]
// CHECK:STDOUT:   %AccessMissingConcrete: %AccessMissingConcrete.type = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.assoc_type.44d: type = assoc_entity_type %Interface.type.981 [concrete]
// CHECK:STDOUT:   %assoc0.4ff: %Interface.assoc_type.44d = assoc_entity element0, @Interface.%X [concrete]
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
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .AccessMissingGeneric = %AccessMissingGeneric.decl
// CHECK:STDOUT:     .AccessMissingConcrete = %AccessMissingConcrete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: %Interface.type.e32 = interface_decl @Interface [concrete = constants.%Interface.generic] {
// CHECK:STDOUT:     %T.patt.loc4_21.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingGeneric.decl: %AccessMissingGeneric.type = fn_decl @AccessMissingGeneric [concrete = constants.%AccessMissingGeneric] {
// CHECK:STDOUT:     %T.patt.loc8_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %I.patt.loc8_35.1: @AccessMissingGeneric.%Interface.type.loc8_50.2 (%Interface.type.d32) = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_35.2 (constants.%I.patt.47c)]
// CHECK:STDOUT:     %return.patt: @AccessMissingGeneric.%T.loc8_25.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @AccessMissingGeneric.%T.loc8_25.2 (%T) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_56: type = name_ref T, %T.loc8_25.1 [symbolic = %T.loc8_25.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc8_25.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_25.2 (constants.%T)]
// CHECK:STDOUT:     %.loc8: type = splice_block %Interface.type.loc8_50.1 [symbolic = %Interface.type.loc8_50.2 (constants.%Interface.type.d32)] {
// CHECK:STDOUT:       %Interface.ref: %Interface.type.e32 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %T.ref.loc8_49: type = name_ref T, %T.loc8_25.1 [symbolic = %T.loc8_25.2 (constants.%T)]
// CHECK:STDOUT:       %Interface.type.loc8_50.1: type = facet_type <@Interface, @Interface(constants.%T)> [symbolic = %Interface.type.loc8_50.2 (constants.%Interface.type.d32)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc8_35.1: @AccessMissingGeneric.%Interface.type.loc8_50.2 (%Interface.type.d32) = bind_symbolic_name I, 1 [symbolic = %I.loc8_35.2 (constants.%I.a5f)]
// CHECK:STDOUT:     %return.param: ref @AccessMissingGeneric.%T.loc8_25.2 (%T) = out_param call_param0
// CHECK:STDOUT:     %return: ref @AccessMissingGeneric.%T.loc8_25.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingConcrete.decl: %AccessMissingConcrete.type = fn_decl @AccessMissingConcrete [concrete = constants.%AccessMissingConcrete] {
// CHECK:STDOUT:     %I.patt.loc19_26.1: %Interface.type.981 = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc19_26.2 (constants.%I.patt.1f8)]
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc19_49: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc19_49: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc19: type = splice_block %Interface.type [concrete = constants.%Interface.type.981] {
// CHECK:STDOUT:       %Interface.ref: %Interface.type.e32 = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.generic]
// CHECK:STDOUT:       %int_32.loc19_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc19_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Interface.type: type = facet_type <@Interface, @Interface(constants.%i32)> [concrete = constants.%Interface.type.981]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc19_26.1: %Interface.type.981 = bind_symbolic_name I, 0 [symbolic = %I.loc19_26.2 (constants.%I.d08)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Interface(%T.loc4_21.1: type) {
// CHECK:STDOUT:   %T.loc4_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_21.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_21.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface, @Interface(%T.loc4_21.2)> [symbolic = %Interface.type (constants.%Interface.type.d32)]
// CHECK:STDOUT:   %Self.2: %Interface.type.d32 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface.%Interface.type (%Interface.type.d32) [symbolic = %Interface.assoc_type (constants.%Interface.assoc_type.3bd)]
// CHECK:STDOUT:   %assoc0: @Interface.%Interface.assoc_type (%Interface.assoc_type.3bd) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.ed7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Interface.%Interface.type (%Interface.type.d32) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %X: @X.%T (%T) = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Interface.%Interface.assoc_type (%Interface.assoc_type.3bd) = assoc_entity element0, @Interface.%X [symbolic = @Interface.%assoc0 (constants.%assoc0.ed7)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (%X)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Interface.%T.loc4_21.1: type, @Interface.%Self.1: @Interface.%Interface.type (%Interface.type.d32)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @X.%T (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const X:! @X.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingGeneric(%T.loc8_25.1: type, %I.loc8_35.1: @AccessMissingGeneric.%Interface.type.loc8_50.2 (%Interface.type.d32)) {
// CHECK:STDOUT:   %T.loc8_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Interface.type.loc8_50.2: type = facet_type <@Interface, @Interface(%T.loc8_25.2)> [symbolic = %Interface.type.loc8_50.2 (constants.%Interface.type.d32)]
// CHECK:STDOUT:   %I.loc8_35.2: %Interface.type.d32 = bind_symbolic_name I, 1 [symbolic = %I.loc8_35.2 (constants.%I.a5f)]
// CHECK:STDOUT:   %I.patt.loc8_35.2: %Interface.type.d32 = symbolic_binding_pattern I, 1 [symbolic = %I.patt.loc8_35.2 (constants.%I.patt.47c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @AccessMissingGeneric.%Interface.type.loc8_50.2 (%Interface.type.d32) [symbolic = %require_complete (constants.%require_complete.6f3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc8_25.1: type](%I.patt.loc8_35.1: @AccessMissingGeneric.%Interface.type.loc8_50.2 (%Interface.type.d32)) -> @AccessMissingGeneric.%T.loc8_25.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: @AccessMissingGeneric.%Interface.type.loc8_50.2 (%Interface.type.d32) = name_ref I, %I.loc8_35.1 [symbolic = %I.loc8_35.2 (constants.%I.a5f)]
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingConcrete(%I.loc19_26.1: %Interface.type.981) {
// CHECK:STDOUT:   %I.loc19_26.2: %Interface.type.981 = bind_symbolic_name I, 0 [symbolic = %I.loc19_26.2 (constants.%I.d08)]
// CHECK:STDOUT:   %I.patt.loc19_26.2: %Interface.type.981 = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc19_26.2 (constants.%I.patt.1f8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%I.patt.loc19_26.1: %Interface.type.981) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %Interface.type.981 = name_ref I, %I.loc19_26.1 [symbolic = %I.loc19_26.2 (constants.%I.d08)]
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%T) {
// CHECK:STDOUT:   %T.loc4_21.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_21.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.d32
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.3bd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ed7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(%T.loc4_21.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingGeneric(constants.%T, constants.%I.a5f) {
// CHECK:STDOUT:   %T.loc8_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_25.2 => constants.%T
// CHECK:STDOUT:   %Interface.type.loc8_50.2 => constants.%Interface.type.d32
// CHECK:STDOUT:   %I.loc8_35.2 => constants.%I.a5f
// CHECK:STDOUT:   %I.patt.loc8_35.2 => constants.%I.a5f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(@AccessMissingGeneric.%T.loc8_25.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_21.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_21.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Interface.type => constants.%Interface.type.981
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Interface.assoc_type => constants.%Interface.assoc_type.44d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4ff
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingConcrete(constants.%I.d08) {
// CHECK:STDOUT:   %I.loc19_26.2 => constants.%I.d08
// CHECK:STDOUT:   %I.patt.loc19_26.2 => constants.%I.d08
// CHECK:STDOUT: }
// CHECK:STDOUT:
