// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/array.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/array.carbon
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

// --- type_only.carbon

library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: type_only.carbon:[[@LINE+7]]:48: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn F[T:! type](a: array(T, 3)) -> T { return a[0]; }
// CHECK:STDERR:                                                ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F[T:! type](a: array(T, 3)) -> T { return a[0]; }

fn G() -> C {
  var a: array(C, 3) = ({}, {}, {});
  return F(a);
}

// --- bound_only.carbon

library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: bound_only.carbon:[[@LINE+7]]:54: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn F[N:! Core.IntLiteral()](a: array(C, N)) -> i32 { return N; }
// CHECK:STDERR:                                                      ^~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F[N:! Core.IntLiteral()](a: array(C, N)) -> i32 { return N; }

fn G() -> i32 {
  var a: array(C, 3) = ({}, {}, {});
  return F(a);
}

// --- type_and_bound.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type, N:! Core.IntLiteral()](a: array(T, N)) {}

fn G() {
  var a: array(C, 3) = ({}, {}, {});
  F(a);
}

// --- fail_bound_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE+7]]:48: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn F[T:! type](a: array(T, 2)) -> T { return a[0]; }
// CHECK:STDERR:                                                ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F[T:! type](a: array(T, 2)) -> T { return a[0]; }

fn G() -> C {
  // TODO: We succeed at deducing T here but fail to convert. Is this the right behavior?
  var a: array(C, 3) = ({}, {}, {});
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE+10]]:12: error: member access into undefined facet type `Core.ImplicitAs([C; 2])` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE-11]]:16: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F[T:! type](a: array(T, 2)) -> T { return a[0]; }
  // CHECK:STDERR:                ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_type_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:54: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn F[N:! Core.IntLiteral()](a: array(C, N)) -> i32 { return N; }
// CHECK:STDERR:                                                      ^~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F[N:! Core.IntLiteral()](a: array(C, N)) -> i32 { return N; }

fn G() -> i32 {
  // TODO: We succeed at deducing N here but fail to convert. Is this the right behavior?
  var a: array(D, 3) = ({}, {}, {});
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+10]]:12: error: member access into undefined facet type `Core.ImplicitAs([C; 3])` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE-11]]:29: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F[N:! Core.IntLiteral()](a: array(C, N)) -> i32 { return N; }
  // CHECK:STDERR:                             ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_bound_type_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: fail_bound_type_mismatch.carbon:[[@LINE+7]]:27: error: member access into undefined facet type `Core.ImplicitAs(Core.IntLiteral)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn F[N:! i32](a: array(C, N)) -> i32 { return N; }
// CHECK:STDERR:                           ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F[N:! i32](a: array(C, N)) -> i32 { return N; }

fn G() -> i32 {
  var a: array(C, 3) = ({}, {}, {});
  // TODO: This fails because the array bound in `F` is effectively
  //   `N.(ImplicitAs(IntLiteral).Convert)()`
  // which we can't deduce through. We should decide if we want to support
  // deductions of that form. If not, it'd be nice to diagnose this situation
  // better.
  return F(a);
}

// CHECK:STDOUT: --- type_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.743: type = array_type %int_3, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.06f: <witness> = require_complete_type %array_type.743 [symbolic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.002 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %complete_type.dd1: <witness> = complete_type_witness %array_type.002 [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc13_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc13_29.2 (%array_type.743) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc13_29.2 (%array_type.743) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc13_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc13_6.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13_35: type = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc13_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc13_29.2 (%array_type.743) = value_param call_param0
// CHECK:STDOUT:     %.loc13_29: type = splice_block %array_type.loc13_29.1 [symbolic = %array_type.loc13_29.2 (constants.%array_type.743)] {
// CHECK:STDOUT:       %T.ref.loc13_25: type = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:       %array_type.loc13_29.1: type = array_type %int_3, %T [symbolic = %array_type.loc13_29.2 (constants.%array_type.743)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc13_29.2 (%array_type.743) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc13_6.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc13_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_6.1: type) {
// CHECK:STDOUT:   %T.loc13_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %array_type.loc13_29.2: type = array_type constants.%int_3, @F.%T.loc13_6.2 (%T) [symbolic = %array_type.loc13_29.2 (constants.%array_type.743)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_32: <witness> = require_complete_type @F.%T.loc13_6.2 (%T) [symbolic = %require_complete.loc13_32 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc13_17: <witness> = require_complete_type @F.%array_type.loc13_29.2 (%array_type.743) [symbolic = %require_complete.loc13_17 (constants.%require_complete.06f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc13_6.1: type](%a.param_patt: @F.%array_type.loc13_29.2 (%array_type.743)) -> @F.%T.loc13_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @F.%array_type.loc13_29.2 (%array_type.743) = name_ref a, %a
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13_48: %i32 = converted %int_0, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc13_49.1: ref @F.%array_type.loc13_29.2 (%array_type.743) = value_as_ref %a.ref
// CHECK:STDOUT:     %.loc13_49.2: ref @F.%T.loc13_6.2 (%T) = array_index %.loc13_49.1, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc13_49.3: @F.%T.loc13_6.2 (%T) = bind_value %.loc13_49.2 [concrete = <error>]
// CHECK:STDOUT:     return %.loc13_49.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type.002 = binding_pattern a
// CHECK:STDOUT:     %.loc16_3.1: %array_type.002 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.002 = var a
// CHECK:STDOUT:   %.loc16_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc16_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc16_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc16_35.1: %tuple.type = tuple_literal (%.loc16_26.1, %.loc16_30.1, %.loc16_34.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc16_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc16_26.2: init %C = class_init (), %.loc16_35.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc16_35.3: init %C = converted %.loc16_26.1, %.loc16_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc16_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc16_30.2: init %C = class_init (), %.loc16_35.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc16_35.5: init %C = converted %.loc16_30.1, %.loc16_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc16_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc16_34.2: init %C = class_init (), %.loc16_35.6 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc16_35.7: init %C = converted %.loc16_34.1, %.loc16_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc16_35.8: init %array_type.002 = array_init (%.loc16_35.3, %.loc16_35.5, %.loc16_35.7) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc16_3.2: init %array_type.002 = converted %.loc16_35.1, %.loc16_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc16_3.2
// CHECK:STDOUT:   %.loc16_20: type = splice_block %array_type [concrete = constants.%array_type.002] {
// CHECK:STDOUT:     %C.ref.loc16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C [concrete = constants.%array_type.002]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.002 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.002 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc15: ref %C = splice_block %return {}
// CHECK:STDOUT:   %.loc17: %array_type.002 = bind_value %a.ref
// CHECK:STDOUT:   %F.call: init %C = call %F.specific_fn(%.loc17) to %.loc15
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_6.2 => constants.%T
// CHECK:STDOUT:   %array_type.loc13_29.2 => constants.%array_type.743
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc13_6.2 => constants.%C
// CHECK:STDOUT:   %array_type.loc13_29.2 => constants.%array_type.002
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_32 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc13_17 => constants.%complete_type.dd1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bound_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %array_type.6a2: type = array_type %N, %C [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d82: <witness> = require_complete_type %array_type.6a2 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.002 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_3) [concrete]
// CHECK:STDOUT:   %complete_type.dd1: <witness> = complete_type_witness %array_type.002 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc13_6.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc13_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc13_42.2 (%array_type.6a2) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc13_42.2 (%array_type.6a2) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13_26.1: type = splice_block %.loc13_26.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc13_26.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc13_26.3: type = converted %int_literal.make_type, %.loc13_26.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc13_6.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc13_6.2 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc13_42.2 (%array_type.6a2) = value_param call_param0
// CHECK:STDOUT:     %.loc13_42: type = splice_block %array_type.loc13_42.1 [symbolic = %array_type.loc13_42.2 (constants.%array_type.6a2)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc13_41: Core.IntLiteral = name_ref N, %N.loc13_6.1 [symbolic = %N.loc13_6.2 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc13_42.1: type = array_type %N.ref.loc13_41, %C [symbolic = %array_type.loc13_42.2 (constants.%array_type.6a2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc13_42.2 (%array_type.6a2) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
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
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc13_6.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc13_6.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc13_6.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc13_6.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc13_6.2 (constants.%N.patt)]
// CHECK:STDOUT:   %array_type.loc13_42.2: type = array_type %N.loc13_6.2, %C [symbolic = %array_type.loc13_42.2 (constants.%array_type.6a2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%array_type.loc13_42.2 (%array_type.6a2) [symbolic = %require_complete (constants.%require_complete.d82)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.patt.loc13_6.1: Core.IntLiteral](%a.param_patt: @F.%array_type.loc13_42.2 (%array_type.6a2)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc13_61: Core.IntLiteral = name_ref N, %N.loc13_6.1 [symbolic = %N.loc13_6.2 (constants.%N)]
// CHECK:STDOUT:     %.loc13_62: %i32 = converted %N.ref.loc13_61, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type.002 = binding_pattern a
// CHECK:STDOUT:     %.loc16_3.1: %array_type.002 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.002 = var a
// CHECK:STDOUT:   %.loc16_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc16_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc16_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc16_35.1: %tuple.type = tuple_literal (%.loc16_26.1, %.loc16_30.1, %.loc16_34.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc16_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc16_26.2: init %C = class_init (), %.loc16_35.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc16_35.3: init %C = converted %.loc16_26.1, %.loc16_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc16_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc16_30.2: init %C = class_init (), %.loc16_35.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc16_35.5: init %C = converted %.loc16_30.1, %.loc16_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc16_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc16_34.2: init %C = class_init (), %.loc16_35.6 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc16_35.7: init %C = converted %.loc16_34.1, %.loc16_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc16_35.8: init %array_type.002 = array_init (%.loc16_35.3, %.loc16_35.5, %.loc16_35.7) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc16_3.2: init %array_type.002 = converted %.loc16_35.1, %.loc16_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc16_3.2
// CHECK:STDOUT:   %.loc16_20: type = splice_block %array_type [concrete = constants.%array_type.002] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C [concrete = constants.%array_type.002]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.002 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.002 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_3) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc17_12: %array_type.002 = bind_value %a.ref
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%.loc17_12)
// CHECK:STDOUT:   %.loc17_14.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc17_14.2: %i32 = converted %F.call, %.loc17_14.1
// CHECK:STDOUT:   return %.loc17_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc13_6.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc13_6.2 => constants.%N
// CHECK:STDOUT:   %array_type.loc13_42.2 => constants.%array_type.6a2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_3) {
// CHECK:STDOUT:   %N.loc13_6.2 => constants.%int_3
// CHECK:STDOUT:   %N.patt.loc13_6.2 => constants.%int_3
// CHECK:STDOUT:   %array_type.loc13_42.2 => constants.%array_type.002
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.dd1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- type_and_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %array_type.bb5: type = array_type %N, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.bb5 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.002 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%C, %int_3) [concrete]
// CHECK:STDOUT:   %complete_type.dd1: <witness> = complete_type_witness %array_type.002 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc6_16.1: Core.IntLiteral = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc6_16.2 (constants.%N.patt)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc6_52.2 (%array_type.bb5) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc6_52.2 (%array_type.bb5) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc6_36.1: type = splice_block %.loc6_36.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_36.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_36.3: type = converted %int_literal.make_type, %.loc6_36.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_16.1: Core.IntLiteral = bind_symbolic_name N, 1 [symbolic = %N.loc6_16.2 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_52.2 (%array_type.bb5) = value_param call_param0
// CHECK:STDOUT:     %.loc6_52: type = splice_block %array_type.loc6_52.1 [symbolic = %array_type.loc6_52.2 (constants.%array_type.bb5)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, %N.loc6_16.1 [symbolic = %N.loc6_16.2 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc6_52.1: type = array_type %N.ref, %T [symbolic = %array_type.loc6_52.2 (constants.%array_type.bb5)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_52.2 (%array_type.bb5) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type, %N.loc6_16.1: Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc6_16.2: Core.IntLiteral = bind_symbolic_name N, 1 [symbolic = %N.loc6_16.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_16.2: Core.IntLiteral = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc6_16.2 (constants.%N.patt)]
// CHECK:STDOUT:   %array_type.loc6_52.2: type = array_type %N.loc6_16.2, @F.%T.loc6_6.2 (%T) [symbolic = %array_type.loc6_52.2 (constants.%array_type.bb5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%array_type.loc6_52.2 (%array_type.bb5) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc6_6.1: type, %N.patt.loc6_16.1: Core.IntLiteral](%a.param_patt: @F.%array_type.loc6_52.2 (%array_type.bb5)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type.002 = binding_pattern a
// CHECK:STDOUT:     %.loc9_3.1: %array_type.002 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.002 = var a
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_26.2: init %C = class_init (), %.loc9_35.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.3: init %C = converted %.loc9_26.1, %.loc9_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_30.2: init %C = class_init (), %.loc9_35.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.5: init %C = converted %.loc9_30.1, %.loc9_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_34.2: init %C = class_init (), %.loc9_35.6 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.7: init %C = converted %.loc9_34.1, %.loc9_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.8: init %array_type.002 = array_init (%.loc9_35.3, %.loc9_35.5, %.loc9_35.7) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_3.2: init %array_type.002 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3.2
// CHECK:STDOUT:   %.loc9_20: type = splice_block %array_type [concrete = constants.%array_type.002] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C [concrete = constants.%array_type.002]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.002 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.002 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%int_3) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc10: %array_type.002 = bind_value %a.ref
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %N.loc6_16.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc6_16.2 => constants.%N
// CHECK:STDOUT:   %array_type.loc6_52.2 => constants.%array_type.bb5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%int_3) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %N.loc6_16.2 => constants.%int_3
// CHECK:STDOUT:   %N.patt.loc6_16.2 => constants.%int_3
// CHECK:STDOUT:   %array_type.loc6_52.2 => constants.%array_type.002
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.dd1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bound_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array_type.9d4: type = array_type %int_2, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.d11: <witness> = require_complete_type %array_type.9d4 [symbolic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type.002 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %array_type.15a: type = array_type %int_2, %C [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %complete_type.8eb: <witness> = complete_type_witness %array_type.15a [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc13_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc13_29.2 (%array_type.9d4) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc13_29.2 (%array_type.9d4) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc13_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc13_6.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13_35: type = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc13_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc13_29.2 (%array_type.9d4) = value_param call_param0
// CHECK:STDOUT:     %.loc13_29: type = splice_block %array_type.loc13_29.1 [symbolic = %array_type.loc13_29.2 (constants.%array_type.9d4)] {
// CHECK:STDOUT:       %T.ref.loc13_25: type = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:       %array_type.loc13_29.1: type = array_type %int_2, %T [symbolic = %array_type.loc13_29.2 (constants.%array_type.9d4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc13_29.2 (%array_type.9d4) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc13_6.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc13_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_6.1: type) {
// CHECK:STDOUT:   %T.loc13_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %array_type.loc13_29.2: type = array_type constants.%int_2, @F.%T.loc13_6.2 (%T) [symbolic = %array_type.loc13_29.2 (constants.%array_type.9d4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_32: <witness> = require_complete_type @F.%T.loc13_6.2 (%T) [symbolic = %require_complete.loc13_32 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc13_17: <witness> = require_complete_type @F.%array_type.loc13_29.2 (%array_type.9d4) [symbolic = %require_complete.loc13_17 (constants.%require_complete.d11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc13_6.1: type](%a.param_patt: @F.%array_type.loc13_29.2 (%array_type.9d4)) -> @F.%T.loc13_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @F.%array_type.loc13_29.2 (%array_type.9d4) = name_ref a, %a
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13_48: %i32 = converted %int_0, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc13_49.1: ref @F.%array_type.loc13_29.2 (%array_type.9d4) = value_as_ref %a.ref
// CHECK:STDOUT:     %.loc13_49.2: ref @F.%T.loc13_6.2 (%T) = array_index %.loc13_49.1, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc13_49.3: @F.%T.loc13_6.2 (%T) = bind_value %.loc13_49.2 [concrete = <error>]
// CHECK:STDOUT:     return %.loc13_49.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type.002 = binding_pattern a
// CHECK:STDOUT:     %.loc17_3.1: %array_type.002 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.002 = var a
// CHECK:STDOUT:   %.loc17_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc17_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc17_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc17_35.1: %tuple.type = tuple_literal (%.loc17_26.1, %.loc17_30.1, %.loc17_34.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc17_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc17_26.2: init %C = class_init (), %.loc17_35.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc17_35.3: init %C = converted %.loc17_26.1, %.loc17_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc17_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc17_30.2: init %C = class_init (), %.loc17_35.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc17_35.5: init %C = converted %.loc17_30.1, %.loc17_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc17_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc17_34.2: init %C = class_init (), %.loc17_35.6 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc17_35.7: init %C = converted %.loc17_34.1, %.loc17_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc17_35.8: init %array_type.002 = array_init (%.loc17_35.3, %.loc17_35.5, %.loc17_35.7) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc17_3.2: init %array_type.002 = converted %.loc17_35.1, %.loc17_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc17_3.2
// CHECK:STDOUT:   %.loc17_20: type = splice_block %array_type [concrete = constants.%array_type.002] {
// CHECK:STDOUT:     %C.ref.loc17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C [concrete = constants.%array_type.002]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.002 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.002 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc15: ref %C = splice_block %return {}
// CHECK:STDOUT:   %.loc28: %array_type.15a = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %C = call %F.specific_fn(<error>) to %.loc15
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_6.2 => constants.%T
// CHECK:STDOUT:   %array_type.loc13_29.2 => constants.%array_type.9d4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc13_6.2 => constants.%C
// CHECK:STDOUT:   %array_type.loc13_29.2 => constants.%array_type.15a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_32 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc13_17 => constants.%complete_type.8eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %array_type.6a2: type = array_type %N, %C [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d82: <witness> = require_complete_type %array_type.6a2 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.fe4: type = array_type %int_3, %D [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.fe4 = tuple_value (%D.val, %D.val, %D.val) [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_3) [concrete]
// CHECK:STDOUT:   %complete_type.dd1: <witness> = complete_type_witness %array_type.002 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc14_6.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc14_42.2 (%array_type.6a2) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc14_42.2 (%array_type.6a2) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14_26.1: type = splice_block %.loc14_26.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc14_26.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc14_26.3: type = converted %int_literal.make_type, %.loc14_26.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc14_6.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc14_6.2 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc14_42.2 (%array_type.6a2) = value_param call_param0
// CHECK:STDOUT:     %.loc14_42: type = splice_block %array_type.loc14_42.1 [symbolic = %array_type.loc14_42.2 (constants.%array_type.6a2)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc14_41: Core.IntLiteral = name_ref N, %N.loc14_6.1 [symbolic = %N.loc14_6.2 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc14_42.1: type = array_type %N.ref.loc14_41, %C [symbolic = %array_type.loc14_42.2 (constants.%array_type.6a2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc14_42.2 (%array_type.6a2) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
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
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
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
// CHECK:STDOUT: generic fn @F(%N.loc14_6.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc14_6.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc14_6.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc14_6.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_6.2 (constants.%N.patt)]
// CHECK:STDOUT:   %array_type.loc14_42.2: type = array_type %N.loc14_6.2, %C [symbolic = %array_type.loc14_42.2 (constants.%array_type.6a2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%array_type.loc14_42.2 (%array_type.6a2) [symbolic = %require_complete (constants.%require_complete.d82)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.patt.loc14_6.1: Core.IntLiteral](%a.param_patt: @F.%array_type.loc14_42.2 (%array_type.6a2)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc14_61: Core.IntLiteral = name_ref N, %N.loc14_6.1 [symbolic = %N.loc14_6.2 (constants.%N)]
// CHECK:STDOUT:     %.loc14_62: %i32 = converted %N.ref.loc14_61, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type.fe4 = binding_pattern a
// CHECK:STDOUT:     %.loc18_3.1: %array_type.fe4 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.fe4 = var a
// CHECK:STDOUT:   %.loc18_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc18_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc18_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc18_35.1: %tuple.type = tuple_literal (%.loc18_26.1, %.loc18_30.1, %.loc18_34.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc18_35.2: ref %D = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc18_26.2: init %D = class_init (), %.loc18_35.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc18_35.3: init %D = converted %.loc18_26.1, %.loc18_26.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc18_35.4: ref %D = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc18_30.2: init %D = class_init (), %.loc18_35.4 [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc18_35.5: init %D = converted %.loc18_30.1, %.loc18_30.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc18_35.6: ref %D = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc18_34.2: init %D = class_init (), %.loc18_35.6 [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc18_35.7: init %D = converted %.loc18_34.1, %.loc18_34.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc18_35.8: init %array_type.fe4 = array_init (%.loc18_35.3, %.loc18_35.5, %.loc18_35.7) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc18_3.2: init %array_type.fe4 = converted %.loc18_35.1, %.loc18_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc18_3.2
// CHECK:STDOUT:   %.loc18_20: type = splice_block %array_type [concrete = constants.%array_type.fe4] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %D [concrete = constants.%array_type.fe4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.fe4 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.fe4 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_3) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc29_12: %array_type.002 = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(<error>)
// CHECK:STDOUT:   %.loc29_14.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc29_14.2: %i32 = converted %F.call, %.loc29_14.1
// CHECK:STDOUT:   return %.loc29_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc14_6.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc14_6.2 => constants.%N
// CHECK:STDOUT:   %array_type.loc14_42.2 => constants.%array_type.6a2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_3) {
// CHECK:STDOUT:   %N.loc14_6.2 => constants.%int_3
// CHECK:STDOUT:   %N.patt.loc14_6.2 => constants.%int_3
// CHECK:STDOUT:   %array_type.loc14_42.2 => constants.%array_type.002
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.dd1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bound_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.8e2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%C.val, %C.val, %C.val) [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc13_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc13_6.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc13_34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_34: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13_10: type = splice_block %i32.loc13_10 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc13_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc13_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc13_6.1: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc13_6.2 (constants.%N.51e)]
// CHECK:STDOUT:     %a.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.loc13_28: type = splice_block %array_type [concrete = <error>] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc13_27: %i32 = name_ref N, %N.loc13_6.1 [symbolic = %N.loc13_6.2 (constants.%N.51e)]
// CHECK:STDOUT:       %.loc13_27: Core.IntLiteral = converted %N.ref.loc13_27, <error> [concrete = <error>]
// CHECK:STDOUT:       %array_type: type = array_type <error>, %C [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: <error> = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
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
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc13_6.1: %i32) {
// CHECK:STDOUT:   %N.loc13_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc13_6.2 (constants.%N.51e)]
// CHECK:STDOUT:   %N.patt.loc13_6.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc13_6.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.patt.loc13_6.1: %i32](%a.param_patt: <error>) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc13_47: %i32 = name_ref N, %N.loc13_6.1 [symbolic = %N.loc13_6.2 (constants.%N.51e)]
// CHECK:STDOUT:     return %N.ref.loc13_47
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type = binding_pattern a
// CHECK:STDOUT:     %.loc16_3.1: %array_type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %.loc16_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc16_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc16_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc16_35.1: %tuple.type = tuple_literal (%.loc16_26.1, %.loc16_30.1, %.loc16_34.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc16_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc16_26.2: init %C = class_init (), %.loc16_35.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc16_35.3: init %C = converted %.loc16_26.1, %.loc16_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc16_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc16_30.2: init %C = class_init (), %.loc16_35.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc16_35.5: init %C = converted %.loc16_30.1, %.loc16_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc16_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc16_34.2: init %C = class_init (), %.loc16_35.6 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc16_35.7: init %C = converted %.loc16_34.1, %.loc16_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc16_35.8: init %array_type = array_init (%.loc16_35.3, %.loc16_35.5, %.loc16_35.7) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc16_3.2: init %array_type = converted %.loc16_35.1, %.loc16_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc16_3.2
// CHECK:STDOUT:   %.loc16_20: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type = name_ref a, %a
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc13_6.2 => constants.%N.51e
// CHECK:STDOUT:   %N.patt.loc13_6.2 => constants.%N.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:
