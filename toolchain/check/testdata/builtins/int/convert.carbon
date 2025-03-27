// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/convert.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/convert.carbon
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

// --- int_ops.carbon

library "[[@TEST_NAME]]";

// Size preserving
fn Int32ToInt32(a: i32) -> i32 = "int.convert";
fn Int32ToUint32(a: i32) -> u32 = "int.convert";
fn Uint32ToInt32(a: u32) -> i32 = "int.convert";
fn Uint32ToUint32(a: u32) -> u32 = "int.convert";
fn IntLiteralToIntLiteral(a: Core.IntLiteral()) -> Core.IntLiteral() = "int.convert";

// Narrowing
fn Int32ToInt16(a: i32) -> i16 = "int.convert";
fn Int32ToUint16(a: i32) -> u16 = "int.convert";
fn Uint32ToInt16(a: u32) -> i16 = "int.convert";
fn Uint32ToUint16(a: u32) -> u16 = "int.convert";
fn IntLiteralToInt16(a: Core.IntLiteral()) -> i16 = "int.convert";
fn IntLiteralToUint16(a: Core.IntLiteral()) -> u16 = "int.convert";

// Widening
fn Int32ToInt64(a: i32) -> i64 = "int.convert";
fn Int32ToUint64(a: i32) -> u64 = "int.convert";
fn Uint32ToInt64(a: u32) -> i64 = "int.convert";
fn Uint32ToUint64(a: u32) -> u64 = "int.convert";
fn Int32ToIntLiteral(a: i32) -> Core.IntLiteral() = "int.convert";
fn Uint32ToIntLiteral(a: u32) -> Core.IntLiteral() = "int.convert";

class Expect[T:! type](N:! T) {}
fn Test[T:! type](N:! T) -> Expect(N) { return {}; }

// --- fail_self_test.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

fn F() {
  // Ensure our testing machinery works.
  // CHECK:STDERR: fail_self_test.carbon:[[@LINE+18]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt32(0)) as Expect(1 as i32);
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_self_test.carbon:[[@LINE-10]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:5:17: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToInt32(a: i32) -> i32 = "int.convert";
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_self_test.carbon:[[@LINE+7]]:35: error: member access into undefined facet type `Core.As(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt32(0)) as Expect(1 as i32);
  // CHECK:STDERR:                                   ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt32(0)) as Expect(1 as i32);
}

// --- identity.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

fn F() {
  // CHECK:STDERR: identity.carbon:[[@LINE+14]]:21: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt32(-0x8000_0000)) as Expect(-0x8000_0000 as i32);
  // CHECK:STDERR:                     ^~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: identity.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt32(-0x8000_0000)) as Expect(-0x8000_0000 as i32);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt32(-0x8000_0000)) as Expect(-0x8000_0000 as i32);
  // CHECK:STDERR: identity.carbon:[[@LINE+14]]:21: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt32(-1)) as Expect(-1 as i32);
  // CHECK:STDERR:                     ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: identity.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt32(-1)) as Expect(-1 as i32);
  // CHECK:STDERR:                                    ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt32(-1)) as Expect(-1 as i32);
  // CHECK:STDERR: identity.carbon:[[@LINE+18]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt32(0)) as Expect(0 as i32);
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: identity.carbon:[[@LINE-39]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:5:17: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToInt32(a: i32) -> i32 = "int.convert";
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: identity.carbon:[[@LINE+7]]:35: error: member access into undefined facet type `Core.As(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt32(0)) as Expect(0 as i32);
  // CHECK:STDERR:                                   ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt32(0)) as Expect(0 as i32);
  // CHECK:STDERR: identity.carbon:[[@LINE+18]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt32(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as i32);
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: identity.carbon:[[@LINE-58]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:5:17: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToInt32(a: i32) -> i32 = "int.convert";
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: identity.carbon:[[@LINE+7]]:45: error: member access into undefined facet type `Core.As(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt32(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as i32);
  // CHECK:STDERR:                                             ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt32(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as i32);

  // CHECK:STDERR: identity.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint32(0)) as Expect(0 as u32);
  // CHECK:STDERR:                       ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: identity.carbon:[[@LINE-78]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:8:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint32(a: u32) -> u32 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: identity.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Core.As(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint32(0)) as Expect(0 as u32);
  // CHECK:STDERR:                                     ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint32(0)) as Expect(0 as u32);
  // CHECK:STDERR: identity.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint32(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as u32);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: identity.carbon:[[@LINE-97]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:8:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint32(a: u32) -> u32 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: identity.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint32(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as u32);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint32(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as u32);
  // CHECK:STDERR: identity.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint32(0x8000_0000)) as Expect(0x8000_0000 as u32);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: identity.carbon:[[@LINE-116]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:8:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint32(a: u32) -> u32 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: identity.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint32(0x8000_0000)) as Expect(0x8000_0000 as u32);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint32(0x8000_0000)) as Expect(0x8000_0000 as u32);
  // CHECK:STDERR: identity.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint32(0xFFFF_FFFF)) as Expect(0xFFFF_FFFF as u32);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: identity.carbon:[[@LINE-135]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:8:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint32(a: u32) -> u32 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: identity.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint32(0xFFFF_FFFF)) as Expect(0xFFFF_FFFF as u32);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint32(0xFFFF_FFFF)) as Expect(0xFFFF_FFFF as u32);

  Test(IntLiteralToIntLiteral(0x1_0000_0000_0000_0000)) as
      Expect(0x1_0000_0000_0000_0000);
  // CHECK:STDERR: identity.carbon:[[@LINE+14]]:31: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToIntLiteral(-1)) as Expect(-1);
  // CHECK:STDERR:                               ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: identity.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToIntLiteral(-1)) as Expect(-1);
  // CHECK:STDERR:                                              ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(IntLiteralToIntLiteral(-1)) as Expect(-1);
}

// --- same_size.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

fn F() {
  // CHECK:STDERR: same_size.carbon:[[@LINE+14]]:22: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint32(-0x8000_0000)) as Expect(0x8000_0000 as u32);
  // CHECK:STDERR:                      ^~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: same_size.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint32(-0x8000_0000)) as Expect(0x8000_0000 as u32);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint32(-0x8000_0000)) as Expect(0x8000_0000 as u32);
  // CHECK:STDERR: same_size.carbon:[[@LINE+14]]:22: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint32(-1)) as Expect(0xFFFF_FFFF as u32);
  // CHECK:STDERR:                      ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: same_size.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Core.As(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint32(-1)) as Expect(0xFFFF_FFFF as u32);
  // CHECK:STDERR:                                     ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint32(-1)) as Expect(0xFFFF_FFFF as u32);
  // CHECK:STDERR: same_size.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint32(0)) as Expect(0 as u32);
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: same_size.carbon:[[@LINE-39]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:6:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: same_size.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.As(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint32(0)) as Expect(0 as u32);
  // CHECK:STDERR:                                    ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint32(0)) as Expect(0 as u32);
  // CHECK:STDERR: same_size.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint32(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as u32);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: same_size.carbon:[[@LINE-58]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:6:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: same_size.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint32(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as u32);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint32(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as u32);

  // CHECK:STDERR: same_size.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt32(0)) as Expect(0 as i32);
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: same_size.carbon:[[@LINE-78]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:7:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt32(a: u32) -> i32 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: same_size.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.As(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt32(0)) as Expect(0 as i32);
  // CHECK:STDERR:                                    ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt32(0)) as Expect(0 as i32);
  // CHECK:STDERR: same_size.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt32(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as i32);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: same_size.carbon:[[@LINE-97]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:7:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt32(a: u32) -> i32 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: same_size.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt32(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as i32);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt32(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as i32);
  // CHECK:STDERR: same_size.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt32(0x8000_0000)) as Expect(-0x8000_0000 as i32);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: same_size.carbon:[[@LINE-116]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:7:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt32(a: u32) -> i32 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: same_size.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt32(0x8000_0000)) as Expect(-0x8000_0000 as i32);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt32(0x8000_0000)) as Expect(-0x8000_0000 as i32);
  // CHECK:STDERR: same_size.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt32(0xFFFF_FFFF)) as Expect(-1 as i32);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: same_size.carbon:[[@LINE-135]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:7:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt32(a: u32) -> i32 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: same_size.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt32(0xFFFF_FFFF)) as Expect(-1 as i32);
  // CHECK:STDERR:                                              ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt32(0xFFFF_FFFF)) as Expect(-1 as i32);
}

// --- truncate.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

fn F() {
  // CHECK:STDERR: truncate.carbon:[[@LINE+14]]:21: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(-0x8000_0000)) as Expect(0 as i16);
  // CHECK:STDERR:                     ^~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(i16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(-0x8000_0000)) as Expect(0 as i16);
  // CHECK:STDERR:                                              ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt16(-0x8000_0000)) as Expect(0 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+14]]:21: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(-0x7FFF_EDCC)) as Expect(0x1234 as i16);
  // CHECK:STDERR:                     ^~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(i16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(-0x7FFF_EDCC)) as Expect(0x1234 as i16);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt16(-0x7FFF_EDCC)) as Expect(0x1234 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+14]]:21: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(-0x7FFF_1234)) as Expect(-0x1234 as i16);
  // CHECK:STDERR:                     ^~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(-0x7FFF_1234)) as Expect(-0x1234 as i16);
  // CHECK:STDERR:                                              ^~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt16(-0x7FFF_1234)) as Expect(-0x1234 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+14]]:21: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(-0x8000)) as Expect(-0x8000 as i16);
  // CHECK:STDERR:                     ^~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:41: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(-0x8000)) as Expect(-0x8000 as i16);
  // CHECK:STDERR:                                         ^~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt16(-0x8000)) as Expect(-0x8000 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+14]]:21: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(-1)) as Expect(-1 as i16);
  // CHECK:STDERR:                     ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(-1)) as Expect(-1 as i16);
  // CHECK:STDERR:                                    ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt16(-1)) as Expect(-1 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(0)) as Expect(0 as i16);
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-84]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:12:17: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToInt16(a: i32) -> i16 = "int.convert";
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:35: error: member access into undefined facet type `Core.As(i16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(0)) as Expect(0 as i16);
  // CHECK:STDERR:                                   ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt16(0)) as Expect(0 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(0x7FFF)) as Expect(0x7FFF as i16);
  // CHECK:STDERR:                     ^~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-103]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:12:17: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToInt16(a: i32) -> i16 = "int.convert";
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Core.As(i16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(0x7FFF)) as Expect(0x7FFF as i16);
  // CHECK:STDERR:                                        ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt16(0x7FFF)) as Expect(0x7FFF as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(0xFFFF)) as Expect(-1 as i16);
  // CHECK:STDERR:                     ^~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-122]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:12:17: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToInt16(a: i32) -> i16 = "int.convert";
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(0xFFFF)) as Expect(-1 as i16);
  // CHECK:STDERR:                                        ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt16(0xFFFF)) as Expect(-1 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(0x7FFF_1234)) as Expect(0x1234 as i16);
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-141]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:12:17: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToInt16(a: i32) -> i16 = "int.convert";
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:45: error: member access into undefined facet type `Core.As(i16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(0x7FFF_1234)) as Expect(0x1234 as i16);
  // CHECK:STDERR:                                             ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt16(0x7FFF_1234)) as Expect(0x1234 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(0x7FFF_EDCC)) as Expect(-0x1234 as i16);
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-160]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:12:17: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToInt16(a: i32) -> i16 = "int.convert";
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:45: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(0x7FFF_EDCC)) as Expect(-0x1234 as i16);
  // CHECK:STDERR:                                             ^~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt16(0x7FFF_EDCC)) as Expect(-0x1234 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(0x7FFF_FFFF)) as Expect(-1 as i16);
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-179]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:12:17: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToInt16(a: i32) -> i16 = "int.convert";
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:45: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt16(0x7FFF_FFFF)) as Expect(-1 as i16);
  // CHECK:STDERR:                                             ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt16(0x7FFF_FFFF)) as Expect(-1 as i16);

  // CHECK:STDERR: truncate.carbon:[[@LINE+14]]:22: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(-0x8000_0000)) as Expect(0 as u16);
  // CHECK:STDERR:                      ^~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(-0x8000_0000)) as Expect(0 as u16);
  // CHECK:STDERR:                                               ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint16(-0x8000_0000)) as Expect(0 as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+14]]:22: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(-0x7FFF_EDCC)) as Expect(0x1234 as u16);
  // CHECK:STDERR:                      ^~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(-0x7FFF_EDCC)) as Expect(0x1234 as u16);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint16(-0x7FFF_EDCC)) as Expect(0x1234 as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+14]]:22: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(-0x7FFF_1234)) as Expect(0xEDCC as u16);
  // CHECK:STDERR:                      ^~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(-0x7FFF_1234)) as Expect(0xEDCC as u16);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint16(-0x7FFF_1234)) as Expect(0xEDCC as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+14]]:22: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(-0x8000)) as Expect(0x8000 as u16);
  // CHECK:STDERR:                      ^~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:42: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(-0x8000)) as Expect(0x8000 as u16);
  // CHECK:STDERR:                                          ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint16(-0x8000)) as Expect(0x8000 as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+14]]:22: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(-1)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                      ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(-1)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                                     ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint16(-1)) as Expect(0xFFFF as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(0)) as Expect(0 as u16);
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-274]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:13:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToUint16(a: i32) -> u16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(0)) as Expect(0 as u16);
  // CHECK:STDERR:                                    ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint16(0)) as Expect(0 as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(0x7FFF)) as Expect(0x7FFF as u16);
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-293]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:13:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToUint16(a: i32) -> u16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:41: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(0x7FFF)) as Expect(0x7FFF as u16);
  // CHECK:STDERR:                                         ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint16(0x7FFF)) as Expect(0x7FFF as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(0xFFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-312]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:13:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToUint16(a: i32) -> u16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:41: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(0xFFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                                         ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint16(0xFFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(0x7FFF_1234)) as Expect(0x1234 as u16);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-331]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:13:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToUint16(a: i32) -> u16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(0x7FFF_1234)) as Expect(0x1234 as u16);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint16(0x7FFF_1234)) as Expect(0x1234 as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(0x7FFF_EDCC)) as Expect(0xEDCC as u16);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-350]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:13:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToUint16(a: i32) -> u16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(0x7FFF_EDCC)) as Expect(0xEDCC as u16);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint16(0x7FFF_EDCC)) as Expect(0xEDCC as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(0x7FFF_FFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-369]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:13:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToUint16(a: i32) -> u16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint16(0x7FFF_FFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint16(0x7FFF_FFFF)) as Expect(0xFFFF as u16);

  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0x8000_0000)) as Expect(0 as i16);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-389]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:14:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt16(a: u32) -> i16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(i16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0x8000_0000)) as Expect(0 as i16);
  // CHECK:STDERR:                                              ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt16(0x8000_0000)) as Expect(0 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0xFFFF_1234)) as Expect(0x1234 as i16);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-408]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:14:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt16(a: u32) -> i16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(i16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0xFFFF_1234)) as Expect(0x1234 as i16);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt16(0xFFFF_1234)) as Expect(0x1234 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0xFFFF_EDCC)) as Expect(-0x1234 as i16);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-427]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:14:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt16(a: u32) -> i16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0xFFFF_EDCC)) as Expect(-0x1234 as i16);
  // CHECK:STDERR:                                              ^~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt16(0xFFFF_EDCC)) as Expect(-0x1234 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0xFFFF_8000)) as Expect(-0x8000 as i16);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-446]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:14:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt16(a: u32) -> i16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0xFFFF_8000)) as Expect(-0x8000 as i16);
  // CHECK:STDERR:                                              ^~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt16(0xFFFF_8000)) as Expect(-0x8000 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0xFFFF_FFFF)) as Expect(-1 as i16);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-465]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:14:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt16(a: u32) -> i16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0xFFFF_FFFF)) as Expect(-1 as i16);
  // CHECK:STDERR:                                              ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt16(0xFFFF_FFFF)) as Expect(-1 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0)) as Expect(0 as i16);
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-484]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:14:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt16(a: u32) -> i16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.As(i16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0)) as Expect(0 as i16);
  // CHECK:STDERR:                                    ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt16(0)) as Expect(0 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0x7FFF)) as Expect(0x7FFF as i16);
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-503]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:14:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt16(a: u32) -> i16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:41: error: member access into undefined facet type `Core.As(i16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0x7FFF)) as Expect(0x7FFF as i16);
  // CHECK:STDERR:                                         ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt16(0x7FFF)) as Expect(0x7FFF as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0xFFFF)) as Expect(-1 as i16);
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-522]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:14:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt16(a: u32) -> i16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:41: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0xFFFF)) as Expect(-1 as i16);
  // CHECK:STDERR:                                         ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt16(0xFFFF)) as Expect(-1 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0x7FFF_1234)) as Expect(0x1234 as i16);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-541]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:14:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt16(a: u32) -> i16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(i16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0x7FFF_1234)) as Expect(0x1234 as i16);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt16(0x7FFF_1234)) as Expect(0x1234 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0x7FFF_EDCC)) as Expect(-0x1234 as i16);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-560]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:14:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt16(a: u32) -> i16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0x7FFF_EDCC)) as Expect(-0x1234 as i16);
  // CHECK:STDERR:                                              ^~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt16(0x7FFF_EDCC)) as Expect(-0x1234 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0x7FFF_FFFF)) as Expect(-1 as i16);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-579]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:14:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt16(a: u32) -> i16 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt16(0x7FFF_FFFF)) as Expect(-1 as i16);
  // CHECK:STDERR:                                              ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt16(0x7FFF_FFFF)) as Expect(-1 as i16);

  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0x8000_0000)) as Expect(0 as u16);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-599]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:15:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint16(a: u32) -> u16 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0x8000_0000)) as Expect(0 as u16);
  // CHECK:STDERR:                                               ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint16(0x8000_0000)) as Expect(0 as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0xFFFF_1234)) as Expect(0x1234 as u16);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-618]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:15:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint16(a: u32) -> u16 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0xFFFF_1234)) as Expect(0x1234 as u16);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint16(0xFFFF_1234)) as Expect(0x1234 as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0xFFFF_EDCC)) as Expect(0xEDCC as u16);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-637]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:15:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint16(a: u32) -> u16 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0xFFFF_EDCC)) as Expect(0xEDCC as u16);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint16(0xFFFF_EDCC)) as Expect(0xEDCC as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0xFFFF_8000)) as Expect(0x8000 as u16);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-656]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:15:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint16(a: u32) -> u16 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0xFFFF_8000)) as Expect(0x8000 as u16);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint16(0xFFFF_8000)) as Expect(0x8000 as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0xFFFF_FFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-675]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:15:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint16(a: u32) -> u16 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0xFFFF_FFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint16(0xFFFF_FFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0)) as Expect(0 as u16);
  // CHECK:STDERR:                       ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-694]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:15:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint16(a: u32) -> u16 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0)) as Expect(0 as u16);
  // CHECK:STDERR:                                     ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint16(0)) as Expect(0 as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0x7FFF)) as Expect(0x7FFF as u16);
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-713]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:15:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint16(a: u32) -> u16 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:42: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0x7FFF)) as Expect(0x7FFF as u16);
  // CHECK:STDERR:                                          ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint16(0x7FFF)) as Expect(0x7FFF as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0xFFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-732]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:15:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint16(a: u32) -> u16 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:42: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0xFFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                                          ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint16(0xFFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0x7FFF_1234)) as Expect(0x1234 as u16);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-751]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:15:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint16(a: u32) -> u16 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0x7FFF_1234)) as Expect(0x1234 as u16);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint16(0x7FFF_1234)) as Expect(0x1234 as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0x7FFF_EDCC)) as Expect(0xEDCC as u16);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-770]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:15:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint16(a: u32) -> u16 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0x7FFF_EDCC)) as Expect(0xEDCC as u16);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint16(0x7FFF_EDCC)) as Expect(0xEDCC as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0x7FFF_FFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: truncate.carbon:[[@LINE-789]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:15:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint16(a: u32) -> u16 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint16(0x7FFF_FFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint16(0x7FFF_FFFF)) as Expect(0xFFFF as u16);

  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Core.As(i16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToInt16(0)) as Expect(0 as i16);
  // CHECK:STDERR:                                        ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(IntLiteralToInt16(0)) as Expect(0 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:45: error: member access into undefined facet type `Core.As(i16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToInt16(0x7FFF)) as Expect(0x7FFF as i16);
  // CHECK:STDERR:                                             ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(IntLiteralToInt16(0x7FFF)) as Expect(0x7FFF as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:45: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToInt16(0x8000)) as Expect(-0x8000 as i16);
  // CHECK:STDERR:                                             ^~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(IntLiteralToInt16(0x8000)) as Expect(-0x8000 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:45: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToInt16(0xFFFF)) as Expect(-1 as i16);
  // CHECK:STDERR:                                             ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(IntLiteralToInt16(0xFFFF)) as Expect(-1 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(i16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToInt16(0x1_2345)) as Expect(0x2345 as i16);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(IntLiteralToInt16(0x1_2345)) as Expect(0x2345 as i16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+14]]:26: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToInt16(-1)) as Expect(-1 as i16);
  // CHECK:STDERR:                          ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:41: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToInt16(-1)) as Expect(-1 as i16);
  // CHECK:STDERR:                                         ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(IntLiteralToInt16(-1)) as Expect(-1 as i16);

  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:41: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToUint16(0)) as Expect(0 as u16);
  // CHECK:STDERR:                                         ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(IntLiteralToUint16(0)) as Expect(0 as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToUint16(0x7FFF)) as Expect(0x7FFF as u16);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(IntLiteralToUint16(0x7FFF)) as Expect(0x7FFF as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToUint16(0x8000)) as Expect(0x8000 as u16);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(IntLiteralToUint16(0x8000)) as Expect(0x8000 as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToUint16(0xFFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(IntLiteralToUint16(0xFFFF)) as Expect(0xFFFF as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:48: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToUint16(0x1_2345)) as Expect(0x2345 as u16);
  // CHECK:STDERR:                                                ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(IntLiteralToUint16(0x1_2345)) as Expect(0x2345 as u16);
  // CHECK:STDERR: truncate.carbon:[[@LINE+14]]:27: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToUint16(-1)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                           ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: truncate.carbon:[[@LINE+7]]:42: error: member access into undefined facet type `Core.As(u16)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(IntLiteralToUint16(-1)) as Expect(0xFFFF as u16);
  // CHECK:STDERR:                                          ^~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(IntLiteralToUint16(-1)) as Expect(0xFFFF as u16);
}

// --- zero_extend.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

fn F() {
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt64(0)) as Expect(0 as i64);
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: zero_extend.carbon:[[@LINE-9]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:22:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt64(a: u32) -> i64 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.As(i64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt64(0)) as Expect(0 as i64);
  // CHECK:STDERR:                                    ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt64(0)) as Expect(0 as i64);
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt64(0x1234_5678)) as Expect(0x1234_5678 as i64);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: zero_extend.carbon:[[@LINE-28]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:22:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt64(a: u32) -> i64 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(i64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt64(0x1234_5678)) as Expect(0x1234_5678 as i64);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt64(0x1234_5678)) as Expect(0x1234_5678 as i64);
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt64(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as i64);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: zero_extend.carbon:[[@LINE-47]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:22:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt64(a: u32) -> i64 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(i64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt64(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as i64);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt64(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as i64);
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt64(0x8000_0000)) as Expect(0x8000_0000 as i64);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: zero_extend.carbon:[[@LINE-66]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:22:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt64(a: u32) -> i64 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(i64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt64(0x8000_0000)) as Expect(0x8000_0000 as i64);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt64(0x8000_0000)) as Expect(0x8000_0000 as i64);
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt64(0xFFFF_FFFF)) as Expect(0xFFFF_FFFF as i64);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: zero_extend.carbon:[[@LINE-85]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:22:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToInt64(a: u32) -> i64 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(i64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToInt64(0xFFFF_FFFF)) as Expect(0xFFFF_FFFF as i64);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToInt64(0xFFFF_FFFF)) as Expect(0xFFFF_FFFF as i64);

  // CHECK:STDERR: zero_extend.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint64(0)) as Expect(0 as u64);
  // CHECK:STDERR:                       ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: zero_extend.carbon:[[@LINE-105]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:23:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint64(a: u32) -> u64 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Core.As(u64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint64(0)) as Expect(0 as u64);
  // CHECK:STDERR:                                     ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint64(0)) as Expect(0 as u64);
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint64(0x1234_5678)) as Expect(0x1234_5678 as u64);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: zero_extend.carbon:[[@LINE-124]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:23:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint64(a: u32) -> u64 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint64(0x1234_5678)) as Expect(0x1234_5678 as u64);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint64(0x1234_5678)) as Expect(0x1234_5678 as u64);
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint64(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as u64);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: zero_extend.carbon:[[@LINE-143]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:23:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint64(a: u32) -> u64 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint64(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as u64);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint64(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as u64);
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint64(0x8000_0000)) as Expect(0x8000_0000 as u64);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: zero_extend.carbon:[[@LINE-162]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:23:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint64(a: u32) -> u64 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint64(0x8000_0000)) as Expect(0x8000_0000 as u64);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint64(0x8000_0000)) as Expect(0x8000_0000 as u64);
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+18]]:23: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint64(0xFFFF_FFFF)) as Expect(0xFFFF_FFFF as u64);
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: zero_extend.carbon:[[@LINE-181]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:23:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToUint64(a: u32) -> u64 = "int.convert";
  // CHECK:STDERR:                   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToUint64(0xFFFF_FFFF)) as Expect(0xFFFF_FFFF as u64);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Uint32ToUint64(0xFFFF_FFFF)) as Expect(0xFFFF_FFFF as u64);

  // CHECK:STDERR: zero_extend.carbon:[[@LINE+11]]:27: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToIntLiteral(0x1234_5678)) as Expect(0x1234_5678);
  // CHECK:STDERR:                           ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: zero_extend.carbon:[[@LINE-201]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:25:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToIntLiteral(a: u32) -> Core.IntLiteral() = "int.convert";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  Test(Uint32ToIntLiteral(0x1234_5678)) as Expect(0x1234_5678);
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+11]]:27: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToIntLiteral(0x8765_4321)) as Expect(0x8765_4321);
  // CHECK:STDERR:                           ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: zero_extend.carbon:[[@LINE-213]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:25:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToIntLiteral(a: u32) -> Core.IntLiteral() = "int.convert";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  Test(Uint32ToIntLiteral(0x8765_4321)) as Expect(0x8765_4321);
  // CHECK:STDERR: zero_extend.carbon:[[@LINE+11]]:27: error: member access into undefined facet type `Core.ImplicitAs(u32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Uint32ToIntLiteral(0xFFFF_FFFF)) as Expect(0xFFFF_FFFF);
  // CHECK:STDERR:                           ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: zero_extend.carbon:[[@LINE-225]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:25:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Uint32ToIntLiteral(a: u32) -> Core.IntLiteral() = "int.convert";
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  Test(Uint32ToIntLiteral(0xFFFF_FFFF)) as Expect(0xFFFF_FFFF);
}

// --- sign_extend.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

fn F() {
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+18]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt64(0)) as Expect(0 as i64);
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: sign_extend.carbon:[[@LINE-9]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:20:17: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToInt64(a: i32) -> i64 = "int.convert";
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+7]]:35: error: member access into undefined facet type `Core.As(i64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt64(0)) as Expect(0 as i64);
  // CHECK:STDERR:                                   ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt64(0)) as Expect(0 as i64);
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+18]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt64(0x1234_5678)) as Expect(0x1234_5678 as i64);
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: sign_extend.carbon:[[@LINE-28]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:20:17: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToInt64(a: i32) -> i64 = "int.convert";
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+7]]:45: error: member access into undefined facet type `Core.As(i64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt64(0x1234_5678)) as Expect(0x1234_5678 as i64);
  // CHECK:STDERR:                                             ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt64(0x1234_5678)) as Expect(0x1234_5678 as i64);
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+18]]:21: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt64(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as i64);
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: sign_extend.carbon:[[@LINE-47]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:20:17: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToInt64(a: i32) -> i64 = "int.convert";
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+7]]:45: error: member access into undefined facet type `Core.As(i64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt64(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as i64);
  // CHECK:STDERR:                                             ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt64(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as i64);
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+14]]:21: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt64(-1)) as Expect(-1 as i64);
  // CHECK:STDERR:                     ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToInt64(-1)) as Expect(-1 as i64);
  // CHECK:STDERR:                                    ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToInt64(-1)) as Expect(-1 as i64);

  // CHECK:STDERR: sign_extend.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint64(0)) as Expect(0 as u64);
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: sign_extend.carbon:[[@LINE-82]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:21:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToUint64(a: i32) -> u64 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `Core.As(u64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint64(0)) as Expect(0 as u64);
  // CHECK:STDERR:                                    ^~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint64(0)) as Expect(0 as u64);
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint64(0x1234_5678)) as Expect(0x1234_5678 as u64);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: sign_extend.carbon:[[@LINE-101]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:21:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToUint64(a: i32) -> u64 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(u64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint64(0x1234_5678)) as Expect(0x1234_5678 as u64);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint64(0x1234_5678)) as Expect(0x1234_5678 as u64);
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+18]]:22: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint64(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as u64);
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: sign_extend.carbon:[[@LINE-120]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:21:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToUint64(a: i32) -> u64 = "int.convert";
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+7]]:46: error: member access into undefined facet type `Core.As(u64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint64(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as u64);
  // CHECK:STDERR:                                              ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint64(0x7FFF_FFFF)) as Expect(0x7FFF_FFFF as u64);
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+14]]:22: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint64(-1)) as Expect(0xFFFF_FFFF_FFFF_FFFF as u64);
  // CHECK:STDERR:                      ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Core.As(u64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint64(-1)) as Expect(0xFFFF_FFFF_FFFF_FFFF as u64);
  // CHECK:STDERR:                                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint64(-1)) as Expect(0xFFFF_FFFF_FFFF_FFFF as u64);
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+14]]:22: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint64(-0x8000_0000)) as Expect(0xFFFF_FFFF_8000_0000 as u64);
  // CHECK:STDERR:                      ^~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Core.As(u64)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToUint64(-0x8000_0000)) as Expect(0xFFFF_FFFF_8000_0000 as u64);
  // CHECK:STDERR:                                               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:7:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface As(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToUint64(-0x8000_0000)) as Expect(0xFFFF_FFFF_8000_0000 as u64);

  // CHECK:STDERR: sign_extend.carbon:[[@LINE+11]]:26: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToIntLiteral(0x1234_5678)) as Expect(0x1234_5678);
  // CHECK:STDERR:                          ^~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: sign_extend.carbon:[[@LINE-170]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:24:22: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int32ToIntLiteral(a: i32) -> Core.IntLiteral() = "int.convert";
  // CHECK:STDERR:                      ^~~~~~
  // CHECK:STDERR:
  Test(Int32ToIntLiteral(0x1234_5678)) as Expect(0x1234_5678);
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+14]]:26: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToIntLiteral(-0x1234_5678)) as Expect(-0x1234_5678);
  // CHECK:STDERR:                          ^~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+7]]:51: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToIntLiteral(-0x1234_5678)) as Expect(-0x1234_5678);
  // CHECK:STDERR:                                                   ^~~~~~~~~~~~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToIntLiteral(-0x1234_5678)) as Expect(-0x1234_5678);
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+14]]:26: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToIntLiteral(-1)) as Expect(-1);
  // CHECK:STDERR:                          ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: sign_extend.carbon:[[@LINE+7]]:41: error: member access into undefined facet type `Core.Negate` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   Test(Int32ToIntLiteral(-1)) as Expect(-1);
  // CHECK:STDERR:                                         ^~
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon:25:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Negate {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(Int32ToIntLiteral(-1)) as Expect(-1);
}

// --- fail_not_constant.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let not_constant: Core.IntLiteral() = 0;

// CHECK:STDERR: fail_not_constant.carbon:[[@LINE+8]]:33: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let convert_not_constant: i16 = IntLiteralToInt16(not_constant);
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_constant.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:16:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn IntLiteralToInt16(a: Core.IntLiteral()) -> i16 = "int.convert";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let convert_not_constant: i16 = IntLiteralToInt16(not_constant);
