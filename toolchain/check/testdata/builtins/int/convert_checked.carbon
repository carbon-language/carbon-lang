// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/convert_checked.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/convert_checked.carbon
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

fn NegateI32(a: i32) -> i32 = "int.snegate";
fn SubI32(a: i32, b: i32) -> i32 = "int.ssub";
fn AddU32(a: u32, b: u32) -> u32 = "int.uadd";
fn IntLiteral() -> type = "int_literal.make_type";

// Size preserving
fn Int32ToInt32(a: i32) -> i32 = "int.convert_checked";
fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
fn Uint32ToInt32(a: u32) -> i32 = "int.convert_checked";
fn Uint32ToUint32(a: u32) -> u32 = "int.convert_checked";
fn IntLiteralToIntLiteral(a: IntLiteral()) -> IntLiteral() =
    "int.convert_checked";

// Narrowing
fn Int32ToInt16(a: i32) -> i16 = "int.convert_checked";
fn Int32ToUint16(a: i32) -> u16 = "int.convert_checked";
fn Uint32ToInt16(a: u32) -> i16 = "int.convert_checked";
fn Uint32ToUint16(a: u32) -> u16 = "int.convert_checked";
fn IntLiteralToInt16(a: IntLiteral()) -> i16 = "int.convert_checked";
fn IntLiteralToUint16(a: IntLiteral()) -> u16 = "int.convert_checked";

// Widening
fn Int32ToInt64(a: i32) -> i64 = "int.convert_checked";
fn Int32ToUint64(a: i32) -> u64 = "int.convert_checked";
fn Uint32ToInt64(a: u32) -> i64 = "int.convert_checked";
fn Uint32ToUint64(a: u32) -> u64 = "int.convert_checked";
fn Int32ToIntLiteral(a: i32) -> IntLiteral() = "int.convert_checked";
fn Uint32ToUintLiteral(a: u32) -> IntLiteral() = "int.convert_checked";

// --- identity.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: identity.carbon:[[@LINE+11]]:27: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let a: i32 = Int32ToInt32(0);
// CHECK:STDERR:                           ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: identity.carbon:[[@LINE-8]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:10:17: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToInt32(a: i32) -> i32 = "int.convert_checked";
// CHECK:STDERR:                 ^~~~~~
// CHECK:STDERR:
let a: i32 = Int32ToInt32(0);
// CHECK:STDERR: identity.carbon:[[@LINE+11]]:27: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let b: i32 = Int32ToInt32(0x7FFF_FFFF);
// CHECK:STDERR:                           ^~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: identity.carbon:[[@LINE-20]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:10:17: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToInt32(a: i32) -> i32 = "int.convert_checked";
// CHECK:STDERR:                 ^~~~~~
// CHECK:STDERR:
let b: i32 = Int32ToInt32(0x7FFF_FFFF);
// CHECK:STDERR: identity.carbon:[[@LINE+22]]:44: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let c: i32 = Int32ToInt32(SubI32(NegateI32(0x7FFF_FFFF), 1));
// CHECK:STDERR:                                            ^~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: identity.carbon:[[@LINE-32]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:4:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn NegateI32(a: i32) -> i32 = "int.snegate";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: identity.carbon:[[@LINE+11]]:58: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let c: i32 = Int32ToInt32(SubI32(NegateI32(0x7FFF_FFFF), 1));
// CHECK:STDERR:                                                          ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: identity.carbon:[[@LINE-43]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:5:19: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn SubI32(a: i32, b: i32) -> i32 = "int.ssub";
// CHECK:STDERR:                   ^~~~~~
// CHECK:STDERR:
let c: i32 = Int32ToInt32(SubI32(NegateI32(0x7FFF_FFFF), 1));
// CHECK:STDERR: identity.carbon:[[@LINE+11]]:74: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let d: IntLiteral() = IntLiteralToIntLiteral(Int32ToIntLiteral(NegateI32(1)));
// CHECK:STDERR:                                                                          ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: identity.carbon:[[@LINE-55]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:4:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn NegateI32(a: i32) -> i32 = "int.snegate";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
let d: IntLiteral() = IntLiteralToIntLiteral(Int32ToIntLiteral(NegateI32(1)));

// --- same_size.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: same_size.carbon:[[@LINE+11]]:30: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let max: u32 = Int32ToUint32(0x7FFF_FFFF);
// CHECK:STDERR:                              ^~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: same_size.carbon:[[@LINE-8]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let max: u32 = Int32ToUint32(0x7FFF_FFFF);
// CHECK:STDERR: same_size.carbon:[[@LINE+11]]:54: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let max_roundtrip: i32 = Uint32ToInt32(Int32ToUint32(0x7FFF_FFFF));
// CHECK:STDERR:                                                      ^~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: same_size.carbon:[[@LINE-20]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let max_roundtrip: i32 = Uint32ToInt32(Int32ToUint32(0x7FFF_FFFF));

// --- truncate.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: truncate.carbon:[[@LINE+11]]:28: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let a: u16 = Int32ToUint16(0);
// CHECK:STDERR:                            ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-8]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:19:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint16(a: i32) -> u16 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let a: u16 = Int32ToUint16(0);
// CHECK:STDERR: truncate.carbon:[[@LINE+11]]:28: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let b: u16 = Int32ToUint16(0xFFFF);
// CHECK:STDERR:                            ^~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-20]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:19:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint16(a: i32) -> u16 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let b: u16 = Int32ToUint16(0xFFFF);

// CHECK:STDERR: truncate.carbon:[[@LINE+11]]:27: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let c: i16 = Int32ToInt16(0x7FFF);
// CHECK:STDERR:                           ^~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-33]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:18:17: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToInt16(a: i32) -> i16 = "int.convert_checked";
// CHECK:STDERR:                 ^~~~~~
// CHECK:STDERR:
let c: i16 = Int32ToInt16(0x7FFF);
// CHECK:STDERR: truncate.carbon:[[@LINE+11]]:37: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let d: i16 = Int32ToInt16(NegateI32(0x8000));
// CHECK:STDERR:                                     ^~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-45]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:4:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn NegateI32(a: i32) -> i32 = "int.snegate";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
let d: i16 = Int32ToInt16(NegateI32(0x8000));

// CHECK:STDERR: truncate.carbon:[[@LINE+11]]:43: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let e: u16 = Uint32ToUint16(Int32ToUint32(0));
// CHECK:STDERR:                                           ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-58]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let e: u16 = Uint32ToUint16(Int32ToUint32(0));
// CHECK:STDERR: truncate.carbon:[[@LINE+11]]:43: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let f: u16 = Uint32ToUint16(Int32ToUint32(0xFFFF));
// CHECK:STDERR:                                           ^~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-70]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let f: u16 = Uint32ToUint16(Int32ToUint32(0xFFFF));

// CHECK:STDERR: truncate.carbon:[[@LINE+11]]:42: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let g: i16 = Uint32ToInt16(Int32ToUint32(0));
// CHECK:STDERR:                                          ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-83]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let g: i16 = Uint32ToInt16(Int32ToUint32(0));
// CHECK:STDERR: truncate.carbon:[[@LINE+11]]:42: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let h: i16 = Uint32ToInt16(Int32ToUint32(0x7FFF));
// CHECK:STDERR:                                          ^~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-95]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let h: i16 = Uint32ToInt16(Int32ToUint32(0x7FFF));

// CHECK:STDERR: truncate.carbon:[[@LINE+11]]:70: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let lit_i16_min: i16 = IntLiteralToInt16(Int32ToIntLiteral(NegateI32(0x8000)));
// CHECK:STDERR:                                                                      ^~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-108]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:4:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn NegateI32(a: i32) -> i32 = "int.snegate";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
let lit_i16_min: i16 = IntLiteralToInt16(Int32ToIntLiteral(NegateI32(0x8000)));
// CHECK:STDERR: truncate.carbon:[[@LINE+11]]:60: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let lit_i16_max: i16 = IntLiteralToInt16(Int32ToIntLiteral(0x7FFF));
// CHECK:STDERR:                                                            ^~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-120]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:30:22: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToIntLiteral(a: i32) -> IntLiteral() = "int.convert_checked";
// CHECK:STDERR:                      ^~~~~~
// CHECK:STDERR:
let lit_i16_max: i16 = IntLiteralToInt16(Int32ToIntLiteral(0x7FFF));

// CHECK:STDERR: truncate.carbon:[[@LINE+11]]:61: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let lit_u16_min: u16 = IntLiteralToUint16(Int32ToIntLiteral(0));
// CHECK:STDERR:                                                             ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-133]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:30:22: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToIntLiteral(a: i32) -> IntLiteral() = "int.convert_checked";
// CHECK:STDERR:                      ^~~~~~
// CHECK:STDERR:
let lit_u16_min: u16 = IntLiteralToUint16(Int32ToIntLiteral(0));
// CHECK:STDERR: truncate.carbon:[[@LINE+11]]:61: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let lit_u16_max: u16 = IntLiteralToUint16(Int32ToIntLiteral(0xFFFF));
// CHECK:STDERR:                                                             ^~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-145]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:30:22: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToIntLiteral(a: i32) -> IntLiteral() = "int.convert_checked";
// CHECK:STDERR:                      ^~~~~~
// CHECK:STDERR:
let lit_u16_max: u16 = IntLiteralToUint16(Int32ToIntLiteral(0xFFFF));

// --- zero_extend.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: zero_extend.carbon:[[@LINE+11]]:43: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let a: u64 = Uint32ToUint64(Int32ToUint32(0));
// CHECK:STDERR:                                           ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: zero_extend.carbon:[[@LINE-8]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let a: u64 = Uint32ToUint64(Int32ToUint32(0));
let b: u64 = Uint32ToUint64(
  AddU32(
    // CHECK:STDERR: zero_extend.carbon:[[@LINE+22]]:26: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
    // CHECK:STDERR:     AddU32(Int32ToUint32(0x7FFF_FFFF), Int32ToUint32(0x7FFF_FFFF)),
    // CHECK:STDERR:                          ^~~~~~~~~~~
    // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
    // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: zero_extend.carbon:[[@LINE-22]]:1: in import [InImport]
    // CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
    // CHECK:STDERR:                  ^~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: zero_extend.carbon:[[@LINE+11]]:54: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
    // CHECK:STDERR:     AddU32(Int32ToUint32(0x7FFF_FFFF), Int32ToUint32(0x7FFF_FFFF)),
    // CHECK:STDERR:                                                      ^~~~~~~~~~~
    // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
    // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: zero_extend.carbon:[[@LINE-33]]:1: in import [InImport]
    // CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
    // CHECK:STDERR:                  ^~~~~~
    // CHECK:STDERR:
    AddU32(Int32ToUint32(0x7FFF_FFFF), Int32ToUint32(0x7FFF_FFFF)),
    // CHECK:STDERR: zero_extend.carbon:[[@LINE+11]]:19: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
    // CHECK:STDERR:     Int32ToUint32(1)));
    // CHECK:STDERR:                   ^
    // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
    // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: zero_extend.carbon:[[@LINE-45]]:1: in import [InImport]
    // CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
    // CHECK:STDERR:                  ^~~~~~
    // CHECK:STDERR:
    Int32ToUint32(1)));

// CHECK:STDERR: zero_extend.carbon:[[@LINE+11]]:42: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let c: i64 = Uint32ToInt64(Int32ToUint32(0));
// CHECK:STDERR:                                          ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: zero_extend.carbon:[[@LINE-58]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let c: i64 = Uint32ToInt64(Int32ToUint32(0));
let d: i64 = Uint32ToInt64(
  AddU32(
    // CHECK:STDERR: zero_extend.carbon:[[@LINE+22]]:26: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
    // CHECK:STDERR:     AddU32(Int32ToUint32(0x7FFF_FFFF), Int32ToUint32(0x7FFF_FFFF)),
    // CHECK:STDERR:                          ^~~~~~~~~~~
    // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
    // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: zero_extend.carbon:[[@LINE-72]]:1: in import [InImport]
    // CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
    // CHECK:STDERR:                  ^~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: zero_extend.carbon:[[@LINE+11]]:54: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
    // CHECK:STDERR:     AddU32(Int32ToUint32(0x7FFF_FFFF), Int32ToUint32(0x7FFF_FFFF)),
    // CHECK:STDERR:                                                      ^~~~~~~~~~~
    // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
    // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: zero_extend.carbon:[[@LINE-83]]:1: in import [InImport]
    // CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
    // CHECK:STDERR:                  ^~~~~~
    // CHECK:STDERR:
    AddU32(Int32ToUint32(0x7FFF_FFFF), Int32ToUint32(0x7FFF_FFFF)),
    // CHECK:STDERR: zero_extend.carbon:[[@LINE+11]]:19: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
    // CHECK:STDERR:     Int32ToUint32(1)));
    // CHECK:STDERR:                   ^
    // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
    // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: zero_extend.carbon:[[@LINE-95]]:1: in import [InImport]
    // CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
    // CHECK:STDERR:                  ^~~~~~
    // CHECK:STDERR:
    Int32ToUint32(1)));

// --- sign_extend.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: sign_extend.carbon:[[@LINE+11]]:28: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let a: u64 = Int32ToUint64(0);
// CHECK:STDERR:                            ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: sign_extend.carbon:[[@LINE-8]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:27:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint64(a: i32) -> u64 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let a: u64 = Int32ToUint64(0);
// CHECK:STDERR: sign_extend.carbon:[[@LINE+11]]:28: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let b: u64 = Int32ToUint64(0x7FFF_FFFF);
// CHECK:STDERR:                            ^~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: sign_extend.carbon:[[@LINE-20]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:27:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint64(a: i32) -> u64 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let b: u64 = Int32ToUint64(0x7FFF_FFFF);

// CHECK:STDERR: sign_extend.carbon:[[@LINE+22]]:44: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let c: i64 = Int32ToInt64(SubI32(NegateI32(0x7FFF_FFFF), 1));
// CHECK:STDERR:                                            ^~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: sign_extend.carbon:[[@LINE-33]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:4:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn NegateI32(a: i32) -> i32 = "int.snegate";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: sign_extend.carbon:[[@LINE+11]]:58: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let c: i64 = Int32ToInt64(SubI32(NegateI32(0x7FFF_FFFF), 1));
// CHECK:STDERR:                                                          ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: sign_extend.carbon:[[@LINE-44]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:5:19: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn SubI32(a: i32, b: i32) -> i32 = "int.ssub";
// CHECK:STDERR:                   ^~~~~~
// CHECK:STDERR:
let c: i64 = Int32ToInt64(SubI32(NegateI32(0x7FFF_FFFF), 1));
// CHECK:STDERR: sign_extend.carbon:[[@LINE+11]]:27: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let d: i64 = Int32ToInt64(0x7FFF_FFFF);
// CHECK:STDERR:                           ^~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: sign_extend.carbon:[[@LINE-56]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:26:17: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToInt64(a: i32) -> i64 = "int.convert_checked";
// CHECK:STDERR:                 ^~~~~~
// CHECK:STDERR:
let d: i64 = Int32ToInt64(0x7FFF_FFFF);

// --- fail_too_large_u32_for_i32.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let max_plus_one: i32 =
  Uint32ToInt32(
    // CHECK:STDERR: fail_too_large_u32_for_i32.carbon:[[@LINE+11]]:26: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
    // CHECK:STDERR:     AddU32(Int32ToUint32(0x7FFF_FFFF),
    // CHECK:STDERR:                          ^~~~~~~~~~~
    // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
    // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_too_large_u32_for_i32.carbon:[[@LINE-10]]:1: in import [InImport]
    // CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
    // CHECK:STDERR:                  ^~~~~~
    // CHECK:STDERR:
    AddU32(Int32ToUint32(0x7FFF_FFFF),
           // CHECK:STDERR: fail_too_large_u32_for_i32.carbon:[[@LINE+11]]:26: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
           // CHECK:STDERR:            Int32ToUint32(1)));
           // CHECK:STDERR:                          ^
           // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
           // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
           // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           // CHECK:STDERR: fail_too_large_u32_for_i32.carbon:[[@LINE-22]]:1: in import [InImport]
           // CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
           // CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
           // CHECK:STDERR:                  ^~~~~~
           // CHECK:STDERR:
           Int32ToUint32(1)));

// --- fail_too_large_i32_for_i16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_large_i32_for_i16.carbon:[[@LINE+11]]:38: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let max_plus_one: i16 = Int32ToInt16(0x8000);
// CHECK:STDERR:                                      ^~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_too_large_i32_for_i16.carbon:[[@LINE-8]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:18:17: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToInt16(a: i32) -> i16 = "int.convert_checked";
// CHECK:STDERR:                 ^~~~~~
// CHECK:STDERR:
let max_plus_one: i16 = Int32ToInt16(0x8000);

// --- fail_too_large_i32_for_u16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_large_i32_for_u16.carbon:[[@LINE+11]]:39: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let max_plus_one: u16 = Int32ToUint16(0x1_0000);
// CHECK:STDERR:                                       ^~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_too_large_i32_for_u16.carbon:[[@LINE-8]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:19:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint16(a: i32) -> u16 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let max_plus_one: u16 = Int32ToUint16(0x1_0000);

// --- fail_too_large_u32_for_i16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_large_u32_for_i16.carbon:[[@LINE+11]]:53: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let max_plus_one: i16 = Uint32ToInt16(Int32ToUint32(0x8000));
// CHECK:STDERR:                                                     ^~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_too_large_u32_for_i16.carbon:[[@LINE-8]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let max_plus_one: i16 = Uint32ToInt16(Int32ToUint32(0x8000));

// --- fail_too_large_u32_for_u16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_large_u32_for_u16.carbon:[[@LINE+11]]:54: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let max_plus_one: u16 = Uint32ToUint16(Int32ToUint32(0x1_0000));
// CHECK:STDERR:                                                      ^~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_too_large_u32_for_u16.carbon:[[@LINE-8]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:11:18: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
let max_plus_one: u16 = Uint32ToUint16(Int32ToUint32(0x1_0000));

// --- fail_negative_i32_to_u16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_negative_i32_to_u16.carbon:[[@LINE+22]]:50: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let minus_one_to_u16: u16 = Int32ToUint16(SubI32(0, 1));
// CHECK:STDERR:                                                  ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_negative_i32_to_u16.carbon:[[@LINE-8]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:5:11: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn SubI32(a: i32, b: i32) -> i32 = "int.ssub";
// CHECK:STDERR:           ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_negative_i32_to_u16.carbon:[[@LINE+11]]:53: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let minus_one_to_u16: u16 = Int32ToUint16(SubI32(0, 1));
// CHECK:STDERR:                                                     ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_negative_i32_to_u16.carbon:[[@LINE-19]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:5:19: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn SubI32(a: i32, b: i32) -> i32 = "int.ssub";
// CHECK:STDERR:                   ^~~~~~
// CHECK:STDERR:
let minus_one_to_u16: u16 = Int32ToUint16(SubI32(0, 1));

// --- fail_negative_i32_to_u32.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_negative_i32_to_u32.carbon:[[@LINE+22]]:50: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let minus_one_to_u32: u32 = Int32ToUint32(SubI32(0, 1));
// CHECK:STDERR:                                                  ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_negative_i32_to_u32.carbon:[[@LINE-8]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:5:11: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn SubI32(a: i32, b: i32) -> i32 = "int.ssub";
// CHECK:STDERR:           ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_negative_i32_to_u32.carbon:[[@LINE+11]]:53: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let minus_one_to_u32: u32 = Int32ToUint32(SubI32(0, 1));
// CHECK:STDERR:                                                     ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_negative_i32_to_u32.carbon:[[@LINE-19]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:5:19: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn SubI32(a: i32, b: i32) -> i32 = "int.ssub";
// CHECK:STDERR:                   ^~~~~~
// CHECK:STDERR:
let minus_one_to_u32: u32 = Int32ToUint32(SubI32(0, 1));

// --- fail_negative_i32_to_u64.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_negative_i32_to_u64.carbon:[[@LINE+22]]:50: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let minus_one_to_u64: u64 = Int32ToUint64(SubI32(0, 1));
// CHECK:STDERR:                                                  ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_negative_i32_to_u64.carbon:[[@LINE-8]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:5:11: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn SubI32(a: i32, b: i32) -> i32 = "int.ssub";
// CHECK:STDERR:           ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_negative_i32_to_u64.carbon:[[@LINE+11]]:53: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let minus_one_to_u64: u64 = Int32ToUint64(SubI32(0, 1));
// CHECK:STDERR:                                                     ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_negative_i32_to_u64.carbon:[[@LINE-19]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:5:19: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn SubI32(a: i32, b: i32) -> i32 = "int.ssub";
// CHECK:STDERR:                   ^~~~~~
// CHECK:STDERR:
let minus_one_to_u64: u64 = Int32ToUint64(SubI32(0, 1));

// --- fail_too_small_i32_for_i16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_small_i32_for_i16.carbon:[[@LINE+11]]:49: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let min_minus_one: i16 = Int32ToInt16(NegateI32(0x8001));
// CHECK:STDERR:                                                 ^~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_too_small_i32_for_i16.carbon:[[@LINE-8]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:4:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn NegateI32(a: i32) -> i32 = "int.snegate";
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
let min_minus_one: i16 = Int32ToInt16(NegateI32(0x8001));

// --- fail_not_constant.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_not_constant.carbon:[[@LINE+7]]:25: error: member access into undefined facet type `Core.ImplicitAs(i32)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: let not_constant: i32 = 0;
// CHECK:STDERR:                         ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/e732e30cba9f3a5438beaa8b85e336f0/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let not_constant: i32 = 0;

let convert_not_constant_narrow: i16 = Int32ToInt16(not_constant);

let convert_not_constant_same: i32 = Int32ToInt32(not_constant);

let convert_not_constant_widen: i64 = Int32ToInt64(not_constant);
