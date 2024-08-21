// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --no-debug-info --phase=lower --dump-llvm-ir --output=- %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/debug/nodebug.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/debug/nodebug.carbon

// CHECK:STDOUT: ; ModuleID = '/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude.carbon'
// CHECK:STDOUT: source_filename = "/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude.carbon"
// CHECK:STDOUT: ; ModuleID = '/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/comparison.carbon'
// CHECK:STDOUT: source_filename = "/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/comparison.carbon"
// CHECK:STDOUT: ; ModuleID = '/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon'
// CHECK:STDOUT: source_filename = "/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/bitwise.carbon"
// CHECK:STDOUT: ; ModuleID = '/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon'
// CHECK:STDOUT: source_filename = "/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/arithmetic.carbon"
// CHECK:STDOUT: ; ModuleID = '/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types.carbon'
// CHECK:STDOUT: source_filename = "/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types.carbon"
// CHECK:STDOUT: ; ModuleID = '/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/bool.carbon'
// CHECK:STDOUT: source_filename = "/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/bool.carbon"
// CHECK:STDOUT: ; ModuleID = '/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon'
// CHECK:STDOUT: source_filename = "/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/i32.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op(ptr %self, i32 %other) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_17.3 = load i32, ptr %self, align 4
// CHECK:STDOUT:   %int.sadd = add i32 %.loc18_17.3, %other
// CHECK:STDOUT:   store i32 %int.sadd, ptr %self, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op.1(ptr %self) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @Op(ptr %self, i32 1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op.2(ptr %self, i32 %other) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc34_17.3 = load i32, ptr %self, align 4
// CHECK:STDOUT:   %int.and = and i32 %.loc34_17.3, %other
// CHECK:STDOUT:   store i32 %int.and, ptr %self, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op.3(ptr %self, i32 %other) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc48_17.3 = load i32, ptr %self, align 4
// CHECK:STDOUT:   %int.or = or i32 %.loc48_17.3, %other
// CHECK:STDOUT:   store i32 %int.or, ptr %self, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op.4(ptr %self, i32 %other) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc58_17.3 = load i32, ptr %self, align 4
// CHECK:STDOUT:   %int.xor = xor i32 %.loc58_17.3, %other
// CHECK:STDOUT:   store i32 %int.xor, ptr %self, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op.5(ptr %self, i32 %other) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc68_17.3 = load i32, ptr %self, align 4
// CHECK:STDOUT:   %int.sdiv = sdiv i32 %.loc68_17.3, %other
// CHECK:STDOUT:   store i32 %int.sdiv, ptr %self, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op.6(ptr %self, i32 %other) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc83_17.3 = load i32, ptr %self, align 4
// CHECK:STDOUT:   %int.left_shift = shl i32 %.loc83_17.3, %other
// CHECK:STDOUT:   store i32 %int.left_shift, ptr %self, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op.7(ptr %self, i32 %other) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc93_17.3 = load i32, ptr %self, align 4
// CHECK:STDOUT:   %int.smod = srem i32 %.loc93_17.3, %other
// CHECK:STDOUT:   store i32 %int.smod, ptr %self, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op.8(ptr %self, i32 %other) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc103_17.3 = load i32, ptr %self, align 4
// CHECK:STDOUT:   %int.smul = mul i32 %.loc103_17.3, %other
// CHECK:STDOUT:   store i32 %int.smul, ptr %self, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op.9(ptr %self, i32 %other) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc125_17.3 = load i32, ptr %self, align 4
// CHECK:STDOUT:   %int.right_shift = ashr i32 %.loc125_17.3, %other
// CHECK:STDOUT:   store i32 %int.right_shift, ptr %self, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op.10(ptr %self, i32 %other) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc135_17.3 = load i32, ptr %self, align 4
// CHECK:STDOUT:   %int.ssub = sub i32 %.loc135_17.3, %other
// CHECK:STDOUT:   store i32 %int.ssub, ptr %self, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op.11(ptr %self) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @Op.10(ptr %self, i32 1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT: ; ModuleID = '/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators.carbon'
// CHECK:STDOUT: source_filename = "/usr/local/google/home/blaikie/.cache/bazel/_bazel_blaikie/1d654ce96f5dc056ed27ac2156be47b1/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators.carbon"
// CHECK:STDOUT: ; ModuleID = 'nodebug.carbon'
// CHECK:STDOUT: source_filename = "nodebug.carbon"
