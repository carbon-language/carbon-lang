// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --optimize=speed foo.carbon --target=x86_64-unknown-linux-gnu --phase=optimize --dump-llvm-ir --exclude-dump-file-prefix=%{core}
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/optimize/optimize_speed.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/optimize/optimize_speed.carbon

// --- foo.carbon

import Core library "range";

fn Rand() -> i32;

fn NoInlineWithOz() -> i32 {
  return Rand() + Rand();
}

fn CallNoInlineWithOptSize() -> i32 {
  // Should not be inlined with --optimize=size, because the body is larger than the call.
  return NoInlineWithOz();
}

fn VectorizedWithOptSpeed(a: array(i32, 65536)*) {
  // Should be vectorized with --optimize=speed, but not in other modes.
  var n: i32 = 0;
  while (n < 65536) {
    (*a)[n] *= 2;
    ++n;
  }
}

// CHECK:STDOUT: ; ModuleID = 'foo.carbon'
// CHECK:STDOUT: source_filename = "foo.carbon"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CRand.Main() local_unnamed_addr
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CNoInlineWithOz.Main() local_unnamed_addr #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Rand.call.loc7_15 = tail call i32 @_CRand.Main() #0, !dbg !7
// CHECK:STDOUT:   %Rand.call.loc7_24 = tail call i32 @_CRand.Main() #0, !dbg !8
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %Rand.call.loc7_24, %Rand.call.loc7_15, !dbg !7
// CHECK:STDOUT:   ret i32 %Int.as.AddWith.impl.Op.call, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallNoInlineWithOptSize.Main() local_unnamed_addr #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Rand.call.loc7_15.i = tail call i32 @_CRand.Main() #0, !dbg !11
// CHECK:STDOUT:   %Rand.call.loc7_24.i = tail call i32 @_CRand.Main() #0, !dbg !13
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call.i = add i32 %Rand.call.loc7_24.i, %Rand.call.loc7_15.i, !dbg !11
// CHECK:STDOUT:   ret i32 %Int.as.AddWith.impl.Op.call.i, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
// CHECK:STDOUT: define void @_CVectorizedWithOptSpeed.Main(ptr captures(none) %a) local_unnamed_addr #1 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br label %vector.body, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: vector.body:                                      ; preds = %vector.body, %entry
// CHECK:STDOUT:   %index = phi i32 [ 0, %entry ], [ %index.next.1, %vector.body ], !dbg !17
// CHECK:STDOUT:   %0 = zext nneg i32 %index to i64, !dbg !23
// CHECK:STDOUT:   %1 = getelementptr inbounds nuw i32, ptr %a, i64 %0, !dbg !23
// CHECK:STDOUT:   %2 = getelementptr inbounds nuw i8, ptr %1, i64 16, !dbg !23
// CHECK:STDOUT:   %wide.load = load <4 x i32>, ptr %1, align 4, !dbg !23
// CHECK:STDOUT:   %wide.load4 = load <4 x i32>, ptr %2, align 4, !dbg !23
// CHECK:STDOUT:   %3 = shl <4 x i32> %wide.load, splat (i32 1), !dbg !23
// CHECK:STDOUT:   %4 = shl <4 x i32> %wide.load4, splat (i32 1), !dbg !23
// CHECK:STDOUT:   store <4 x i32> %3, ptr %1, align 4, !dbg !23
// CHECK:STDOUT:   store <4 x i32> %4, ptr %2, align 4, !dbg !23
// CHECK:STDOUT:   %5 = zext nneg i32 %index to i64, !dbg !23
// CHECK:STDOUT:   %6 = getelementptr inbounds nuw i32, ptr %a, i64 %5, !dbg !23
// CHECK:STDOUT:   %7 = getelementptr inbounds nuw i8, ptr %6, i64 32, !dbg !23
// CHECK:STDOUT:   %8 = getelementptr inbounds nuw i8, ptr %6, i64 48, !dbg !23
// CHECK:STDOUT:   %wide.load.1 = load <4 x i32>, ptr %7, align 4, !dbg !23
// CHECK:STDOUT:   %wide.load4.1 = load <4 x i32>, ptr %8, align 4, !dbg !23
// CHECK:STDOUT:   %9 = shl <4 x i32> %wide.load.1, splat (i32 1), !dbg !23
// CHECK:STDOUT:   %10 = shl <4 x i32> %wide.load4.1, splat (i32 1), !dbg !23
// CHECK:STDOUT:   store <4 x i32> %9, ptr %7, align 4, !dbg !23
// CHECK:STDOUT:   store <4 x i32> %10, ptr %8, align 4, !dbg !23
// CHECK:STDOUT:   %index.next.1 = add nuw nsw i32 %index, 16, !dbg !17
// CHECK:STDOUT:   %11 = icmp eq i32 %index.next.1, 65536, !dbg !16
// CHECK:STDOUT:   br i1 %11, label %while.done, label %vector.body, !dbg !16, !llvm.loop !24
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %vector.body
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder i32 %index, { 1, 0, 2 }
// CHECK:STDOUT:   uselistorder ptr %1, { 1, 2, 0 }
// CHECK:STDOUT:   uselistorder ptr %2, { 1, 0 }
// CHECK:STDOUT:   uselistorder ptr %8, { 1, 0 }
// CHECK:STDOUT:   uselistorder i32 %index.next.1, { 1, 0 }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nofree norecurse nosync nounwind memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "foo.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "NoInlineWithOz", linkageName: "_CNoInlineWithOz.Main", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 7, column: 19, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "CallNoInlineWithOptSize", linkageName: "_CCallNoInlineWithOptSize.Main", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 7, column: 10, scope: !4, inlinedAt: !12)
// CHECK:STDOUT: !12 = distinct !DILocation(line: 12, column: 10, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 7, column: 19, scope: !4, inlinedAt: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 12, column: 3, scope: !10)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "VectorizedWithOptSpeed", linkageName: "_CVectorizedWithOptSpeed.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 18, column: 9, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 275, column: 3, scope: !18, inlinedAt: !20)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk.Int.Core:AddAssignWith.Core.5dfb78ae56583d8e", scope: null, file: !19, line: 275, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !20 = distinct !DILocation(line: 341, column: 5, scope: !21, inlinedAt: !22)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !19, line: 339, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = distinct !DILocation(line: 20, column: 5, scope: !15)
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 5, scope: !15)
// CHECK:STDOUT: !24 = distinct !{!24, !25, !26}
// CHECK:STDOUT: !25 = !{!"llvm.loop.isvectorized", i32 1}
// CHECK:STDOUT: !26 = !{!"llvm.loop.unroll.runtime.disable"}
// CHECK:STDOUT: !27 = !DILocation(line: 15, column: 1, scope: !15)
