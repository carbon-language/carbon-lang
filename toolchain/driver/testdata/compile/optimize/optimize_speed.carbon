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
// CHECK:STDOUT:   %Rand.call.loc7_15 = tail call i32 @_CRand.Main() #0, !dbg !8
// CHECK:STDOUT:   %Rand.call.loc7_24 = tail call i32 @_CRand.Main() #0, !dbg !9
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %Rand.call.loc7_24, %Rand.call.loc7_15, !dbg !8
// CHECK:STDOUT:   ret i32 %Int.as.AddWith.impl.Op.call, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallNoInlineWithOptSize.Main() local_unnamed_addr #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Rand.call.loc7_15.i = tail call i32 @_CRand.Main() #0, !dbg !12
// CHECK:STDOUT:   %Rand.call.loc7_24.i = tail call i32 @_CRand.Main() #0, !dbg !14
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call.i = add i32 %Rand.call.loc7_24.i, %Rand.call.loc7_15.i, !dbg !12
// CHECK:STDOUT:   ret i32 %Int.as.AddWith.impl.Op.call.i, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
// CHECK:STDOUT: define void @_CVectorizedWithOptSpeed.Main(ptr nofree captures(none) %a) local_unnamed_addr #1 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br label %vector.body, !dbg !22
// CHECK:STDOUT:
// CHECK:STDOUT: vector.body:                                      ; preds = %vector.body, %entry
// CHECK:STDOUT:   %index = phi i32 [ 0, %entry ], [ %index.next.1, %vector.body ], !dbg !23
// CHECK:STDOUT:   %0 = zext nneg i32 %index to i64, !dbg !38
// CHECK:STDOUT:   %1 = getelementptr inbounds nuw [4 x i8], ptr %a, i64 %0, !dbg !38
// CHECK:STDOUT:   %2 = getelementptr inbounds nuw i8, ptr %1, i64 16, !dbg !38
// CHECK:STDOUT:   %wide.load = load <4 x i32>, ptr %1, align 4, !dbg !38
// CHECK:STDOUT:   %wide.load4 = load <4 x i32>, ptr %2, align 4, !dbg !38
// CHECK:STDOUT:   %3 = shl <4 x i32> %wide.load, splat (i32 1), !dbg !38
// CHECK:STDOUT:   %4 = shl <4 x i32> %wide.load4, splat (i32 1), !dbg !38
// CHECK:STDOUT:   store <4 x i32> %3, ptr %1, align 4, !dbg !38
// CHECK:STDOUT:   store <4 x i32> %4, ptr %2, align 4, !dbg !38
// CHECK:STDOUT:   %5 = zext nneg i32 %index to i64, !dbg !38
// CHECK:STDOUT:   %6 = getelementptr inbounds nuw [4 x i8], ptr %a, i64 %5, !dbg !38
// CHECK:STDOUT:   %7 = getelementptr inbounds nuw i8, ptr %6, i64 32, !dbg !38
// CHECK:STDOUT:   %8 = getelementptr inbounds nuw i8, ptr %6, i64 48, !dbg !38
// CHECK:STDOUT:   %wide.load.1 = load <4 x i32>, ptr %7, align 4, !dbg !38
// CHECK:STDOUT:   %wide.load4.1 = load <4 x i32>, ptr %8, align 4, !dbg !38
// CHECK:STDOUT:   %9 = shl <4 x i32> %wide.load.1, splat (i32 1), !dbg !38
// CHECK:STDOUT:   %10 = shl <4 x i32> %wide.load4.1, splat (i32 1), !dbg !38
// CHECK:STDOUT:   store <4 x i32> %9, ptr %7, align 4, !dbg !38
// CHECK:STDOUT:   store <4 x i32> %10, ptr %8, align 4, !dbg !38
// CHECK:STDOUT:   %index.next.1 = add nuw nsw i32 %index, 16, !dbg !23
// CHECK:STDOUT:   %11 = icmp eq i32 %index.next.1, 65536, !dbg !22
// CHECK:STDOUT:   br i1 %11, label %while.done, label %vector.body, !dbg !22, !llvm.loop !39
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %vector.body
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder i32 %index, { 1, 0, 2 }
// CHECK:STDOUT:   uselistorder ptr %1, { 1, 2, 0 }
// CHECK:STDOUT:   uselistorder ptr %2, { 1, 0 }
// CHECK:STDOUT:   uselistorder ptr %8, { 1, 0 }
// CHECK:STDOUT:   uselistorder i32 %index.next.1, { 1, 0 }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) local_unnamed_addr #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nofree norecurse nosync nounwind memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "foo.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "NoInlineWithOz", linkageName: "_CNoInlineWithOz.Main", scope: null, file: !1, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DILocation(line: 7, column: 10, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 7, column: 19, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "CallNoInlineWithOptSize", linkageName: "_CCallNoInlineWithOptSize.Main", scope: null, file: !1, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !12 = !DILocation(line: 7, column: 10, scope: !4, inlinedAt: !13)
// CHECK:STDOUT: !13 = distinct !DILocation(line: 12, column: 10, scope: !11)
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 19, scope: !4, inlinedAt: !13)
// CHECK:STDOUT: !15 = !DILocation(line: 12, column: 3, scope: !11)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "VectorizedWithOptSpeed", linkageName: "_CVectorizedWithOptSpeed.Main", scope: null, file: !1, line: 15, type: !17, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{null, !19}
// CHECK:STDOUT: !19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !16, type: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 18, column: 9, scope: !16)
// CHECK:STDOUT: !23 = !DILocation(line: 331, column: 3, scope: !24, inlinedAt: !31)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.6f9e8c1abf8bb3ea.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.44914b6ff6901661", scope: null, file: !25, line: 331, type: !26, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !28)
// CHECK:STDOUT: !25 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !26 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !27 = !{null, !7, !7}
// CHECK:STDOUT: !28 = !{!29, !30}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !24, type: !7)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 2, scope: !24, type: !7)
// CHECK:STDOUT: !31 = distinct !DILocation(line: 398, column: 5, scope: !32, inlinedAt: !37)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !25, line: 396, type: !33, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !35)
// CHECK:STDOUT: !33 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !34 = !{null, !7}
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !32, type: !7)
// CHECK:STDOUT: !37 = distinct !DILocation(line: 20, column: 5, scope: !16)
// CHECK:STDOUT: !38 = !DILocation(line: 19, column: 5, scope: !16)
// CHECK:STDOUT: !39 = distinct !{!39, !40, !41}
// CHECK:STDOUT: !40 = !{!"llvm.loop.isvectorized", i32 1}
// CHECK:STDOUT: !41 = !{!"llvm.loop.unroll.runtime.disable"}
// CHECK:STDOUT: !42 = !DILocation(line: 15, column: 1, scope: !16)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !1, line: 17, type: !33, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !44)
// CHECK:STDOUT: !44 = !{!45}
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !43, type: !7)
// CHECK:STDOUT: !46 = !DILocation(line: 17, column: 3, scope: !43)
