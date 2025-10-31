// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --optimize=debug foo.carbon --target=x86_64-unknown-linux-gnu --phase=optimize --dump-llvm-ir --exclude-dump-file-prefix=%{core}
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/optimize/optimize_debug.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/optimize/optimize_debug.carbon

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
// CHECK:STDOUT:   br label %while.body, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %entry, %while.body
// CHECK:STDOUT:   %n.var.03 = phi i32 [ 0, %entry ], [ %1, %while.body ]
// CHECK:STDOUT:   %0 = zext nneg i32 %n.var.03 to i64, !dbg !17
// CHECK:STDOUT:   %.loc19_11.array.index = getelementptr inbounds nuw i32, ptr %a, i64 %0, !dbg !17
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.call = load i32, ptr %.loc19_11.array.index, align 4, !dbg !17
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.call1 = shl i32 %Int.as.MulAssignWith.impl.Op.call, 1, !dbg !17
// CHECK:STDOUT:   store i32 %Int.as.MulAssignWith.impl.Op.call1, ptr %.loc19_11.array.index, align 4, !dbg !17
// CHECK:STDOUT:   %1 = add nuw nsw i32 %n.var.03, 1, !dbg !18
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.call = icmp samesign ult i32 %n.var.03, 65535, !dbg !24
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Less.call, label %while.body, label %while.done, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.body
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder label %while.body, { 1, 0 }
// CHECK:STDOUT:   uselistorder i32 %n.var.03, { 0, 2, 1 }
// CHECK:STDOUT:   uselistorder ptr %.loc19_11.array.index, { 1, 0 }
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
// CHECK:STDOUT: !17 = !DILocation(line: 19, column: 5, scope: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 275, column: 3, scope: !19, inlinedAt: !21)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk.Int.Core:AddAssignWith.Core.5dfb78ae56583d8e", scope: null, file: !20, line: 275, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !20 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DILocation(line: 341, column: 5, scope: !22, inlinedAt: !23)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !20, line: 339, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !23 = distinct !DILocation(line: 20, column: 5, scope: !15)
// CHECK:STDOUT: !24 = !DILocation(line: 18, column: 10, scope: !15)
// CHECK:STDOUT: !25 = !DILocation(line: 15, column: 1, scope: !15)
