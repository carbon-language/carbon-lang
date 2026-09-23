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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'foo.carbon'
// CHECK:STDOUT: source_filename = "foo.carbon"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CRand.Main() local_unnamed_addr
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CNoInlineWithOz.Main() local_unnamed_addr #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Rand.call.loc4_15 = tail call i32 @_CRand.Main() #0, !dbg !13
// CHECK:STDOUT:   %Rand.call.loc4_24 = tail call i32 @_CRand.Main() #0, !dbg !14
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %Rand.call.loc4_24, %Rand.call.loc4_15, !dbg !13
// CHECK:STDOUT:   ret i32 %Int.as.AddWith.impl.Op.call, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallNoInlineWithOptSize.Main() local_unnamed_addr #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Rand.call.loc4_15.i = tail call i32 @_CRand.Main() #0, !dbg !89
// CHECK:STDOUT:   %Rand.call.loc4_24.i = tail call i32 @_CRand.Main() #0, !dbg !90
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call.i = add i32 %Rand.call.loc4_24.i, %Rand.call.loc4_15.i, !dbg !89
// CHECK:STDOUT:   ret i32 %Int.as.AddWith.impl.Op.call.i, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
// CHECK:STDOUT: define void @_CVectorizedWithOptSpeed.Main(ptr nofree captures(none) %a) local_unnamed_addr #1 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br label %while.body, !dbg !26
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %entry, %while.body
// CHECK:STDOUT:   %n.var.03 = phi i32 [ 0, %entry ], [ %1, %while.body ]
// CHECK:STDOUT:   %0 = zext nneg i32 %n.var.03 to i64, !dbg !31
// CHECK:STDOUT:   %.loc16_11.array.index = getelementptr inbounds nuw [4 x i8], ptr %a, i64 %0, !dbg !31
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.call = load i32, ptr %.loc16_11.array.index, align 4, !dbg !31
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.call1 = shl i32 %Int.as.MulAssignWith.impl.Op.call, 1, !dbg !31
// CHECK:STDOUT:   store i32 %Int.as.MulAssignWith.impl.Op.call1, ptr %.loc16_11.array.index, align 4, !dbg !31
// CHECK:STDOUT:   %1 = add nuw nsw i32 %n.var.03, 1, !dbg !100
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.call = icmp samesign ult i32 %n.var.03, 65535, !dbg !27
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Less.call, label %while.body, label %while.done, !dbg !26
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.body
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder label %while.body, { 1, 0 }
// CHECK:STDOUT:   uselistorder i32 %n.var.03, { 0, 2, 1 }
// CHECK:STDOUT:   uselistorder ptr %.loc16_11.array.index, { 1, 0 }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) local_unnamed_addr #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) local_unnamed_addr #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) local_unnamed_addr #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nofree norecurse nosync nounwind memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!1}
// CHECK:STDOUT: !llvm.module.flags = !{!83, !84}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !DIFile(filename: "foo.carbon", directory: "")
// CHECK:STDOUT: !1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !0, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !3 = !{!2}
// CHECK:STDOUT: !4 = !DISubroutineType(types: !3)
// CHECK:STDOUT: !5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !6 = !{null, !5}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !{null, !2}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "NoInlineWithOz", linkageName: "_CNoInlineWithOz.Main", scope: null, file: !0, line: 3, type: !4, spFlags: DISPFlagDefinition, unit: !1)
// CHECK:STDOUT: !13 = !DILocation(line: 4, column: 10, scope: !10)
// CHECK:STDOUT: !14 = !DILocation(line: 4, column: 19, scope: !10)
// CHECK:STDOUT: !15 = !DILocation(line: 4, column: 3, scope: !10)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "CallNoInlineWithOptSize", linkageName: "_CCallNoInlineWithOptSize.Main", scope: null, file: !0, line: 7, type: !4, spFlags: DISPFlagDefinition, unit: !1)
// CHECK:STDOUT: !20 = !DILocation(line: 9, column: 3, scope: !16)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "VectorizedWithOptSpeed", linkageName: "_CVectorizedWithOptSpeed.Main", scope: null, file: !0, line: 12, type: !7, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !36)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !21, type: !5)
// CHECK:STDOUT: !26 = !DILocation(line: 15, column: 9, scope: !21)
// CHECK:STDOUT: !27 = !DILocation(line: 15, column: 10, scope: !21)
// CHECK:STDOUT: !31 = !DILocation(line: 16, column: 5, scope: !21)
// CHECK:STDOUT: !35 = !DILocation(line: 12, column: 1, scope: !21)
// CHECK:STDOUT: !36 = !{!22}
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !0, line: 14, type: !9, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !40)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !2)
// CHECK:STDOUT: !39 = !DILocation(line: 14, column: 3, scope: !37)
// CHECK:STDOUT: !40 = !{!38}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !0, line: 14, type: !9, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !44)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !2)
// CHECK:STDOUT: !43 = !DILocation(line: 14, column: 3, scope: !41)
// CHECK:STDOUT: !44 = !{!42}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !0, line: 14, type: !9, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !2)
// CHECK:STDOUT: !47 = !DILocation(line: 14, column: 3, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !49, line: 396, type: !9, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !57)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !2)
// CHECK:STDOUT: !54 = !{null, !2, !2}
// CHECK:STDOUT: !55 = !DISubroutineType(types: !54)
// CHECK:STDOUT: !57 = !{!51}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895", scope: null, file: !49, line: 331, type: !55, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !65)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !2)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 2, scope: !58, type: !2)
// CHECK:STDOUT: !65 = !{!59, !60}
// CHECK:STDOUT: !83 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !84 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !88 = distinct !DILocation(line: 9, column: 10, scope: !16)
// CHECK:STDOUT: !89 = distinct !DILocation(line: 4, column: 10, scope: !10, inlinedAt: !88)
// CHECK:STDOUT: !90 = distinct !DILocation(line: 4, column: 19, scope: !10, inlinedAt: !88)
// CHECK:STDOUT: !98 = distinct !DILocation(line: 17, column: 5, scope: !21)
// CHECK:STDOUT: !99 = distinct !DILocation(line: 398, column: 5, scope: !50, inlinedAt: !98)
// CHECK:STDOUT: !100 = distinct !DILocation(line: 331, column: 3, scope: !58, inlinedAt: !99)
// CHECK:STDOUT:
