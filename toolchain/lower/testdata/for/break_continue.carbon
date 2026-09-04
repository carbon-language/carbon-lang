// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/for/break_continue.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/for/break_continue.carbon

fn F() -> bool;
fn G() -> bool;
fn H();

fn For() {
  for (_: i32 in Core.Range(100)) {
    if (F()) { break; }
    if (G()) { continue; }
    H();
  }
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'break_continue.carbon'
// CHECK:STDOUT: source_filename = "break_continue.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CH.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_32.1.temp = alloca { i32, i32 }, align 4, !dbg !7
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %.loc18_33.1.temp = alloca <{ i32, i1 }>, align 4, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_32.1.temp), !dbg !7
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc18_32.1.temp, i32 100), !dbg !7
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc18_32.1.temp), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %if.else.loc20, %if.then.loc20, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc18_33.1.temp, ptr %.loc18_32.1.temp, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.a5214d4ceffb522b(ptr %.loc18_33.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.a5214d4ceffb522b(ptr %.loc18_33.1.temp), !dbg !8
// CHECK:STDOUT:   %F.call = call i1 @_CF.Main(), !dbg !9
// CHECK:STDOUT:   br i1 %F.call, label %if.then.loc19, label %if.else.loc19, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc19:                                    ; preds = %for.body
// CHECK:STDOUT:   br label %for.done, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc19:                                    ; preds = %for.body
// CHECK:STDOUT:   %G.call = call i1 @_CG.Main(), !dbg !12
// CHECK:STDOUT:   br i1 %G.call, label %if.then.loc20, label %if.else.loc20, !dbg !13
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   call void @"_COp.891c4207e57d0f40:core.Destroy.Core"(ptr %.loc18_33.1.temp), !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !15
// CHECK:STDOUT:   call void @"_COp.891c4207e57d0f40:core.Destroy.Core"(ptr %.loc18_33.1.temp), !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %if.then.loc19, %for.next
// CHECK:STDOUT:   call void @"_COp.891c4207e57d0f40:core.Destroy.Core"(ptr %.loc18_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.c9184c281ed4335d:core.Destroy.Core"(ptr %.loc18_32.1.temp), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0239a84bcf0315a7:core.Destroy.Core"(ptr %self) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ad05ff27ec7895c0:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ae16cb2db38d93af:core.Destroy.Core"(ptr %self) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.891c4207e57d0f40:core.Destroy.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e933d5ff8212e171:core.Destroy.Core"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.c9184c281ed4335d:core.Destroy.Core"(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !58
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !58
// CHECK:STDOUT:   ret i32 %1, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !60 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !66
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !67
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !66
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !68
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !68
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !69
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !69
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !70
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !71
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !72
// CHECK:STDOUT:   call void @_CSome.Optional.Core.a5214d4ceffb522b(ptr %return, i32 %7), !dbg !73
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %1), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.a5214d4ceffb522b(ptr %return), !dbg !75
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %1), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.a5214d4ceffb522b(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %self), !dbg !83
// CHECK:STDOUT:   ret i1 %1, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.a5214d4ceffb522b(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %self), !dbg !88
// CHECK:STDOUT:   ret i32 %1, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !90 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 1), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.a5214d4ceffb522b(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !96 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %return, i32 %value), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.a5214d4ceffb522b(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !103 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %return), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %value) #0 !dbg !108 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !111
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !111
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !111
// CHECK:STDOUT:   ret i1 %2, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %value) #0 !dbg !113 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !116
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !116
// CHECK:STDOUT:   ret i32 %1, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 %other) #2 !dbg !118 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %other), !dbg !124
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !125
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !125
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !125
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !126 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !129
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !129
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !130
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !130
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !132 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !133
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !133
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !135 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !140
// CHECK:STDOUT:   ret i32 %1, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !142 {
// CHECK:STDOUT:   ret i32 %self, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.891c4207e57d0f40:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "break_continue.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !1, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 18, column: 18, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 18, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 19, column: 9, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 19, column: 8, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 19, column: 16, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 20, column: 9, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 20, column: 8, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 20, column: 16, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 21, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 17, column: 1, scope: !4)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1d6a7672bf2f14c6:core.Destroy.Core", scope: null, file: !1, line: 18, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !22)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{null, !21}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !18, type: !21)
// CHECK:STDOUT: !24 = !DILocation(line: 18, column: 7, scope: !18)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0239a84bcf0315a7:core.Destroy.Core", scope: null, file: !1, line: 18, type: !26, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !29)
// CHECK:STDOUT: !26 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !27 = !{null, !28}
// CHECK:STDOUT: !28 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !25, type: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 18, column: 7, scope: !25)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ad05ff27ec7895c0:core.Destroy.Core", scope: null, file: !1, line: 18, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !21)
// CHECK:STDOUT: !35 = !DILocation(line: 18, column: 7, scope: !32)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ae16cb2db38d93af:core.Destroy.Core", scope: null, file: !1, line: 18, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !37)
// CHECK:STDOUT: !37 = !{!38}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !36, type: !21)
// CHECK:STDOUT: !39 = !DILocation(line: 18, column: 7, scope: !36)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Op", linkageName: "_COp.891c4207e57d0f40:core.Destroy.Core", scope: null, file: !1, line: 18, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !21)
// CHECK:STDOUT: !43 = !DILocation(line: 18, column: 7, scope: !40)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e933d5ff8212e171:core.Destroy.Core", scope: null, file: !1, line: 18, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !45)
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !44, type: !21)
// CHECK:STDOUT: !47 = !DILocation(line: 18, column: 18, scope: !44)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c9184c281ed4335d:core.Destroy.Core", scope: null, file: !1, line: 18, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !49)
// CHECK:STDOUT: !49 = !{!50}
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !48, type: !21)
// CHECK:STDOUT: !51 = !DILocation(line: 18, column: 18, scope: !48)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !53, line: 19, type: !54, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !56)
// CHECK:STDOUT: !53 = !DIFile(filename: "{{.*}}/prelude/range.carbon", directory: "")
// CHECK:STDOUT: !54 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !55 = !{!28, !21}
// CHECK:STDOUT: !56 = !{!57}
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !52, type: !21)
// CHECK:STDOUT: !58 = !DILocation(line: 19, column: 43, scope: !52)
// CHECK:STDOUT: !59 = !DILocation(line: 19, column: 36, scope: !52)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !53, line: 20, type: !61, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DISubroutineType(types: !62)
// CHECK:STDOUT: !62 = !{!21, !21, !21}
// CHECK:STDOUT: !63 = !{!64, !65}
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !60, type: !21)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 2, scope: !60, type: !21)
// CHECK:STDOUT: !66 = !DILocation(line: 21, column: 7, scope: !60)
// CHECK:STDOUT: !67 = !DILocation(line: 21, column: 27, scope: !60)
// CHECK:STDOUT: !68 = !DILocation(line: 22, column: 19, scope: !60)
// CHECK:STDOUT: !69 = !DILocation(line: 22, column: 11, scope: !60)
// CHECK:STDOUT: !70 = !DILocation(line: 22, column: 10, scope: !60)
// CHECK:STDOUT: !71 = !DILocation(line: 23, column: 9, scope: !60)
// CHECK:STDOUT: !72 = !DILocation(line: 24, column: 38, scope: !60)
// CHECK:STDOUT: !73 = !DILocation(line: 24, column: 16, scope: !60)
// CHECK:STDOUT: !74 = !DILocation(line: 24, column: 9, scope: !60)
// CHECK:STDOUT: !75 = !DILocation(line: 26, column: 16, scope: !60)
// CHECK:STDOUT: !76 = !DILocation(line: 26, column: 9, scope: !60)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.a5214d4ceffb522b", scope: null, file: !78, line: 36, type: !79, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !81)
// CHECK:STDOUT: !78 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !79 = !DISubroutineType(types: !80)
// CHECK:STDOUT: !80 = !{!21, !21}
// CHECK:STDOUT: !81 = !{!82}
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !77, type: !21)
// CHECK:STDOUT: !83 = !DILocation(line: 37, column: 12, scope: !77)
// CHECK:STDOUT: !84 = !DILocation(line: 37, column: 5, scope: !77)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.a5214d4ceffb522b", scope: null, file: !78, line: 39, type: !54, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !86)
// CHECK:STDOUT: !86 = !{!87}
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !85, type: !21)
// CHECK:STDOUT: !88 = !DILocation(line: 40, column: 12, scope: !85)
// CHECK:STDOUT: !89 = !DILocation(line: 40, column: 5, scope: !85)
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !91, line: 396, type: !26, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !92)
// CHECK:STDOUT: !91 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !92 = !{!93}
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !90, type: !28)
// CHECK:STDOUT: !94 = !DILocation(line: 398, column: 5, scope: !90)
// CHECK:STDOUT: !95 = !DILocation(line: 396, column: 3, scope: !90)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.a5214d4ceffb522b", scope: null, file: !78, line: 33, type: !97, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !99)
// CHECK:STDOUT: !97 = !DISubroutineType(types: !98)
// CHECK:STDOUT: !98 = !{!21, !28}
// CHECK:STDOUT: !99 = !{!100}
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !96, type: !28)
// CHECK:STDOUT: !101 = !DILocation(line: 34, column: 12, scope: !96)
// CHECK:STDOUT: !102 = !DILocation(line: 34, column: 5, scope: !96)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.a5214d4ceffb522b", scope: null, file: !78, line: 30, type: !104, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !104 = !DISubroutineType(types: !105)
// CHECK:STDOUT: !105 = !{!21}
// CHECK:STDOUT: !106 = !DILocation(line: 31, column: 12, scope: !103)
// CHECK:STDOUT: !107 = !DILocation(line: 31, column: 5, scope: !103)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !78, line: 141, type: !79, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !109)
// CHECK:STDOUT: !109 = !{!110}
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !108, type: !21)
// CHECK:STDOUT: !111 = !DILocation(line: 142, column: 12, scope: !108)
// CHECK:STDOUT: !112 = !DILocation(line: 142, column: 5, scope: !108)
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !78, line: 144, type: !54, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !114)
// CHECK:STDOUT: !114 = !{!115}
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !113, type: !21)
// CHECK:STDOUT: !116 = !DILocation(line: 145, column: 12, scope: !113)
// CHECK:STDOUT: !117 = !DILocation(line: 145, column: 5, scope: !113)
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895", scope: null, file: !91, line: 331, type: !119, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !121)
// CHECK:STDOUT: !119 = !DISubroutineType(types: !120)
// CHECK:STDOUT: !120 = !{null, !28, !28}
// CHECK:STDOUT: !121 = !{!122, !123}
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !118, type: !28)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 2, scope: !118, type: !28)
// CHECK:STDOUT: !124 = !DILocation(line: 0, scope: !118)
// CHECK:STDOUT: !125 = !DILocation(line: 331, column: 3, scope: !118)
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !78, line: 131, type: !97, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !127)
// CHECK:STDOUT: !127 = !{!128}
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !126, type: !28)
// CHECK:STDOUT: !129 = !DILocation(line: 137, column: 5, scope: !126)
// CHECK:STDOUT: !130 = !DILocation(line: 138, column: 5, scope: !126)
// CHECK:STDOUT: !131 = !DILocation(line: 139, column: 5, scope: !126)
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !78, line: 125, type: !104, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !133 = !DILocation(line: 128, column: 5, scope: !132)
// CHECK:STDOUT: !134 = !DILocation(line: 129, column: 5, scope: !132)
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !91, line: 64, type: !136, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !138)
// CHECK:STDOUT: !136 = !DISubroutineType(types: !137)
// CHECK:STDOUT: !137 = !{!28, !28}
// CHECK:STDOUT: !138 = !{!139}
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !135, type: !28)
// CHECK:STDOUT: !140 = !DILocation(line: 66, column: 17, scope: !135)
// CHECK:STDOUT: !141 = !DILocation(line: 66, column: 5, scope: !135)
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !91, line: 59, type: !136, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !143)
// CHECK:STDOUT: !143 = !{!144}
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !142, type: !28)
// CHECK:STDOUT: !145 = !DILocation(line: 59, column: 28, scope: !142)
// CHECK:STDOUT:
