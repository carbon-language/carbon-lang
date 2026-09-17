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
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_32.1.temp = alloca { i32, i32 }, align 4, !dbg !20
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !21
// CHECK:STDOUT:   %.loc18_33.1.temp = alloca <{ i32, i1 }>, align 4, !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_32.1.temp), !dbg !20
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc18_32.1.temp, i32 100), !dbg !20
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc18_32.1.temp), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !21
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !21
// CHECK:STDOUT:   br label %for.next, !dbg !21
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %if.else.loc20, %if.then.loc20, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_33.1.temp), !dbg !21
// CHECK:STDOUT:   call void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc18_33.1.temp, ptr %.loc18_32.1.temp, ptr %var), !dbg !21
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.d56e393be6e32080(ptr %.loc18_33.1.temp), !dbg !21
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !21
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.d56e393be6e32080(ptr %.loc18_33.1.temp), !dbg !21
// CHECK:STDOUT:   %F.call = call i1 @_CF.Main(), !dbg !28
// CHECK:STDOUT:   br i1 %F.call, label %if.then.loc19, label %if.else.loc19, !dbg !29
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc19:                                    ; preds = %for.body
// CHECK:STDOUT:   br label %for.done, !dbg !30
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc19:                                    ; preds = %for.body
// CHECK:STDOUT:   %G.call = call i1 @_CG.Main(), !dbg !31
// CHECK:STDOUT:   br i1 %G.call, label %if.then.loc20, label %if.else.loc20, !dbg !32
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.31a8c47192dc2066(ptr %.loc18_33.1.temp), !dbg !21
// CHECK:STDOUT:   br label %for.next, !dbg !33
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !34
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.31a8c47192dc2066(ptr %.loc18_33.1.temp), !dbg !21
// CHECK:STDOUT:   br label %for.next, !dbg !35
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %if.then.loc19, %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.31a8c47192dc2066(ptr %.loc18_33.1.temp), !dbg !21
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %var), !dbg !21
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5a42872629477478(ptr %.loc18_32.1.temp), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.32ec81fcfb7ba5dc:core.Destroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c3be8338012ae399:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c800cdf671d92dcc:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f5c18a58c263f72b:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.e933d5ff8212e171:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c9184c281ed4335d:core.Destroy.Core"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !73
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !73
// CHECK:STDOUT:   ret i32 %1, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !76 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !83
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !83
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !85
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !83
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !88
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !88
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !87
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !87
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !89
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !92
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !93
// CHECK:STDOUT:   call void @_CSome.Optional.Core.d56e393be6e32080(ptr %return, i32 %7), !dbg !94
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %1), !dbg !83
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.d56e393be6e32080(ptr %return), !dbg !96
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %1), !dbg !83
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.d56e393be6e32080(ptr %self) #0 !dbg !100 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %self), !dbg !106
// CHECK:STDOUT:   ret i1 %1, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.d56e393be6e32080(ptr %self) #0 !dbg !109 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %self), !dbg !115
// CHECK:STDOUT:   ret i32 %1, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.31a8c47192dc2066(ptr %self) #0 !dbg !119 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f5c18a58c263f72b:core.Destroy.Core"(ptr %self), !dbg !123
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !126 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !130
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.5a42872629477478(ptr %self) #0 !dbg !133 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c9184c281ed4335d:core.Destroy.Core"(ptr %self), !dbg !137
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !141 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 1), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.d56e393be6e32080(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !149 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %return, i32 %value), !dbg !154
// CHECK:STDOUT:   ret void, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.d56e393be6e32080(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !157 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %return), !dbg !160
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %value) #0 !dbg !162 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !167
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !167
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !167
// CHECK:STDOUT:   ret i1 %2, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %value) #0 !dbg !170 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !175
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !175
// CHECK:STDOUT:   ret i32 %1, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 %other) #2 !dbg !178 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %other), !dbg !181
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !182
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !182
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !182
// CHECK:STDOUT:   ret void, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !186 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !195
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !195
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !197
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !197
// CHECK:STDOUT:   ret void, !dbg !198
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !200 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !207
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !207
// CHECK:STDOUT:   ret void, !dbg !208
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !209 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !214
// CHECK:STDOUT:   ret i32 %1, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !218 {
// CHECK:STDOUT:   ret i32 %self, !dbg !224
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.31a8c47192dc2066, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "break_continue.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{null}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{!7, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{null, !6}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = !{null, !7}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !4, line: 17, type: !11, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !20 = !DILocation(line: 18, column: 18, scope: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 18, column: 7, scope: !18)
// CHECK:STDOUT: !22 = !{!6, !7}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{!7, !7, !7}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!7, !7}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 19, column: 9, scope: !18)
// CHECK:STDOUT: !29 = !DILocation(line: 19, column: 8, scope: !18)
// CHECK:STDOUT: !30 = !DILocation(line: 19, column: 16, scope: !18)
// CHECK:STDOUT: !31 = !DILocation(line: 20, column: 9, scope: !18)
// CHECK:STDOUT: !32 = !DILocation(line: 20, column: 8, scope: !18)
// CHECK:STDOUT: !33 = !DILocation(line: 20, column: 16, scope: !18)
// CHECK:STDOUT: !34 = !DILocation(line: 21, column: 5, scope: !18)
// CHECK:STDOUT: !35 = !DILocation(line: 18, column: 3, scope: !18)
// CHECK:STDOUT: !36 = !DILocation(line: 17, column: 1, scope: !18)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core", scope: null, file: !4, line: 18, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !40)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !7)
// CHECK:STDOUT: !39 = !DILocation(line: 18, column: 7, scope: !37)
// CHECK:STDOUT: !40 = !{!38}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.32ec81fcfb7ba5dc:core.Destroy.Core", scope: null, file: !4, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !44)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !6)
// CHECK:STDOUT: !43 = !DILocation(line: 18, column: 7, scope: !41)
// CHECK:STDOUT: !44 = !{!42}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c3be8338012ae399:core.Destroy.Core", scope: null, file: !4, line: 18, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !7)
// CHECK:STDOUT: !47 = !DILocation(line: 18, column: 7, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c800cdf671d92dcc:core.Destroy.Core", scope: null, file: !4, line: 18, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !7)
// CHECK:STDOUT: !51 = !DILocation(line: 18, column: 7, scope: !49)
// CHECK:STDOUT: !52 = !{!50}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f5c18a58c263f72b:core.Destroy.Core", scope: null, file: !4, line: 18, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !7)
// CHECK:STDOUT: !55 = !DILocation(line: 18, column: 7, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.e933d5ff8212e171:core.Destroy.Core", scope: null, file: !4, line: 18, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !7)
// CHECK:STDOUT: !59 = !DILocation(line: 18, column: 18, scope: !57)
// CHECK:STDOUT: !60 = !{!58}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c9184c281ed4335d:core.Destroy.Core", scope: null, file: !4, line: 18, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !64)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !7)
// CHECK:STDOUT: !63 = !DILocation(line: 18, column: 18, scope: !61)
// CHECK:STDOUT: !64 = !{!62}
// CHECK:STDOUT: !67 = !DIFile(filename: "{{.*}}/prelude/range.carbon", directory: "")
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !67, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !75)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !7)
// CHECK:STDOUT: !73 = !DILocation(line: 19, column: 43, scope: !68)
// CHECK:STDOUT: !74 = !DILocation(line: 19, column: 36, scope: !68)
// CHECK:STDOUT: !75 = !{!69}
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !67, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !98)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !7)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 2, scope: !76, type: !7)
// CHECK:STDOUT: !83 = !DILocation(line: 21, column: 7, scope: !76)
// CHECK:STDOUT: !85 = !DILocation(line: 21, column: 27, scope: !76)
// CHECK:STDOUT: !87 = !DILocation(line: 22, column: 11, scope: !76)
// CHECK:STDOUT: !88 = !DILocation(line: 22, column: 19, scope: !76)
// CHECK:STDOUT: !89 = !DILocation(line: 22, column: 10, scope: !76)
// CHECK:STDOUT: !92 = !DILocation(line: 23, column: 9, scope: !76)
// CHECK:STDOUT: !93 = !DILocation(line: 24, column: 38, scope: !76)
// CHECK:STDOUT: !94 = !DILocation(line: 24, column: 16, scope: !76)
// CHECK:STDOUT: !95 = !DILocation(line: 24, column: 9, scope: !76)
// CHECK:STDOUT: !96 = !DILocation(line: 26, column: 16, scope: !76)
// CHECK:STDOUT: !97 = !DILocation(line: 26, column: 9, scope: !76)
// CHECK:STDOUT: !98 = !{!77, !78}
// CHECK:STDOUT: !99 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.d56e393be6e32080", scope: null, file: !99, line: 36, type: !27, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !108)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !7)
// CHECK:STDOUT: !106 = !DILocation(line: 37, column: 12, scope: !100)
// CHECK:STDOUT: !107 = !DILocation(line: 37, column: 5, scope: !100)
// CHECK:STDOUT: !108 = !{!101}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.d56e393be6e32080", scope: null, file: !99, line: 39, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !117)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !7)
// CHECK:STDOUT: !115 = !DILocation(line: 40, column: 12, scope: !109)
// CHECK:STDOUT: !116 = !DILocation(line: 40, column: 5, scope: !109)
// CHECK:STDOUT: !117 = !{!110}
// CHECK:STDOUT: !118 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.31a8c47192dc2066", scope: null, file: !118, line: 26, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !125)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !7)
// CHECK:STDOUT: !123 = !DILocation(line: 28, column: 5, scope: !119)
// CHECK:STDOUT: !124 = !DILocation(line: 26, column: 3, scope: !119)
// CHECK:STDOUT: !125 = !{!120}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !118, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !132)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !6)
// CHECK:STDOUT: !130 = !DILocation(line: 28, column: 5, scope: !126)
// CHECK:STDOUT: !131 = !DILocation(line: 26, column: 3, scope: !126)
// CHECK:STDOUT: !132 = !{!127}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.5a42872629477478", scope: null, file: !118, line: 26, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !139)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !7)
// CHECK:STDOUT: !137 = !DILocation(line: 28, column: 5, scope: !133)
// CHECK:STDOUT: !138 = !DILocation(line: 26, column: 3, scope: !133)
// CHECK:STDOUT: !139 = !{!134}
// CHECK:STDOUT: !140 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !140, line: 396, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !148)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !6)
// CHECK:STDOUT: !144 = !DILocation(line: 398, column: 5, scope: !141)
// CHECK:STDOUT: !145 = !{null, !6, !6}
// CHECK:STDOUT: !146 = !DISubroutineType(types: !145)
// CHECK:STDOUT: !147 = !DILocation(line: 396, column: 3, scope: !141)
// CHECK:STDOUT: !148 = !{!142}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.d56e393be6e32080", scope: null, file: !99, line: 33, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !156)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !6)
// CHECK:STDOUT: !154 = !DILocation(line: 34, column: 12, scope: !149)
// CHECK:STDOUT: !155 = !DILocation(line: 34, column: 5, scope: !149)
// CHECK:STDOUT: !156 = !{!150}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.d56e393be6e32080", scope: null, file: !99, line: 30, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !160 = !DILocation(line: 31, column: 12, scope: !157)
// CHECK:STDOUT: !161 = !DILocation(line: 31, column: 5, scope: !157)
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692", scope: null, file: !99, line: 141, type: !27, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !169)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !162, type: !7)
// CHECK:STDOUT: !167 = !DILocation(line: 142, column: 12, scope: !162)
// CHECK:STDOUT: !168 = !DILocation(line: 142, column: 5, scope: !162)
// CHECK:STDOUT: !169 = !{!163}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692", scope: null, file: !99, line: 144, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !177)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !7)
// CHECK:STDOUT: !175 = !DILocation(line: 145, column: 12, scope: !170)
// CHECK:STDOUT: !176 = !DILocation(line: 145, column: 5, scope: !170)
// CHECK:STDOUT: !177 = !{!171}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895", scope: null, file: !140, line: 331, type: !146, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !185)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !6)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 2, scope: !178, type: !6)
// CHECK:STDOUT: !181 = !DILocation(line: 0, scope: !178)
// CHECK:STDOUT: !182 = !DILocation(line: 331, column: 3, scope: !178)
// CHECK:STDOUT: !183 = !{!6, !6}
// CHECK:STDOUT: !184 = !DISubroutineType(types: !183)
// CHECK:STDOUT: !185 = !{!179, !180}
// CHECK:STDOUT: !186 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692", scope: null, file: !99, line: 131, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !199)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 1, scope: !186, type: !6)
// CHECK:STDOUT: !195 = !DILocation(line: 137, column: 5, scope: !186)
// CHECK:STDOUT: !197 = !DILocation(line: 138, column: 5, scope: !186)
// CHECK:STDOUT: !198 = !DILocation(line: 139, column: 5, scope: !186)
// CHECK:STDOUT: !199 = !{!187}
// CHECK:STDOUT: !200 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692", scope: null, file: !99, line: 125, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !207 = !DILocation(line: 128, column: 5, scope: !200)
// CHECK:STDOUT: !208 = !DILocation(line: 129, column: 5, scope: !200)
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !140, line: 64, type: !184, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !217)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 1, scope: !209, type: !6)
// CHECK:STDOUT: !214 = !DILocation(line: 66, column: 17, scope: !209)
// CHECK:STDOUT: !216 = !DILocation(line: 66, column: 5, scope: !209)
// CHECK:STDOUT: !217 = !{!210}
// CHECK:STDOUT: !218 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !140, line: 59, type: !184, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !225)
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 1, scope: !218, type: !6)
// CHECK:STDOUT: !224 = !DILocation(line: 59, column: 28, scope: !218)
// CHECK:STDOUT: !225 = !{!219}
// CHECK:STDOUT:
