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
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_32.1.temp = alloca { i32, i32 }, align 4, !dbg !16
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !17
// CHECK:STDOUT:   %.loc18_33.1.temp = alloca <{ i32, i1 }>, align 4, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_32.1.temp), !dbg !16
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc18_32.1.temp, i32 100), !dbg !16
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc18_32.1.temp), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !17
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !17
// CHECK:STDOUT:   br label %for.next, !dbg !17
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %if.else.loc20, %if.then.loc20, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_33.1.temp), !dbg !17
// CHECK:STDOUT:   call void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc18_33.1.temp, ptr %.loc18_32.1.temp, ptr %var), !dbg !17
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.29b2425ca9ad75a4(ptr %.loc18_33.1.temp), !dbg !17
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !17
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.29b2425ca9ad75a4(ptr %.loc18_33.1.temp), !dbg !17
// CHECK:STDOUT:   %F.call = call i1 @_CF.Main(), !dbg !24
// CHECK:STDOUT:   br i1 %F.call, label %if.then.loc19, label %if.else.loc19, !dbg !25
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc19:                                    ; preds = %for.body
// CHECK:STDOUT:   br label %for.done, !dbg !26
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc19:                                    ; preds = %for.body
// CHECK:STDOUT:   %G.call = call i1 @_CG.Main(), !dbg !27
// CHECK:STDOUT:   br i1 %G.call, label %if.then.loc20, label %if.else.loc20, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c6daab0befde28a8(ptr %.loc18_33.1.temp), !dbg !17
// CHECK:STDOUT:   br label %for.next, !dbg !31
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c6daab0befde28a8(ptr %.loc18_33.1.temp), !dbg !17
// CHECK:STDOUT:   br label %for.next, !dbg !33
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %if.then.loc19, %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c6daab0befde28a8(ptr %.loc18_33.1.temp), !dbg !17
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9a03059f269926ed(ptr %var), !dbg !17
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ab8fcbb9cd97e8e7(ptr %.loc18_32.1.temp), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !45
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !45
// CHECK:STDOUT:   ret i32 %1, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !48 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !55
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !55
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !56
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !55
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !59
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !59
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !58
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !58
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !60
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !62
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !63
// CHECK:STDOUT:   call void @_CSome.Optional.Core.29b2425ca9ad75a4(ptr %return, i32 %7), !dbg !64
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9a03059f269926ed(ptr %1), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.29b2425ca9ad75a4(ptr %return), !dbg !66
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9a03059f269926ed(ptr %1), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.29b2425ca9ad75a4(ptr %self) #0 !dbg !70 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %self), !dbg !76
// CHECK:STDOUT:   ret i1 %1, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.29b2425ca9ad75a4(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %self), !dbg !85
// CHECK:STDOUT:   ret i32 %1, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c6daab0befde28a8(ptr %t) #0 !dbg !89 {
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9a03059f269926ed(ptr %t) #0 !dbg !96 {
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ab8fcbb9cd97e8e7(ptr %t) #0 !dbg !103 {
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !111 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 1), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.29b2425ca9ad75a4(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !119 {
// CHECK:STDOUT:   call void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %return, i32 %value), !dbg !124
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.29b2425ca9ad75a4(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !127 {
// CHECK:STDOUT:   call void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %return), !dbg !130
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %value) #0 !dbg !132 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !137
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !137
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !137
// CHECK:STDOUT:   ret i1 %2, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %value) #0 !dbg !140 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !145
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !145
// CHECK:STDOUT:   ret i32 %1, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 %other) #2 !dbg !148 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %other), !dbg !152
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !152
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !152
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !152
// CHECK:STDOUT:   ret void, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !156 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !165
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !165
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !167
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !167
// CHECK:STDOUT:   ret void, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !170 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !177
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !177
// CHECK:STDOUT:   ret void, !dbg !178
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %self) #0 !dbg !179 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !184
// CHECK:STDOUT:   ret i32 %1, !dbg !186
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !188 {
// CHECK:STDOUT:   ret i32 %self, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c6daab0befde28a8, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.9a03059f269926ed, { 2, 1, 0 }
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
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !4, line: 17, type: !11, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !16 = !DILocation(line: 18, column: 18, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 18, column: 7, scope: !14)
// CHECK:STDOUT: !18 = !{!6, !7}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !20 = !{!7, !7, !6}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !22 = !{!7, !7}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !DILocation(line: 19, column: 9, scope: !14)
// CHECK:STDOUT: !25 = !DILocation(line: 19, column: 8, scope: !14)
// CHECK:STDOUT: !26 = !DILocation(line: 19, column: 16, scope: !14)
// CHECK:STDOUT: !27 = !DILocation(line: 20, column: 9, scope: !14)
// CHECK:STDOUT: !28 = !DILocation(line: 20, column: 8, scope: !14)
// CHECK:STDOUT: !29 = !{null, !7}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 20, column: 16, scope: !14)
// CHECK:STDOUT: !32 = !DILocation(line: 21, column: 5, scope: !14)
// CHECK:STDOUT: !33 = !DILocation(line: 18, column: 3, scope: !14)
// CHECK:STDOUT: !34 = !{null, !6}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 17, column: 1, scope: !14)
// CHECK:STDOUT: !39 = !DIFile(filename: "{{.*}}/prelude/range.carbon", directory: "")
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !39, line: 19, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !47)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !7)
// CHECK:STDOUT: !45 = !DILocation(line: 19, column: 43, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 19, column: 36, scope: !40)
// CHECK:STDOUT: !47 = !{!41}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !39, line: 20, type: !21, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !68)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !7)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 2, scope: !48, type: !6)
// CHECK:STDOUT: !55 = !DILocation(line: 21, column: 7, scope: !48)
// CHECK:STDOUT: !56 = !DILocation(line: 21, column: 27, scope: !48)
// CHECK:STDOUT: !58 = !DILocation(line: 22, column: 11, scope: !48)
// CHECK:STDOUT: !59 = !DILocation(line: 22, column: 19, scope: !48)
// CHECK:STDOUT: !60 = !DILocation(line: 22, column: 10, scope: !48)
// CHECK:STDOUT: !62 = !DILocation(line: 23, column: 9, scope: !48)
// CHECK:STDOUT: !63 = !DILocation(line: 24, column: 38, scope: !48)
// CHECK:STDOUT: !64 = !DILocation(line: 24, column: 16, scope: !48)
// CHECK:STDOUT: !65 = !DILocation(line: 24, column: 9, scope: !48)
// CHECK:STDOUT: !66 = !DILocation(line: 26, column: 16, scope: !48)
// CHECK:STDOUT: !67 = !DILocation(line: 26, column: 9, scope: !48)
// CHECK:STDOUT: !68 = !{!49, !50}
// CHECK:STDOUT: !69 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.29b2425ca9ad75a4", scope: null, file: !69, line: 36, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !78)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !7)
// CHECK:STDOUT: !76 = !DILocation(line: 37, column: 12, scope: !70)
// CHECK:STDOUT: !77 = !DILocation(line: 37, column: 5, scope: !70)
// CHECK:STDOUT: !78 = !{!71}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.29b2425ca9ad75a4", scope: null, file: !69, line: 39, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !87)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !7)
// CHECK:STDOUT: !85 = !DILocation(line: 40, column: 12, scope: !79)
// CHECK:STDOUT: !86 = !DILocation(line: 40, column: 5, scope: !79)
// CHECK:STDOUT: !87 = !{!80}
// CHECK:STDOUT: !88 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c6daab0befde28a8", scope: null, file: !88, line: 35, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !95)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !7)
// CHECK:STDOUT: !94 = !DILocation(line: 35, column: 1, scope: !89)
// CHECK:STDOUT: !95 = !{!90}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9a03059f269926ed", scope: null, file: !88, line: 35, type: !35, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !102)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !6)
// CHECK:STDOUT: !101 = !DILocation(line: 35, column: 1, scope: !96)
// CHECK:STDOUT: !102 = !{!97}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ab8fcbb9cd97e8e7", scope: null, file: !88, line: 35, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !109)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !7)
// CHECK:STDOUT: !108 = !DILocation(line: 35, column: 1, scope: !103)
// CHECK:STDOUT: !109 = !{!104}
// CHECK:STDOUT: !110 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !110, line: 396, type: !35, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !118)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !6)
// CHECK:STDOUT: !114 = !DILocation(line: 398, column: 5, scope: !111)
// CHECK:STDOUT: !115 = !{null, !6, !6}
// CHECK:STDOUT: !116 = !DISubroutineType(types: !115)
// CHECK:STDOUT: !117 = !DILocation(line: 396, column: 3, scope: !111)
// CHECK:STDOUT: !118 = !{!112}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.29b2425ca9ad75a4", scope: null, file: !69, line: 33, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !126)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !6)
// CHECK:STDOUT: !124 = !DILocation(line: 34, column: 12, scope: !119)
// CHECK:STDOUT: !125 = !DILocation(line: 34, column: 5, scope: !119)
// CHECK:STDOUT: !126 = !{!120}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.29b2425ca9ad75a4", scope: null, file: !69, line: 30, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !130 = !DILocation(line: 31, column: 12, scope: !127)
// CHECK:STDOUT: !131 = !DILocation(line: 31, column: 5, scope: !127)
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !69, line: 141, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !139)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !7)
// CHECK:STDOUT: !137 = !DILocation(line: 142, column: 12, scope: !132)
// CHECK:STDOUT: !138 = !DILocation(line: 142, column: 5, scope: !132)
// CHECK:STDOUT: !139 = !{!133}
// CHECK:STDOUT: !140 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !69, line: 144, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !147)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !140, type: !7)
// CHECK:STDOUT: !145 = !DILocation(line: 145, column: 12, scope: !140)
// CHECK:STDOUT: !146 = !DILocation(line: 145, column: 5, scope: !140)
// CHECK:STDOUT: !147 = !{!141}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26", scope: null, file: !110, line: 331, type: !116, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !155)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !6)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 2, scope: !148, type: !6)
// CHECK:STDOUT: !152 = !DILocation(line: 331, column: 3, scope: !148)
// CHECK:STDOUT: !153 = !{!6, !6}
// CHECK:STDOUT: !154 = !DISubroutineType(types: !153)
// CHECK:STDOUT: !155 = !{!149, !150}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !69, line: 131, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !169)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !6)
// CHECK:STDOUT: !165 = !DILocation(line: 137, column: 5, scope: !156)
// CHECK:STDOUT: !167 = !DILocation(line: 138, column: 5, scope: !156)
// CHECK:STDOUT: !168 = !DILocation(line: 139, column: 5, scope: !156)
// CHECK:STDOUT: !169 = !{!157}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "None", linkageName: "_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !69, line: 125, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !177 = !DILocation(line: 128, column: 5, scope: !170)
// CHECK:STDOUT: !178 = !DILocation(line: 129, column: 5, scope: !170)
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701", scope: null, file: !110, line: 64, type: !154, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !187)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !6)
// CHECK:STDOUT: !184 = !DILocation(line: 66, column: 17, scope: !179)
// CHECK:STDOUT: !186 = !DILocation(line: 66, column: 5, scope: !179)
// CHECK:STDOUT: !187 = !{!180}
// CHECK:STDOUT: !188 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !110, line: 59, type: !154, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !195)
// CHECK:STDOUT: !189 = !DILocalVariable(arg: 1, scope: !188, type: !6)
// CHECK:STDOUT: !194 = !DILocation(line: 59, column: 28, scope: !188)
// CHECK:STDOUT: !195 = !{!189}
// CHECK:STDOUT:
