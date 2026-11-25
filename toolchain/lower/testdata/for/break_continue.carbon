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
// CHECK:STDOUT:   br label %for.next, !dbg !29
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !30
// CHECK:STDOUT:   br label %for.next, !dbg !31
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %if.then.loc19, %for.next
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !36 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw i8, ptr %self, i64 0, !dbg !41
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !41
// CHECK:STDOUT:   ret i32 %1, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !44 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !51
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !52
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !51
// CHECK:STDOUT:   %end = getelementptr inbounds nuw i8, ptr %self, i64 4, !dbg !55
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !55
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !54
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !54
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !56
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !58
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !61
// CHECK:STDOUT:   call void @_CSome.Optional.Core.29b2425ca9ad75a4(ptr %return, i32 %7), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.29b2425ca9ad75a4(ptr %return), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.29b2425ca9ad75a4(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %self), !dbg !74
// CHECK:STDOUT:   ret i1 %1, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.29b2425ca9ad75a4(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %self), !dbg !83
// CHECK:STDOUT:   ret i32 %1, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 1), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.29b2425ca9ad75a4(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !95 {
// CHECK:STDOUT:   call void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %return, i32 %value), !dbg !100
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.29b2425ca9ad75a4(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !103 {
// CHECK:STDOUT:   call void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %return), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %value) #0 !dbg !108 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw i8, ptr %value, i64 4, !dbg !113
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !113
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !113
// CHECK:STDOUT:   ret i1 %2, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %value) #0 !dbg !116 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw i8, ptr %value, i64 0, !dbg !121
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !121
// CHECK:STDOUT:   ret i32 %1, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 %other) #2 !dbg !124 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %other), !dbg !128
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !128
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !128
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !128
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !132 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw i8, ptr %return, i64 0, !dbg !141
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !141
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw i8, ptr %return, i64 4, !dbg !143
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !143
// CHECK:STDOUT:   ret void, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !146 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw i8, ptr %return, i64 4, !dbg !153
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !153
// CHECK:STDOUT:   ret void, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %self) #0 !dbg !155 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !160
// CHECK:STDOUT:   ret i32 %1, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !164 {
// CHECK:STDOUT:   ret i32 %self, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
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
// CHECK:STDOUT: !29 = !DILocation(line: 20, column: 16, scope: !14)
// CHECK:STDOUT: !30 = !DILocation(line: 21, column: 5, scope: !14)
// CHECK:STDOUT: !31 = !DILocation(line: 18, column: 3, scope: !14)
// CHECK:STDOUT: !32 = !DILocation(line: 17, column: 1, scope: !14)
// CHECK:STDOUT: !35 = !DIFile(filename: "{{.*}}/prelude/range.carbon", directory: "")
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !35, line: 19, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !43)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !7)
// CHECK:STDOUT: !41 = !DILocation(line: 19, column: 43, scope: !36)
// CHECK:STDOUT: !42 = !DILocation(line: 19, column: 36, scope: !36)
// CHECK:STDOUT: !43 = !{!37}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !35, line: 20, type: !21, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !66)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !7)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 2, scope: !44, type: !6)
// CHECK:STDOUT: !51 = !DILocation(line: 21, column: 7, scope: !44)
// CHECK:STDOUT: !52 = !DILocation(line: 21, column: 27, scope: !44)
// CHECK:STDOUT: !54 = !DILocation(line: 22, column: 11, scope: !44)
// CHECK:STDOUT: !55 = !DILocation(line: 22, column: 19, scope: !44)
// CHECK:STDOUT: !56 = !DILocation(line: 22, column: 10, scope: !44)
// CHECK:STDOUT: !58 = !DILocation(line: 23, column: 9, scope: !44)
// CHECK:STDOUT: !59 = !{null, !6}
// CHECK:STDOUT: !60 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !61 = !DILocation(line: 24, column: 38, scope: !44)
// CHECK:STDOUT: !62 = !DILocation(line: 24, column: 16, scope: !44)
// CHECK:STDOUT: !63 = !DILocation(line: 24, column: 9, scope: !44)
// CHECK:STDOUT: !64 = !DILocation(line: 26, column: 16, scope: !44)
// CHECK:STDOUT: !65 = !DILocation(line: 26, column: 9, scope: !44)
// CHECK:STDOUT: !66 = !{!45, !46}
// CHECK:STDOUT: !67 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.29b2425ca9ad75a4", scope: null, file: !67, line: 36, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !76)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !7)
// CHECK:STDOUT: !74 = !DILocation(line: 37, column: 12, scope: !68)
// CHECK:STDOUT: !75 = !DILocation(line: 37, column: 5, scope: !68)
// CHECK:STDOUT: !76 = !{!69}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.29b2425ca9ad75a4", scope: null, file: !67, line: 39, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !85)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !7)
// CHECK:STDOUT: !83 = !DILocation(line: 40, column: 12, scope: !77)
// CHECK:STDOUT: !84 = !DILocation(line: 40, column: 5, scope: !77)
// CHECK:STDOUT: !85 = !{!78}
// CHECK:STDOUT: !86 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !86, line: 396, type: !60, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !94)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !6)
// CHECK:STDOUT: !90 = !DILocation(line: 398, column: 5, scope: !87)
// CHECK:STDOUT: !91 = !{null, !6, !6}
// CHECK:STDOUT: !92 = !DISubroutineType(types: !91)
// CHECK:STDOUT: !93 = !DILocation(line: 396, column: 3, scope: !87)
// CHECK:STDOUT: !94 = !{!88}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.29b2425ca9ad75a4", scope: null, file: !67, line: 33, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !102)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !6)
// CHECK:STDOUT: !100 = !DILocation(line: 34, column: 12, scope: !95)
// CHECK:STDOUT: !101 = !DILocation(line: 34, column: 5, scope: !95)
// CHECK:STDOUT: !102 = !{!96}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.29b2425ca9ad75a4", scope: null, file: !67, line: 30, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !106 = !DILocation(line: 31, column: 12, scope: !103)
// CHECK:STDOUT: !107 = !DILocation(line: 31, column: 5, scope: !103)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !67, line: 141, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !115)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !7)
// CHECK:STDOUT: !113 = !DILocation(line: 142, column: 12, scope: !108)
// CHECK:STDOUT: !114 = !DILocation(line: 142, column: 5, scope: !108)
// CHECK:STDOUT: !115 = !{!109}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !67, line: 144, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !123)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !7)
// CHECK:STDOUT: !121 = !DILocation(line: 145, column: 12, scope: !116)
// CHECK:STDOUT: !122 = !DILocation(line: 145, column: 5, scope: !116)
// CHECK:STDOUT: !123 = !{!117}
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26", scope: null, file: !86, line: 331, type: !92, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !131)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !124, type: !6)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 2, scope: !124, type: !6)
// CHECK:STDOUT: !128 = !DILocation(line: 331, column: 3, scope: !124)
// CHECK:STDOUT: !129 = !{!6, !6}
// CHECK:STDOUT: !130 = !DISubroutineType(types: !129)
// CHECK:STDOUT: !131 = !{!125, !126}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !67, line: 131, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !145)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !6)
// CHECK:STDOUT: !141 = !DILocation(line: 137, column: 5, scope: !132)
// CHECK:STDOUT: !143 = !DILocation(line: 138, column: 5, scope: !132)
// CHECK:STDOUT: !144 = !DILocation(line: 139, column: 5, scope: !132)
// CHECK:STDOUT: !145 = !{!133}
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "None", linkageName: "_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !67, line: 125, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !153 = !DILocation(line: 128, column: 5, scope: !146)
// CHECK:STDOUT: !154 = !DILocation(line: 129, column: 5, scope: !146)
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701", scope: null, file: !86, line: 64, type: !130, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !163)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !6)
// CHECK:STDOUT: !160 = !DILocation(line: 66, column: 17, scope: !155)
// CHECK:STDOUT: !162 = !DILocation(line: 66, column: 5, scope: !155)
// CHECK:STDOUT: !163 = !{!156}
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !86, line: 59, type: !130, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !171)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !6)
// CHECK:STDOUT: !170 = !DILocation(line: 59, column: 28, scope: !164)
// CHECK:STDOUT: !171 = !{!165}
// CHECK:STDOUT:
