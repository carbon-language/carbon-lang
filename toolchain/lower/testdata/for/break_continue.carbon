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

import Core library "range";

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
// CHECK:STDOUT:   %.loc20_32.1.temp = alloca { i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %.loc20_33.1.temp = alloca { i32, i1 }, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_32.1.temp), !dbg !7
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc20_32.1.temp, i32 100), !dbg !7
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc20_32.1.temp), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %if.else.loc22, %if.then.loc22, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc20_33.1.temp, ptr %.loc20_32.1.temp, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.a88acd4c94838316(ptr %.loc20_33.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.a88acd4c94838316(ptr %.loc20_33.1.temp), !dbg !8
// CHECK:STDOUT:   %F.call = call i1 @_CF.Main(), !dbg !9
// CHECK:STDOUT:   br i1 %F.call, label %if.then.loc21, label %if.else.loc21, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc21:                                    ; preds = %for.body
// CHECK:STDOUT:   br label %for.done, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc21:                                    ; preds = %for.body
// CHECK:STDOUT:   %G.call = call i1 @_CG.Main(), !dbg !12
// CHECK:STDOUT:   br i1 %G.call, label %if.then.loc22, label %if.else.loc22, !dbg !13
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc22:                                    ; preds = %if.else.loc21
// CHECK:STDOUT:   br label %for.next, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc22:                                    ; preds = %if.else.loc21
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !15
// CHECK:STDOUT:   br label %for.next, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %if.then.loc21, %for.next
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !18 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !26
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !26
// CHECK:STDOUT:   ret i32 %1, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret({ i32, i1 }) %return, ptr %self, ptr %cursor) #0 !dbg !28 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !34
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !35
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !34
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !36
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !36
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !37
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !37
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !38
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !39
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !40
// CHECK:STDOUT:   call void @_CSome.Optional.Core.a88acd4c94838316(ptr %return, i32 %7), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.a88acd4c94838316(ptr %return), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.a88acd4c94838316(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.3e8267224c5dc9c2:OptionalStorage.Core.32fa7206ac42c9b7"(ptr %self), !dbg !51
// CHECK:STDOUT:   ret i1 %1, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.a88acd4c94838316(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.3e8267224c5dc9c2:OptionalStorage.Core.32fa7206ac42c9b7"(ptr %self), !dbg !56
// CHECK:STDOUT:   ret i32 %1, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT:   call void @"_COp:thunk.Int.9452f4c51951679b.Core:AddAssignWith.5375cfcbca6d9e35.Core.cfd8ddddc59ca787"(ptr %self, i32 1), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.a88acd4c94838316(ptr sret({ i32, i1 }) %return, i32 %value) #0 !dbg !66 {
// CHECK:STDOUT:   call void @"_CSome.3e8267224c5dc9c2:OptionalStorage.Core.32fa7206ac42c9b7"(ptr %return, i32 %value), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.a88acd4c94838316(ptr sret({ i32, i1 }) %return) #0 !dbg !73 {
// CHECK:STDOUT:   call void @"_CNone.3e8267224c5dc9c2:OptionalStorage.Core.32fa7206ac42c9b7"(ptr %return), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.3e8267224c5dc9c2:OptionalStorage.Core.32fa7206ac42c9b7"(ptr %value) #0 !dbg !78 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %value, i32 0, i32 1, !dbg !81
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !81
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !81
// CHECK:STDOUT:   ret i1 %2, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.3e8267224c5dc9c2:OptionalStorage.Core.32fa7206ac42c9b7"(ptr %value) #0 !dbg !83 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { i32, i1 }, ptr %value, i32 0, i32 0, !dbg !86
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !86
// CHECK:STDOUT:   ret i32 %1, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk.Int.9452f4c51951679b.Core:AddAssignWith.5375cfcbca6d9e35.Core.cfd8ddddc59ca787"(ptr %self, i32 %other) #2 !dbg !88 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.14b8745117b2bc54:ImplicitAs.a9271c1e04015f9c.Core.beca8a3ca33a86fb"(i32 %other), !dbg !94
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !95
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !95
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !95
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.3e8267224c5dc9c2:OptionalStorage.Core.32fa7206ac42c9b7"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !96 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 0, !dbg !99
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !99
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !100
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !100
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.3e8267224c5dc9c2:OptionalStorage.Core.32fa7206ac42c9b7"(ptr sret({ i32, i1 }) %return) #0 !dbg !102 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !103
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !103
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.14b8745117b2bc54:ImplicitAs.a9271c1e04015f9c.Core.beca8a3ca33a86fb"(i32 %self) #0 !dbg !105 {
// CHECK:STDOUT:   ret i32 %self, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "break_continue.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 20, column: 18, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 20, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 21, column: 9, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 21, column: 8, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 21, column: 16, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 22, column: 9, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 22, column: 8, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 22, column: 16, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 23, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 19, column: 1, scope: !4)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !19, line: 25, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !19 = !DIFile(filename: "{{.*}}/range.carbon", directory: "")
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{!22, !23}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !18, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 25, column: 51, scope: !18)
// CHECK:STDOUT: !27 = !DILocation(line: 25, column: 44, scope: !18)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !19, line: 26, type: !29, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !31)
// CHECK:STDOUT: !29 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !30 = !{!23, !23, !23}
// CHECK:STDOUT: !31 = !{!32, !33}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !28, type: !23)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 2, scope: !28, type: !23)
// CHECK:STDOUT: !34 = !DILocation(line: 27, column: 7, scope: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 27, column: 27, scope: !28)
// CHECK:STDOUT: !36 = !DILocation(line: 28, column: 19, scope: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 28, column: 11, scope: !28)
// CHECK:STDOUT: !38 = !DILocation(line: 28, column: 10, scope: !28)
// CHECK:STDOUT: !39 = !DILocation(line: 29, column: 9, scope: !28)
// CHECK:STDOUT: !40 = !DILocation(line: 30, column: 38, scope: !28)
// CHECK:STDOUT: !41 = !DILocation(line: 30, column: 16, scope: !28)
// CHECK:STDOUT: !42 = !DILocation(line: 30, column: 9, scope: !28)
// CHECK:STDOUT: !43 = !DILocation(line: 32, column: 16, scope: !28)
// CHECK:STDOUT: !44 = !DILocation(line: 32, column: 9, scope: !28)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.a88acd4c94838316", scope: null, file: !46, line: 32, type: !47, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !49)
// CHECK:STDOUT: !46 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !47 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !48 = !{!23, !23}
// CHECK:STDOUT: !49 = !{!50}
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !45, type: !23)
// CHECK:STDOUT: !51 = !DILocation(line: 33, column: 12, scope: !45)
// CHECK:STDOUT: !52 = !DILocation(line: 33, column: 5, scope: !45)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.a88acd4c94838316", scope: null, file: !46, line: 35, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !23)
// CHECK:STDOUT: !56 = !DILocation(line: 36, column: 12, scope: !53)
// CHECK:STDOUT: !57 = !DILocation(line: 36, column: 5, scope: !53)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !59, line: 339, type: !60, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !62)
// CHECK:STDOUT: !59 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !60 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !61 = !{null, !22}
// CHECK:STDOUT: !62 = !{!63}
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !58, type: !22)
// CHECK:STDOUT: !64 = !DILocation(line: 341, column: 5, scope: !58)
// CHECK:STDOUT: !65 = !DILocation(line: 339, column: 3, scope: !58)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.a88acd4c94838316", scope: null, file: !46, line: 29, type: !67, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !69)
// CHECK:STDOUT: !67 = !DISubroutineType(types: !68)
// CHECK:STDOUT: !68 = !{!23, !22}
// CHECK:STDOUT: !69 = !{!70}
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !66, type: !22)
// CHECK:STDOUT: !71 = !DILocation(line: 30, column: 12, scope: !66)
// CHECK:STDOUT: !72 = !DILocation(line: 30, column: 5, scope: !66)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.a88acd4c94838316", scope: null, file: !46, line: 26, type: !74, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !74 = !DISubroutineType(types: !75)
// CHECK:STDOUT: !75 = !{!23}
// CHECK:STDOUT: !76 = !DILocation(line: 27, column: 12, scope: !73)
// CHECK:STDOUT: !77 = !DILocation(line: 27, column: 5, scope: !73)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.3e8267224c5dc9c2:OptionalStorage.Core.32fa7206ac42c9b7", scope: null, file: !46, line: 112, type: !47, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !79)
// CHECK:STDOUT: !79 = !{!80}
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !78, type: !23)
// CHECK:STDOUT: !81 = !DILocation(line: 113, column: 12, scope: !78)
// CHECK:STDOUT: !82 = !DILocation(line: 113, column: 5, scope: !78)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.3e8267224c5dc9c2:OptionalStorage.Core.32fa7206ac42c9b7", scope: null, file: !46, line: 115, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !84)
// CHECK:STDOUT: !84 = !{!85}
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !83, type: !23)
// CHECK:STDOUT: !86 = !DILocation(line: 116, column: 12, scope: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 116, column: 5, scope: !83)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk.Int.9452f4c51951679b.Core:AddAssignWith.5375cfcbca6d9e35.Core.cfd8ddddc59ca787", scope: null, file: !59, line: 275, type: !89, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !91)
// CHECK:STDOUT: !89 = !DISubroutineType(types: !90)
// CHECK:STDOUT: !90 = !{null, !22, !22}
// CHECK:STDOUT: !91 = !{!92, !93}
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !88, type: !22)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 2, scope: !88, type: !22)
// CHECK:STDOUT: !94 = !DILocation(line: 4294967295, scope: !88)
// CHECK:STDOUT: !95 = !DILocation(line: 275, column: 3, scope: !88)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.3e8267224c5dc9c2:OptionalStorage.Core.32fa7206ac42c9b7", scope: null, file: !46, line: 104, type: !67, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !97)
// CHECK:STDOUT: !97 = !{!98}
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !96, type: !22)
// CHECK:STDOUT: !99 = !DILocation(line: 108, column: 5, scope: !96)
// CHECK:STDOUT: !100 = !DILocation(line: 109, column: 5, scope: !96)
// CHECK:STDOUT: !101 = !DILocation(line: 110, column: 5, scope: !96)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "None", linkageName: "_CNone.3e8267224c5dc9c2:OptionalStorage.Core.32fa7206ac42c9b7", scope: null, file: !46, line: 99, type: !74, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !103 = !DILocation(line: 101, column: 5, scope: !102)
// CHECK:STDOUT: !104 = !DILocation(line: 102, column: 5, scope: !102)
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.14b8745117b2bc54:ImplicitAs.a9271c1e04015f9c.Core.beca8a3ca33a86fb", scope: null, file: !106, line: 24, type: !107, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !109)
// CHECK:STDOUT: !106 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !107 = !DISubroutineType(types: !108)
// CHECK:STDOUT: !108 = !{!22, !22}
// CHECK:STDOUT: !109 = !{!110}
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !105, type: !22)
// CHECK:STDOUT: !111 = !DILocation(line: 24, column: 38, scope: !105)
