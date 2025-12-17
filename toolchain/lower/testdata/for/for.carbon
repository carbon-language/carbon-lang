// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/for/for.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/for/for.carbon

import Core library "range";

fn F();
fn G();
fn H();

fn For() {
  F();
  for (_: i32 in Core.Range(100)) {
    G();
  }
  H();
}

// CHECK:STDOUT: ; ModuleID = 'for.carbon'
// CHECK:STDOUT: source_filename = "for.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CH.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_32.1.temp = alloca { i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %.loc21_33.1.temp = alloca { i32, i1 }, align 8, !dbg !8
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_32.1.temp), !dbg !7
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc21_32.1.temp, i32 100), !dbg !7
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc21_32.1.temp), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.IntRange.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc21_33.1.temp, ptr %.loc21_32.1.temp, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.ed286a88567486e7(ptr %.loc21_33.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.ed286a88567486e7(ptr %.loc21_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !10
// CHECK:STDOUT:   br label %for.next, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !14 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !22
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !22
// CHECK:STDOUT:   ret i32 %1, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret({ i32, i1 }) %return, ptr %self, ptr %cursor) #0 !dbg !24 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !30
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !31
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !30
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !32
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !32
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !33
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !33
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !34
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !35
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !36
// CHECK:STDOUT:   call void @_CSome.Optional.Core.ed286a88567486e7(ptr %return, i32 %7), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.ed286a88567486e7(ptr %return), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.ed286a88567486e7(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e5bf07b1b9c2cbc2:OptionalStorage.Core.033324e61dc3f972"(ptr %self), !dbg !47
// CHECK:STDOUT:   ret i1 %1, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.ed286a88567486e7(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.e5bf07b1b9c2cbc2:OptionalStorage.Core.033324e61dc3f972"(ptr %self), !dbg !52
// CHECK:STDOUT:   ret i32 %1, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT:   call void @"_COp:thunk.Int.Core:AddAssignWith.Core.532bdfe78dd580b3"(ptr %self, i32 1), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.ed286a88567486e7(ptr sret({ i32, i1 }) %return, i32 %value) #0 !dbg !62 {
// CHECK:STDOUT:   call void @"_CSome.e5bf07b1b9c2cbc2:OptionalStorage.Core.033324e61dc3f972"(ptr %return, i32 %value), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.ed286a88567486e7(ptr sret({ i32, i1 }) %return) #0 !dbg !69 {
// CHECK:STDOUT:   call void @"_CNone.e5bf07b1b9c2cbc2:OptionalStorage.Core.033324e61dc3f972"(ptr %return), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e5bf07b1b9c2cbc2:OptionalStorage.Core.033324e61dc3f972"(ptr %value) #0 !dbg !74 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %value, i32 0, i32 1, !dbg !77
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !77
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !77
// CHECK:STDOUT:   ret i1 %2, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.e5bf07b1b9c2cbc2:OptionalStorage.Core.033324e61dc3f972"(ptr %value) #0 !dbg !79 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { i32, i1 }, ptr %value, i32 0, i32 0, !dbg !82
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !82
// CHECK:STDOUT:   ret i32 %1, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk.Int.Core:AddAssignWith.Core.532bdfe78dd580b3"(ptr %self, i32 %other) #2 !dbg !84 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.e5bf07b1b9c2cbc2:ImplicitAs.Core.033324e61dc3f972"(i32 %other), !dbg !90
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !91
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !91
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !91
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.e5bf07b1b9c2cbc2:OptionalStorage.Core.033324e61dc3f972"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !92 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 0, !dbg !95
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !95
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !96
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !96
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.e5bf07b1b9c2cbc2:OptionalStorage.Core.033324e61dc3f972"(ptr sret({ i32, i1 }) %return) #0 !dbg !98 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !99
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !99
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.e5bf07b1b9c2cbc2:ImplicitAs.Core.033324e61dc3f972"(i32 %self) #0 !dbg !101 {
// CHECK:STDOUT:   ret i32 %self, !dbg !107
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
// CHECK:STDOUT: !3 = !DIFile(filename: "for.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 21, column: 18, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 21, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 22, column: 5, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 21, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 24, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 19, column: 1, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !15, line: 25, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !15 = !DIFile(filename: "{{.*}}/range.carbon", directory: "")
// CHECK:STDOUT: !16 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !17 = !{!18, !19}
// CHECK:STDOUT: !18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !14, type: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 25, column: 51, scope: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 25, column: 44, scope: !14)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !15, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !27)
// CHECK:STDOUT: !25 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !26 = !{!19, !19, !19}
// CHECK:STDOUT: !27 = !{!28, !29}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !24, type: !19)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 2, scope: !24, type: !19)
// CHECK:STDOUT: !30 = !DILocation(line: 27, column: 7, scope: !24)
// CHECK:STDOUT: !31 = !DILocation(line: 27, column: 27, scope: !24)
// CHECK:STDOUT: !32 = !DILocation(line: 28, column: 19, scope: !24)
// CHECK:STDOUT: !33 = !DILocation(line: 28, column: 11, scope: !24)
// CHECK:STDOUT: !34 = !DILocation(line: 28, column: 10, scope: !24)
// CHECK:STDOUT: !35 = !DILocation(line: 29, column: 9, scope: !24)
// CHECK:STDOUT: !36 = !DILocation(line: 30, column: 38, scope: !24)
// CHECK:STDOUT: !37 = !DILocation(line: 30, column: 16, scope: !24)
// CHECK:STDOUT: !38 = !DILocation(line: 30, column: 9, scope: !24)
// CHECK:STDOUT: !39 = !DILocation(line: 32, column: 16, scope: !24)
// CHECK:STDOUT: !40 = !DILocation(line: 32, column: 9, scope: !24)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.ed286a88567486e7", scope: null, file: !42, line: 32, type: !43, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !45)
// CHECK:STDOUT: !42 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !43 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !44 = !{!19, !19}
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !41, type: !19)
// CHECK:STDOUT: !47 = !DILocation(line: 33, column: 12, scope: !41)
// CHECK:STDOUT: !48 = !DILocation(line: 33, column: 5, scope: !41)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.ed286a88567486e7", scope: null, file: !42, line: 35, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !19)
// CHECK:STDOUT: !52 = !DILocation(line: 36, column: 12, scope: !49)
// CHECK:STDOUT: !53 = !DILocation(line: 36, column: 5, scope: !49)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !55, line: 339, type: !56, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !58)
// CHECK:STDOUT: !55 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !56 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !57 = !{null, !18}
// CHECK:STDOUT: !58 = !{!59}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !54, type: !18)
// CHECK:STDOUT: !60 = !DILocation(line: 341, column: 5, scope: !54)
// CHECK:STDOUT: !61 = !DILocation(line: 339, column: 3, scope: !54)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.ed286a88567486e7", scope: null, file: !42, line: 29, type: !63, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DISubroutineType(types: !64)
// CHECK:STDOUT: !64 = !{!19, !18}
// CHECK:STDOUT: !65 = !{!66}
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !62, type: !18)
// CHECK:STDOUT: !67 = !DILocation(line: 30, column: 12, scope: !62)
// CHECK:STDOUT: !68 = !DILocation(line: 30, column: 5, scope: !62)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.ed286a88567486e7", scope: null, file: !42, line: 26, type: !70, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !70 = !DISubroutineType(types: !71)
// CHECK:STDOUT: !71 = !{!19}
// CHECK:STDOUT: !72 = !DILocation(line: 27, column: 12, scope: !69)
// CHECK:STDOUT: !73 = !DILocation(line: 27, column: 5, scope: !69)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e5bf07b1b9c2cbc2:OptionalStorage.Core.033324e61dc3f972", scope: null, file: !42, line: 112, type: !43, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !75)
// CHECK:STDOUT: !75 = !{!76}
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !74, type: !19)
// CHECK:STDOUT: !77 = !DILocation(line: 113, column: 12, scope: !74)
// CHECK:STDOUT: !78 = !DILocation(line: 113, column: 5, scope: !74)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e5bf07b1b9c2cbc2:OptionalStorage.Core.033324e61dc3f972", scope: null, file: !42, line: 115, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !80)
// CHECK:STDOUT: !80 = !{!81}
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !79, type: !19)
// CHECK:STDOUT: !82 = !DILocation(line: 116, column: 12, scope: !79)
// CHECK:STDOUT: !83 = !DILocation(line: 116, column: 5, scope: !79)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk.Int.Core:AddAssignWith.Core.532bdfe78dd580b3", scope: null, file: !55, line: 275, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !87)
// CHECK:STDOUT: !85 = !DISubroutineType(types: !86)
// CHECK:STDOUT: !86 = !{null, !18, !18}
// CHECK:STDOUT: !87 = !{!88, !89}
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !84, type: !18)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 2, scope: !84, type: !18)
// CHECK:STDOUT: !90 = !DILocation(line: 4294967295, scope: !84)
// CHECK:STDOUT: !91 = !DILocation(line: 275, column: 3, scope: !84)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e5bf07b1b9c2cbc2:OptionalStorage.Core.033324e61dc3f972", scope: null, file: !42, line: 104, type: !63, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !93)
// CHECK:STDOUT: !93 = !{!94}
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !92, type: !18)
// CHECK:STDOUT: !95 = !DILocation(line: 108, column: 5, scope: !92)
// CHECK:STDOUT: !96 = !DILocation(line: 109, column: 5, scope: !92)
// CHECK:STDOUT: !97 = !DILocation(line: 110, column: 5, scope: !92)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "None", linkageName: "_CNone.e5bf07b1b9c2cbc2:OptionalStorage.Core.033324e61dc3f972", scope: null, file: !42, line: 99, type: !70, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !99 = !DILocation(line: 101, column: 5, scope: !98)
// CHECK:STDOUT: !100 = !DILocation(line: 102, column: 5, scope: !98)
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e5bf07b1b9c2cbc2:ImplicitAs.Core.033324e61dc3f972", scope: null, file: !102, line: 24, type: !103, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !105)
// CHECK:STDOUT: !102 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !103 = !DISubroutineType(types: !104)
// CHECK:STDOUT: !104 = !{!18, !18}
// CHECK:STDOUT: !105 = !{!106}
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !101, type: !18)
// CHECK:STDOUT: !107 = !DILocation(line: 24, column: 38, scope: !101)
