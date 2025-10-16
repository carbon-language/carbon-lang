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
  for (n: i32 in Core.Range(100)) {
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
// CHECK:STDOUT: define void @_CFor.Main() !dbg !4 {
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
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.30bf744b66445e68(ptr %.loc21_33.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.30bf744b66445e68(ptr %.loc21_33.1.temp), !dbg !8
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
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) !dbg !14 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !16
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !16
// CHECK:STDOUT:   ret i32 %1, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret({ i32, i1 }) %return, ptr %self, ptr %cursor) !dbg !18 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !19
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !20
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !19
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !21
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !21
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !22
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !22
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !23
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !24
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !25
// CHECK:STDOUT:   call void @_CSome.Optional.Core.30bf744b66445e68(ptr %return, i32 %7), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.30bf744b66445e68(ptr %return), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.30bf744b66445e68(ptr %self) !dbg !30 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.2a0d145dae8e0e7f:OptionalStorage.Core.de631560529e9861"(ptr %self), !dbg !32
// CHECK:STDOUT:   ret i1 %1, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.30bf744b66445e68(ptr %self) !dbg !34 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.2a0d145dae8e0e7f:OptionalStorage.Core.de631560529e9861"(ptr %self), !dbg !35
// CHECK:STDOUT:   ret i32 %1, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) !dbg !37 {
// CHECK:STDOUT:   call void @"_COp:thunk.Int.Core:AddAssignWith.Core.2122e6c8c4f4206d"(ptr %self, i32 1), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.30bf744b66445e68(ptr sret({ i32, i1 }) %return, i32 %value) !dbg !41 {
// CHECK:STDOUT:   call void @"_CSome.2a0d145dae8e0e7f:OptionalStorage.Core.de631560529e9861"(ptr %return, i32 %value), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.30bf744b66445e68(ptr sret({ i32, i1 }) %return) !dbg !44 {
// CHECK:STDOUT:   call void @"_CNone.2a0d145dae8e0e7f:OptionalStorage.Core.de631560529e9861"(ptr %return), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.2a0d145dae8e0e7f:OptionalStorage.Core.de631560529e9861"(ptr %value) !dbg !47 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %value, i32 0, i32 1, !dbg !48
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !48
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !48
// CHECK:STDOUT:   ret i1 %2, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.2a0d145dae8e0e7f:OptionalStorage.Core.de631560529e9861"(ptr %value) !dbg !50 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { i32, i1 }, ptr %value, i32 0, i32 0, !dbg !51
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !51
// CHECK:STDOUT:   ret i32 %1, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk.Int.Core:AddAssignWith.Core.2122e6c8c4f4206d"(ptr %self, i32 %other) !dbg !53 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.2a0d145dae8e0e7f:ImplicitAs.Core.de631560529e9861"(i32 %other), !dbg !54
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !55
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !55
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !55
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_CSome.2a0d145dae8e0e7f:OptionalStorage.Core.de631560529e9861"(ptr sret({ i32, i1 }) %return, i32 %self) !dbg !56 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 0, !dbg !57
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !57
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !58
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !58
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_CNone.2a0d145dae8e0e7f:OptionalStorage.Core.de631560529e9861"(ptr sret({ i32, i1 }) %return) !dbg !60 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !61
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !61
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.2a0d145dae8e0e7f:ImplicitAs.Core.de631560529e9861"(i32 %self) !dbg !63 {
// CHECK:STDOUT:   ret i32 %self, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "for.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 21, column: 18, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 21, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 22, column: 5, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 21, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 24, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 19, column: 1, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !15, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !15 = !DIFile(filename: "{{.*}}/range.carbon", directory: "")
// CHECK:STDOUT: !16 = !DILocation(line: 25, column: 51, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 25, column: 44, scope: !14)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !15, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 27, column: 7, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 27, column: 27, scope: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 28, column: 19, scope: !18)
// CHECK:STDOUT: !22 = !DILocation(line: 28, column: 11, scope: !18)
// CHECK:STDOUT: !23 = !DILocation(line: 28, column: 10, scope: !18)
// CHECK:STDOUT: !24 = !DILocation(line: 29, column: 9, scope: !18)
// CHECK:STDOUT: !25 = !DILocation(line: 30, column: 38, scope: !18)
// CHECK:STDOUT: !26 = !DILocation(line: 30, column: 16, scope: !18)
// CHECK:STDOUT: !27 = !DILocation(line: 30, column: 9, scope: !18)
// CHECK:STDOUT: !28 = !DILocation(line: 32, column: 16, scope: !18)
// CHECK:STDOUT: !29 = !DILocation(line: 32, column: 9, scope: !18)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.30bf744b66445e68", scope: null, file: !31, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !31 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !32 = !DILocation(line: 32, column: 12, scope: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 32, column: 5, scope: !30)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.30bf744b66445e68", scope: null, file: !31, line: 34, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !35 = !DILocation(line: 35, column: 12, scope: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 35, column: 5, scope: !34)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !38, line: 339, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !38 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !39 = !DILocation(line: 341, column: 5, scope: !37)
// CHECK:STDOUT: !40 = !DILocation(line: 339, column: 3, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.30bf744b66445e68", scope: null, file: !31, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !42 = !DILocation(line: 29, column: 12, scope: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 29, column: 5, scope: !41)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.30bf744b66445e68", scope: null, file: !31, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !45 = !DILocation(line: 26, column: 12, scope: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 26, column: 5, scope: !44)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.2a0d145dae8e0e7f:OptionalStorage.Core.de631560529e9861", scope: null, file: !31, line: 73, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !48 = !DILocation(line: 74, column: 12, scope: !47)
// CHECK:STDOUT: !49 = !DILocation(line: 74, column: 5, scope: !47)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.2a0d145dae8e0e7f:OptionalStorage.Core.de631560529e9861", scope: null, file: !31, line: 76, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !51 = !DILocation(line: 77, column: 12, scope: !50)
// CHECK:STDOUT: !52 = !DILocation(line: 77, column: 5, scope: !50)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk.Int.Core:AddAssignWith.Core.2122e6c8c4f4206d", scope: null, file: !38, line: 275, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !54 = !DILocation(line: 4294967295, scope: !53)
// CHECK:STDOUT: !55 = !DILocation(line: 275, column: 3, scope: !53)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.2a0d145dae8e0e7f:OptionalStorage.Core.de631560529e9861", scope: null, file: !31, line: 65, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !57 = !DILocation(line: 69, column: 5, scope: !56)
// CHECK:STDOUT: !58 = !DILocation(line: 70, column: 5, scope: !56)
// CHECK:STDOUT: !59 = !DILocation(line: 71, column: 5, scope: !56)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "None", linkageName: "_CNone.2a0d145dae8e0e7f:OptionalStorage.Core.de631560529e9861", scope: null, file: !31, line: 60, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !61 = !DILocation(line: 62, column: 5, scope: !60)
// CHECK:STDOUT: !62 = !DILocation(line: 63, column: 5, scope: !60)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.2a0d145dae8e0e7f:ImplicitAs.Core.de631560529e9861", scope: null, file: !64, line: 24, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !64 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !65 = !DILocation(line: 24, column: 38, scope: !63)
