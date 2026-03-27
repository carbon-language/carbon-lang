// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/iterate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/iterate.carbon

fn G(n: i32);

fn F() {
  let a: array(i32, 6) = (1, 2, 3, 4, 5, 6);
  for (n: i32 in a) {
    G(n);
  }
}

// CHECK:STDOUT: ; ModuleID = 'iterate.carbon'
// CHECK:STDOUT: source_filename = "iterate.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array = internal constant [6 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_43.3.temp = alloca [6 x i32], align 4, !dbg !7
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %.loc17_19.1.temp = alloca { i32, i1 }, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_43.3.temp), !dbg !7
// CHECK:STDOUT:   %.loc16_43.4.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 0, !dbg !7
// CHECK:STDOUT:   %.loc16_43.7.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 1, !dbg !7
// CHECK:STDOUT:   %.loc16_43.10.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 2, !dbg !7
// CHECK:STDOUT:   %.loc16_43.13.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 3, !dbg !7
// CHECK:STDOUT:   %.loc16_43.16.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 4, !dbg !7
// CHECK:STDOUT:   %.loc16_43.19.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 5, !dbg !7
// CHECK:STDOUT:   %array_type.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.7711e6f0e1563534:Iterate.Core.355bb079bb00aa16"(ptr @array), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %array_type.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_19.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.7711e6f0e1563534:Iterate.Core.355bb079bb00aa16"(ptr %.loc17_19.1.temp, ptr @array, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.b5e6acce74484d77(ptr %.loc17_19.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.b5e6acce74484d77(ptr %.loc17_19.1.temp), !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(i32 %Optional.Get.call), !dbg !9
// CHECK:STDOUT:   br label %for.next, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.7711e6f0e1563534:Iterate.Core.355bb079bb00aa16"(ptr %self) #0 !dbg !12 {
// CHECK:STDOUT:   ret i32 0, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.7711e6f0e1563534:Iterate.Core.355bb079bb00aa16"(ptr sret({ i32, i1 }) %return, ptr %self, ptr %cursor) #0 !dbg !21 {
// CHECK:STDOUT:   %1 = load i32, ptr %cursor, align 4, !dbg !27
// CHECK:STDOUT:   %2 = icmp slt i32 %1, 6, !dbg !27
// CHECK:STDOUT:   br i1 %2, label %3, label %7, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !29
// CHECK:STDOUT:   %4 = load i32, ptr %cursor, align 4, !dbg !30
// CHECK:STDOUT:   %5 = sub i32 %4, 1, !dbg !30
// CHECK:STDOUT:   %array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i32 %5, !dbg !31
// CHECK:STDOUT:   %6 = load i32, ptr %array.index, align 4, !dbg !31
// CHECK:STDOUT:   call void @_CSome.Optional.Core.b5e6acce74484d77(ptr %return, i32 %6), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.b5e6acce74484d77(ptr %return), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.b5e6acce74484d77(ptr %self) #0 !dbg !36 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1"(ptr %self), !dbg !42
// CHECK:STDOUT:   ret i1 %1, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.b5e6acce74484d77(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1"(ptr %self), !dbg !47
// CHECK:STDOUT:   ret i32 %1, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT:   call void @"_COp:thunk.Int.9452f4c51951679b.Core:AddAssignWith.e2da7149b7fcf782.Core.23ea0ff7335d4fdc"(ptr %self, i32 1), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.b5e6acce74484d77(ptr sret({ i32, i1 }) %return, i32 %value) #0 !dbg !57 {
// CHECK:STDOUT:   call void @"_CSome.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1"(ptr %return, i32 %value), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.b5e6acce74484d77(ptr sret({ i32, i1 }) %return) #0 !dbg !64 {
// CHECK:STDOUT:   call void @"_CNone.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1"(ptr %return), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1"(ptr %value) #0 !dbg !69 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %value, i32 0, i32 1, !dbg !72
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !72
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !72
// CHECK:STDOUT:   ret i1 %2, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1"(ptr %value) #0 !dbg !74 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { i32, i1 }, ptr %value, i32 0, i32 0, !dbg !77
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !77
// CHECK:STDOUT:   ret i32 %1, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk.Int.9452f4c51951679b.Core:AddAssignWith.e2da7149b7fcf782.Core.23ea0ff7335d4fdc"(ptr %self, i32 %other) #2 !dbg !79 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.93210ed0b06d6c25:ImplicitAs.11da3bb3c3dce533.Core.5da21217b3d19eab"(i32 %other), !dbg !85
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !86
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !86
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !86
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !87 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 0, !dbg !90
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !90
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !91
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !91
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1"(ptr sret({ i32, i1 }) %return) #0 !dbg !93 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !94
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !94
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.93210ed0b06d6c25:ImplicitAs.11da3bb3c3dce533.Core.5da21217b3d19eab"(i32 %self) #0 !dbg !96 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !101
// CHECK:STDOUT:   ret i32 %1, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !103 {
// CHECK:STDOUT:   ret i32 %self, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "iterate.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 16, column: 26, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 17, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 18, column: 5, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 1, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.7711e6f0e1563534:Iterate.Core.355bb079bb00aa16", scope: null, file: !13, line: 23, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
// CHECK:STDOUT: !13 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{!16, !17}
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !12, type: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 23, column: 46, scope: !12)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.7711e6f0e1563534:Iterate.Core.355bb079bb00aa16", scope: null, file: !13, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !22 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !23 = !{!17, !17, !17}
// CHECK:STDOUT: !24 = !{!25, !26}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !21, type: !17)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 2, scope: !21, type: !17)
// CHECK:STDOUT: !27 = !DILocation(line: 25, column: 9, scope: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 25, column: 8, scope: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 26, column: 7, scope: !21)
// CHECK:STDOUT: !30 = !DILocation(line: 27, column: 36, scope: !21)
// CHECK:STDOUT: !31 = !DILocation(line: 27, column: 31, scope: !21)
// CHECK:STDOUT: !32 = !DILocation(line: 27, column: 14, scope: !21)
// CHECK:STDOUT: !33 = !DILocation(line: 27, column: 7, scope: !21)
// CHECK:STDOUT: !34 = !DILocation(line: 29, column: 14, scope: !21)
// CHECK:STDOUT: !35 = !DILocation(line: 29, column: 7, scope: !21)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.b5e6acce74484d77", scope: null, file: !37, line: 33, type: !38, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !40)
// CHECK:STDOUT: !37 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !38 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !39 = !{!17, !17}
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !36, type: !17)
// CHECK:STDOUT: !42 = !DILocation(line: 34, column: 12, scope: !36)
// CHECK:STDOUT: !43 = !DILocation(line: 34, column: 5, scope: !36)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.b5e6acce74484d77", scope: null, file: !37, line: 36, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !45)
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !44, type: !17)
// CHECK:STDOUT: !47 = !DILocation(line: 37, column: 12, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 37, column: 5, scope: !44)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !50, line: 363, type: !51, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !53)
// CHECK:STDOUT: !50 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !51 = !DISubroutineType(types: !52)
// CHECK:STDOUT: !52 = !{null, !16}
// CHECK:STDOUT: !53 = !{!54}
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !49, type: !16)
// CHECK:STDOUT: !55 = !DILocation(line: 365, column: 5, scope: !49)
// CHECK:STDOUT: !56 = !DILocation(line: 363, column: 3, scope: !49)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.b5e6acce74484d77", scope: null, file: !37, line: 30, type: !58, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !59 = !{!17, !16}
// CHECK:STDOUT: !60 = !{!61}
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !57, type: !16)
// CHECK:STDOUT: !62 = !DILocation(line: 31, column: 12, scope: !57)
// CHECK:STDOUT: !63 = !DILocation(line: 31, column: 5, scope: !57)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.b5e6acce74484d77", scope: null, file: !37, line: 27, type: !65, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !65 = !DISubroutineType(types: !66)
// CHECK:STDOUT: !66 = !{!17}
// CHECK:STDOUT: !67 = !DILocation(line: 28, column: 12, scope: !64)
// CHECK:STDOUT: !68 = !DILocation(line: 28, column: 5, scope: !64)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1", scope: null, file: !37, line: 132, type: !38, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !70)
// CHECK:STDOUT: !70 = !{!71}
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !69, type: !17)
// CHECK:STDOUT: !72 = !DILocation(line: 133, column: 12, scope: !69)
// CHECK:STDOUT: !73 = !DILocation(line: 133, column: 5, scope: !69)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1", scope: null, file: !37, line: 135, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !75)
// CHECK:STDOUT: !75 = !{!76}
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !74, type: !17)
// CHECK:STDOUT: !77 = !DILocation(line: 136, column: 12, scope: !74)
// CHECK:STDOUT: !78 = !DILocation(line: 136, column: 5, scope: !74)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk.Int.9452f4c51951679b.Core:AddAssignWith.e2da7149b7fcf782.Core.23ea0ff7335d4fdc", scope: null, file: !50, line: 299, type: !80, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !82)
// CHECK:STDOUT: !80 = !DISubroutineType(types: !81)
// CHECK:STDOUT: !81 = !{null, !16, !16}
// CHECK:STDOUT: !82 = !{!83, !84}
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !79, type: !16)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 2, scope: !79, type: !16)
// CHECK:STDOUT: !85 = !DILocation(line: 4294967295, scope: !79)
// CHECK:STDOUT: !86 = !DILocation(line: 299, column: 3, scope: !79)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1", scope: null, file: !37, line: 122, type: !58, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !88)
// CHECK:STDOUT: !88 = !{!89}
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !87, type: !16)
// CHECK:STDOUT: !90 = !DILocation(line: 128, column: 5, scope: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 129, column: 5, scope: !87)
// CHECK:STDOUT: !92 = !DILocation(line: 130, column: 5, scope: !87)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "None", linkageName: "_CNone.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1", scope: null, file: !37, line: 116, type: !65, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !94 = !DILocation(line: 119, column: 5, scope: !93)
// CHECK:STDOUT: !95 = !DILocation(line: 120, column: 5, scope: !93)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.93210ed0b06d6c25:ImplicitAs.11da3bb3c3dce533.Core.5da21217b3d19eab", scope: null, file: !50, line: 62, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !99)
// CHECK:STDOUT: !97 = !DISubroutineType(types: !98)
// CHECK:STDOUT: !98 = !{!16, !16}
// CHECK:STDOUT: !99 = !{!100}
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !96, type: !16)
// CHECK:STDOUT: !101 = !DILocation(line: 64, column: 17, scope: !96)
// CHECK:STDOUT: !102 = !DILocation(line: 64, column: 5, scope: !96)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !50, line: 57, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !104)
// CHECK:STDOUT: !104 = !{!105}
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !103, type: !16)
// CHECK:STDOUT: !106 = !DILocation(line: 57, column: 36, scope: !103)
