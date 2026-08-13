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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'iterate.carbon'
// CHECK:STDOUT: source_filename = "iterate.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array = internal constant [6 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6]
// CHECK:STDOUT: @array.loc16_43.24 = internal constant [6 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_43.3.temp = alloca [6 x i32], align 4, !dbg !7
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %.loc17_19.5.temp = alloca <{ i32, i1 }>, align 4, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_43.3.temp), !dbg !7
// CHECK:STDOUT:   %.loc16_43.4.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 0, !dbg !7
// CHECK:STDOUT:   %.loc16_43.7.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 1, !dbg !7
// CHECK:STDOUT:   %.loc16_43.10.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 2, !dbg !7
// CHECK:STDOUT:   %.loc16_43.13.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 3, !dbg !7
// CHECK:STDOUT:   %.loc16_43.16.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 4, !dbg !7
// CHECK:STDOUT:   %.loc16_43.19.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 5, !dbg !7
// CHECK:STDOUT:   %array_type.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.e163a239f9fa927c:Iterate.Core.620427d555024fe2"(ptr @array.loc16_43.24), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %array_type.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.e163a239f9fa927c:Iterate.Core.620427d555024fe2"(ptr %.loc17_19.5.temp, ptr @array.loc16_43.24, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.8698ccdc411f2307(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.8698ccdc411f2307(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(i32 %Optional.Get.call), !dbg !9
// CHECK:STDOUT:   call void @"_COp.9619ac5c2f73b9b5:core.Destroy.Core"(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @"_COp.9619ac5c2f73b9b5:core.Destroy.Core"(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.3a5df7cec745952b:core.Destroy.Core"(ptr @array), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.1155b488c23b5086:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.830bbdcf8982aa42:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94ba1c30d421702e:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0fc9634857315db2:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.69700c1fc1b2744c:core.Destroy.Core"(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f49c06ba556be406:core.Destroy.Core"(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9619ac5c2f73b9b5:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3a5df7cec745952b:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.e163a239f9fa927c:Iterate.Core.620427d555024fe2"(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT:   ret i32 0, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.e163a239f9fa927c:Iterate.Core.620427d555024fe2"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !53 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !59
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !60
// CHECK:STDOUT:   %3 = icmp slt i32 %2, 6, !dbg !60
// CHECK:STDOUT:   br i1 %3, label %4, label %8, !dbg !61
// CHECK:STDOUT:
// CHECK:STDOUT: 4:                                                ; preds = %0
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !59
// CHECK:STDOUT:   %5 = load i32, ptr %cursor, align 4, !dbg !62
// CHECK:STDOUT:   store i32 %5, ptr %1, align 4, !dbg !59
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !63
// CHECK:STDOUT:   %6 = load i32, ptr %1, align 4, !dbg !64
// CHECK:STDOUT:   %array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i32 %6, !dbg !65
// CHECK:STDOUT:   %7 = load i32, ptr %array.index, align 4, !dbg !65
// CHECK:STDOUT:   call void @_CSome.Optional.Core.8698ccdc411f2307(ptr %return, i32 %7), !dbg !66
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %1), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.8698ccdc411f2307(ptr %return), !dbg !68
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.8698ccdc411f2307(ptr %self) #0 !dbg !70 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %self), !dbg !76
// CHECK:STDOUT:   ret i1 %1, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.8698ccdc411f2307(ptr %self) #0 !dbg !78 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %self), !dbg !81
// CHECK:STDOUT:   ret i32 %1, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.df57f49075d1fe6b:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.6f9e8c1abf8bb3ea.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.224cb1ed5cb3f064"(ptr %self, i32 1), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.8698ccdc411f2307(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !89 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %return, i32 %value), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.8698ccdc411f2307(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !96 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %return), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %value) #0 !dbg !101 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !104
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !104
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !104
// CHECK:STDOUT:   ret i1 %2, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %value) #0 !dbg !106 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !109
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !109
// CHECK:STDOUT:   ret i32 %1, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.6f9e8c1abf8bb3ea.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.224cb1ed5cb3f064"(ptr %self, i32 %other) #2 !dbg !111 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.f0377c795e793659"(i32 %other), !dbg !117
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !118
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !118
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !118
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !119 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !122
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !122
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !123
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !123
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !125 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !126
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.f0377c795e793659"(i32 %self) #0 !dbg !128 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !133
// CHECK:STDOUT:   ret i32 %1, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !135 {
// CHECK:STDOUT:   ret i32 %self, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.9619ac5c2f73b9b5:core.Destroy.Core", { 1, 0 }
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
// CHECK:STDOUT: !1 = !DIFile(filename: "iterate.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 16, column: 26, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 17, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 18, column: 5, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 1, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !1, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 17, column: 7, scope: !12)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94ba1c30d421702e:core.Destroy.Core", scope: null, file: !1, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null, !22}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !19, type: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 17, column: 7, scope: !19)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0fc9634857315db2:core.Destroy.Core", scope: null, file: !1, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !27)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !26, type: !15)
// CHECK:STDOUT: !29 = !DILocation(line: 17, column: 7, scope: !26)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Op", linkageName: "_COp.69700c1fc1b2744c:core.Destroy.Core", scope: null, file: !1, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !31)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !30, type: !22)
// CHECK:STDOUT: !33 = !DILocation(line: 17, column: 7, scope: !30)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f49c06ba556be406:core.Destroy.Core", scope: null, file: !1, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !22)
// CHECK:STDOUT: !37 = !DILocation(line: 17, column: 7, scope: !34)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9619ac5c2f73b9b5:core.Destroy.Core", scope: null, file: !1, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !22)
// CHECK:STDOUT: !41 = !DILocation(line: 17, column: 7, scope: !38)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3a5df7cec745952b:core.Destroy.Core", scope: null, file: !1, line: 16, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !43)
// CHECK:STDOUT: !43 = !{!44}
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !42, type: !22)
// CHECK:STDOUT: !45 = !DILocation(line: 16, column: 26, scope: !42)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.e163a239f9fa927c:Iterate.Core.620427d555024fe2", scope: null, file: !47, line: 23, type: !48, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !50)
// CHECK:STDOUT: !47 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !48 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !49 = !{!15, !22}
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !46, type: !22)
// CHECK:STDOUT: !52 = !DILocation(line: 23, column: 38, scope: !46)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.e163a239f9fa927c:Iterate.Core.620427d555024fe2", scope: null, file: !47, line: 24, type: !54, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !55 = !{!22, !22, !22}
// CHECK:STDOUT: !56 = !{!57, !58}
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !53, type: !22)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 2, scope: !53, type: !22)
// CHECK:STDOUT: !59 = !DILocation(line: 26, column: 7, scope: !53)
// CHECK:STDOUT: !60 = !DILocation(line: 25, column: 9, scope: !53)
// CHECK:STDOUT: !61 = !DILocation(line: 25, column: 8, scope: !53)
// CHECK:STDOUT: !62 = !DILocation(line: 26, column: 25, scope: !53)
// CHECK:STDOUT: !63 = !DILocation(line: 27, column: 7, scope: !53)
// CHECK:STDOUT: !64 = !DILocation(line: 28, column: 36, scope: !53)
// CHECK:STDOUT: !65 = !DILocation(line: 28, column: 31, scope: !53)
// CHECK:STDOUT: !66 = !DILocation(line: 28, column: 14, scope: !53)
// CHECK:STDOUT: !67 = !DILocation(line: 28, column: 7, scope: !53)
// CHECK:STDOUT: !68 = !DILocation(line: 30, column: 14, scope: !53)
// CHECK:STDOUT: !69 = !DILocation(line: 30, column: 7, scope: !53)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.8698ccdc411f2307", scope: null, file: !71, line: 36, type: !72, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !74)
// CHECK:STDOUT: !71 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !72 = !DISubroutineType(types: !73)
// CHECK:STDOUT: !73 = !{!22, !22}
// CHECK:STDOUT: !74 = !{!75}
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !70, type: !22)
// CHECK:STDOUT: !76 = !DILocation(line: 37, column: 12, scope: !70)
// CHECK:STDOUT: !77 = !DILocation(line: 37, column: 5, scope: !70)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.8698ccdc411f2307", scope: null, file: !71, line: 39, type: !48, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !79)
// CHECK:STDOUT: !79 = !{!80}
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !78, type: !22)
// CHECK:STDOUT: !81 = !DILocation(line: 40, column: 12, scope: !78)
// CHECK:STDOUT: !82 = !DILocation(line: 40, column: 5, scope: !78)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !84, line: 396, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !85)
// CHECK:STDOUT: !84 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !85 = !{!86}
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !83, type: !15)
// CHECK:STDOUT: !87 = !DILocation(line: 398, column: 5, scope: !83)
// CHECK:STDOUT: !88 = !DILocation(line: 396, column: 3, scope: !83)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.8698ccdc411f2307", scope: null, file: !71, line: 33, type: !90, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !92)
// CHECK:STDOUT: !90 = !DISubroutineType(types: !91)
// CHECK:STDOUT: !91 = !{!22, !15}
// CHECK:STDOUT: !92 = !{!93}
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !89, type: !15)
// CHECK:STDOUT: !94 = !DILocation(line: 34, column: 12, scope: !89)
// CHECK:STDOUT: !95 = !DILocation(line: 34, column: 5, scope: !89)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.8698ccdc411f2307", scope: null, file: !71, line: 30, type: !97, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !97 = !DISubroutineType(types: !98)
// CHECK:STDOUT: !98 = !{!22}
// CHECK:STDOUT: !99 = !DILocation(line: 31, column: 12, scope: !96)
// CHECK:STDOUT: !100 = !DILocation(line: 31, column: 5, scope: !96)
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !71, line: 141, type: !72, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !102)
// CHECK:STDOUT: !102 = !{!103}
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !101, type: !22)
// CHECK:STDOUT: !104 = !DILocation(line: 142, column: 12, scope: !101)
// CHECK:STDOUT: !105 = !DILocation(line: 142, column: 5, scope: !101)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !71, line: 144, type: !48, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !107)
// CHECK:STDOUT: !107 = !{!108}
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !106, type: !22)
// CHECK:STDOUT: !109 = !DILocation(line: 145, column: 12, scope: !106)
// CHECK:STDOUT: !110 = !DILocation(line: 145, column: 5, scope: !106)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.6f9e8c1abf8bb3ea.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.224cb1ed5cb3f064", scope: null, file: !84, line: 331, type: !112, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !114)
// CHECK:STDOUT: !112 = !DISubroutineType(types: !113)
// CHECK:STDOUT: !113 = !{null, !15, !15}
// CHECK:STDOUT: !114 = !{!115, !116}
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !111, type: !15)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 2, scope: !111, type: !15)
// CHECK:STDOUT: !117 = !DILocation(line: 0, scope: !111)
// CHECK:STDOUT: !118 = !DILocation(line: 331, column: 3, scope: !111)
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !71, line: 131, type: !90, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !120)
// CHECK:STDOUT: !120 = !{!121}
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !119, type: !15)
// CHECK:STDOUT: !122 = !DILocation(line: 137, column: 5, scope: !119)
// CHECK:STDOUT: !123 = !DILocation(line: 138, column: 5, scope: !119)
// CHECK:STDOUT: !124 = !DILocation(line: 139, column: 5, scope: !119)
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !71, line: 125, type: !97, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !126 = !DILocation(line: 128, column: 5, scope: !125)
// CHECK:STDOUT: !127 = !DILocation(line: 129, column: 5, scope: !125)
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.f0377c795e793659", scope: null, file: !84, line: 64, type: !129, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !131)
// CHECK:STDOUT: !129 = !DISubroutineType(types: !130)
// CHECK:STDOUT: !130 = !{!15, !15}
// CHECK:STDOUT: !131 = !{!132}
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !128, type: !15)
// CHECK:STDOUT: !133 = !DILocation(line: 66, column: 17, scope: !128)
// CHECK:STDOUT: !134 = !DILocation(line: 66, column: 5, scope: !128)
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !84, line: 59, type: !129, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !136)
// CHECK:STDOUT: !136 = !{!137}
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !135, type: !15)
// CHECK:STDOUT: !138 = !DILocation(line: 59, column: 28, scope: !135)
// CHECK:STDOUT:
