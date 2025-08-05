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
// CHECK:STDOUT: @array.loc16_43.22 = internal constant [6 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_43.3.temp = alloca [6 x i32], align 4, !dbg !7
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %.loc17_19.1.temp = alloca { i1, i32 }, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 24, ptr %.loc16_43.3.temp), !dbg !7
// CHECK:STDOUT:   %.loc16_43.4.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 0, !dbg !7
// CHECK:STDOUT:   %.loc16_43.7.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 1, !dbg !7
// CHECK:STDOUT:   %.loc16_43.10.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 2, !dbg !7
// CHECK:STDOUT:   %.loc16_43.13.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 3, !dbg !7
// CHECK:STDOUT:   %.loc16_43.16.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 4, !dbg !7
// CHECK:STDOUT:   %.loc16_43.19.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 5, !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %.loc16_43.3.temp, ptr align 4 @array.loc16_43.22, i64 24, i1 false), !dbg !7
// CHECK:STDOUT:   %array_type.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.bb5cc7788b91e55e:Iterate.Core.b2c3e6e8ff6963ef"(ptr %.loc16_43.3.temp), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %array_type.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %.loc17_19.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.bb5cc7788b91e55e:Iterate.Core.b2c3e6e8ff6963ef"(ptr %.loc17_19.1.temp, ptr %.loc16_43.3.temp, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.b88d1103f417c6d4(ptr %.loc17_19.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.b88d1103f417c6d4(ptr %.loc17_19.1.temp), !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(i32 %Optional.Get.call), !dbg !9
// CHECK:STDOUT:   br label %for.next, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.bb5cc7788b91e55e:Iterate.Core.b2c3e6e8ff6963ef"(ptr %self) !dbg !12 {
// CHECK:STDOUT:   ret i32 0, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_CNext.bb5cc7788b91e55e:Iterate.Core.b2c3e6e8ff6963ef"(ptr sret({ i1, i32 }) %return, ptr %self, ptr %cursor) !dbg !15 {
// CHECK:STDOUT:   %1 = load i32, ptr %cursor, align 4, !dbg !16
// CHECK:STDOUT:   %2 = icmp slt i32 %1, 6, !dbg !16
// CHECK:STDOUT:   br i1 %2, label %3, label %7, !dbg !17
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !18
// CHECK:STDOUT:   %4 = load i32, ptr %cursor, align 4, !dbg !19
// CHECK:STDOUT:   %5 = sub i32 %4, 1, !dbg !19
// CHECK:STDOUT:   %array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i32 %5, !dbg !20
// CHECK:STDOUT:   %6 = load i32, ptr %array.index, align 4, !dbg !20
// CHECK:STDOUT:   call void @_CSome.Optional.Core.b88d1103f417c6d4(ptr %return, i32 %6), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.b88d1103f417c6d4(ptr %return), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.b88d1103f417c6d4(ptr %self) !dbg !25 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 0, !dbg !27
// CHECK:STDOUT:   %1 = load i1, ptr %has_value, align 1, !dbg !27
// CHECK:STDOUT:   ret i1 %1, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.b88d1103f417c6d4(ptr %self) !dbg !29 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 1, !dbg !30
// CHECK:STDOUT:   %1 = load i32, ptr %value, align 4, !dbg !30
// CHECK:STDOUT:   ret i32 %1, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) !dbg !32 {
// CHECK:STDOUT:   call void @"_COp:thunk.Int.Core:AddAssignWith.Core.a6f9794233e6e547"(ptr %self, i32 1), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.b88d1103f417c6d4(ptr sret({ i1, i32 }) %return, i32 %value) !dbg !36 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i1, i32 }, ptr %return, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { i1, i32 }, ptr %return, i32 0, i32 1, !dbg !37
// CHECK:STDOUT:   store i32 %value, ptr %value1, align 4, !dbg !37
// CHECK:STDOUT:   store i1 true, ptr %has_value, align 1, !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.b88d1103f417c6d4(ptr sret({ i1, i32 }) %return) !dbg !39 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i1, i32 }, ptr %return, i32 0, i32 0, !dbg !40
// CHECK:STDOUT:   store i1 false, ptr %has_value, align 1, !dbg !40
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk.Int.Core:AddAssignWith.Core.a6f9794233e6e547"(ptr %self, i32 %other) !dbg !42 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.8b3d5d6a6c17be04:ImplicitAs.Core.1abd39b699024258"(i32 %other), !dbg !43
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !44
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !44
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !44
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.8b3d5d6a6c17be04:ImplicitAs.Core.1abd39b699024258"(i32 %self) !dbg !45 {
// CHECK:STDOUT:   ret i32 %self, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "iterate.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 16, column: 26, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 17, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 18, column: 5, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 1, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.bb5cc7788b91e55e:Iterate.Core.b2c3e6e8ff6963ef", scope: null, file: !13, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !14 = !DILocation(line: 21, column: 39, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.bb5cc7788b91e55e:Iterate.Core.b2c3e6e8ff6963ef", scope: null, file: !13, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 23, column: 9, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 23, column: 8, scope: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 24, column: 7, scope: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 25, column: 36, scope: !15)
// CHECK:STDOUT: !20 = !DILocation(line: 25, column: 31, scope: !15)
// CHECK:STDOUT: !21 = !DILocation(line: 25, column: 14, scope: !15)
// CHECK:STDOUT: !22 = !DILocation(line: 25, column: 7, scope: !15)
// CHECK:STDOUT: !23 = !DILocation(line: 27, column: 14, scope: !15)
// CHECK:STDOUT: !24 = !DILocation(line: 27, column: 7, scope: !15)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.b88d1103f417c6d4", scope: null, file: !26, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !26 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !27 = !DILocation(line: 29, column: 46, scope: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 29, column: 39, scope: !25)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.b88d1103f417c6d4", scope: null, file: !26, line: 30, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !30 = !DILocation(line: 30, column: 38, scope: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 30, column: 31, scope: !29)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !33, line: 332, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !33 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !34 = !DILocation(line: 334, column: 5, scope: !32)
// CHECK:STDOUT: !35 = !DILocation(line: 332, column: 3, scope: !32)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.b88d1103f417c6d4", scope: null, file: !26, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !37 = !DILocation(line: 26, column: 12, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 26, column: 5, scope: !36)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.b88d1103f417c6d4", scope: null, file: !26, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !40 = !DILocation(line: 21, column: 5, scope: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 22, column: 5, scope: !39)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk.Int.Core:AddAssignWith.Core.a6f9794233e6e547", scope: null, file: !33, line: 268, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !43 = !DILocation(line: 4294967295, scope: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 268, column: 3, scope: !42)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.8b3d5d6a6c17be04:ImplicitAs.Core.1abd39b699024258", scope: null, file: !46, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !46 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !47 = !DILocation(line: 18, column: 38, scope: !45)
