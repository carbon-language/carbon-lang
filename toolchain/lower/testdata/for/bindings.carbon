// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/for/bindings.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/for/bindings.carbon

class EmptyRange(T: Core.Copy & Core.Destroy) {
  impl as Core.Iterate where .CursorType = {} and .ElementType = T {
    fn NewCursor(unused self) -> {} {
      return {};
    }
    fn Next(unused self, ref _: {}) -> Core.Optional(T) {
      return Core.Optional(T).None();
    }
  }
}

fn F(m: i32, n: i32*);

fn For() {
  var r: EmptyRange((i32, i32)) = {};

  for ((m: i32, var n: i32) in r) {
    F(m, &n);
  }
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'bindings.carbon'
// CHECK:STDOUT: source_filename = "bindings.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @EmptyRange.val.loc27_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(i32, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %r.var = alloca {}, align 1, !dbg !15
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !17
// CHECK:STDOUT:   %var = alloca {}, align 1, !dbg !20
// CHECK:STDOUT:   %.loc29_33.1.temp = alloca <{ { i32, i32 }, i1 }>, align 4, !dbg !20
// CHECK:STDOUT:   %.loc29_33.8.temp = alloca { i32, i32 }, align 4, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %r.var), !dbg !15
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %r.var, ptr align 1 @EmptyRange.val.loc27_3.1, i64 0, i1 false), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !17
// CHECK:STDOUT:   call void @"_CNewCursor.EmptyRange.96a698c2240e6755.Main:Iterate.Core.894714a7cfcffc26"(ptr %r.var), !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !20
// CHECK:STDOUT:   br label %for.next, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.1.temp), !dbg !20
// CHECK:STDOUT:   call void @"_CNext.EmptyRange.96a698c2240e6755.Main:Iterate.Core.894714a7cfcffc26"(ptr %.loc29_33.1.temp, ptr %r.var, ptr %var), !dbg !20
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.6a8897b1e38f8c5b(ptr %.loc29_33.1.temp), !dbg !20
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.8.temp), !dbg !20
// CHECK:STDOUT:   call void @_CGet.Optional.Core.6a8897b1e38f8c5b(ptr %.loc29_33.8.temp, ptr %.loc29_33.1.temp), !dbg !20
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc29_33.8.temp, i32 0, i32 0, !dbg !20
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc29_33.8.temp, i32 0, i32 1, !dbg !20
// CHECK:STDOUT:   %.loc29_33.11 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !20
// CHECK:STDOUT:   %.loc29_33.12 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !20
// CHECK:STDOUT:   store i32 %.loc29_33.12, ptr %n.var, align 4, !dbg !17
// CHECK:STDOUT:   call void @_CF.Main(i32 %.loc29_33.11, ptr %n.var), !dbg !30
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e75049736efa12f6(ptr %.loc29_33.8.temp), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.46435643ebb2ee3e(ptr %.loc29_33.1.temp), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %n.var), !dbg !17
// CHECK:STDOUT:   br label %for.next, !dbg !33
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.46435643ebb2ee3e(ptr %.loc29_33.1.temp), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %n.var), !dbg !17
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.82f9f5598387b9ac(ptr %var), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4405b7abc1074bea(ptr %r.var), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.c8c09ce0aee178f8:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %tuple.elem1.tuple.elem), !dbg !37
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %tuple.elem0.tuple.elem), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.167f1efaf2876080:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.29d802bf9e968066:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29_33.1.has_value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %self, i32 0, i32 1, !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.27eb7fea99346bf0(ptr %.loc29_33.1.has_value), !dbg !45
// CHECK:STDOUT:   %.loc29_33.4.value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %self, i32 0, i32 0, !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c28b24c1e5f4c659(ptr %.loc29_33.4.value), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f3c675338240c218:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29_33.1.has_value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %self, i32 0, i32 1, !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.27eb7fea99346bf0(ptr %.loc29_33.1.has_value), !dbg !49
// CHECK:STDOUT:   %.loc29_33.4.value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %self, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c28b24c1e5f4c659(ptr %.loc29_33.4.value), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNewCursor.EmptyRange.96a698c2240e6755.Main:Iterate.Core.894714a7cfcffc26"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.EmptyRange.96a698c2240e6755.Main:Iterate.Core.894714a7cfcffc26"(ptr sret(<{ { i32, i32 }, i1 }>) %return, ptr %self, ptr %_) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CNone.Optional.Core.6a8897b1e38f8c5b(ptr %return), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.6a8897b1e38f8c5b(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.81d99cc9206a1cd8:OptionalStorage.Core.894714a7cfcffc26"(ptr %self), !dbg !77
// CHECK:STDOUT:   ret i1 %1, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGet.Optional.Core.6a8897b1e38f8c5b(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !80 {
// CHECK:STDOUT:   call void @"_CGet.81d99cc9206a1cd8:OptionalStorage.Core.894714a7cfcffc26"(ptr %return, ptr %self), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e75049736efa12f6(ptr %t) #0 !dbg !90 {
// CHECK:STDOUT:   call void @"_COp.c8c09ce0aee178f8:core.SubobjectDestroy.Core"(ptr %t), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.46435643ebb2ee3e(ptr %t) #0 !dbg !97 {
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %t) #0 !dbg !104 {
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.82f9f5598387b9ac(ptr %t) #0 !dbg !111 {
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4405b7abc1074bea(ptr %t) #0 !dbg !118 {
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.27eb7fea99346bf0(ptr %t) #0 !dbg !125 {
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c28b24c1e5f4c659(ptr %t) #0 !dbg !132 {
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.6a8897b1e38f8c5b(ptr sret(<{ { i32, i32 }, i1 }>) %return) #0 !dbg !139 {
// CHECK:STDOUT:   call void @"_CNone.81d99cc9206a1cd8:OptionalStorage.Core.894714a7cfcffc26"(ptr %return), !dbg !142
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.81d99cc9206a1cd8:OptionalStorage.Core.894714a7cfcffc26"(ptr %value) #0 !dbg !144 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %value, i32 0, i32 1, !dbg !149
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !149
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !149
// CHECK:STDOUT:   ret i1 %2, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CGet.81d99cc9206a1cd8:OptionalStorage.Core.894714a7cfcffc26"(ptr sret({ i32, i32 }) %return, ptr %value) #0 !dbg !152 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %value, i32 0, i32 0, !dbg !157
// CHECK:STDOUT:   call void @"_COp.4e4edddc7580c79b:Copy.Core.166c3bb67ee40138"(ptr %return, ptr %value1), !dbg !157
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.81d99cc9206a1cd8:OptionalStorage.Core.894714a7cfcffc26"(ptr sret(<{ { i32, i32 }, i1 }>) %return) #0 !dbg !160 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %return, i32 0, i32 1, !dbg !167
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !167
// CHECK:STDOUT:   ret void, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.4e4edddc7580c79b:Copy.Core.166c3bb67ee40138"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !170 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !175
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !175
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !176
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !177
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !177
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !176
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !176
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !176
// CHECK:STDOUT:   ret void, !dbg !178
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.46435643ebb2ee3e, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.7f8dce37f94a83bf, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.27eb7fea99346bf0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c28b24c1e5f4c659, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "bindings.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{null}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{null, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !4, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DILocation(line: 27, column: 3, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 29, column: 17, scope: !14)
// CHECK:STDOUT: !20 = !DILocation(line: 29, column: 7, scope: !14)
// CHECK:STDOUT: !21 = !{!7, !7, !7}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{!7, !7}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 30, column: 5, scope: !14)
// CHECK:STDOUT: !31 = !{null, !6}
// CHECK:STDOUT: !32 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !33 = !DILocation(line: 29, column: 3, scope: !14)
// CHECK:STDOUT: !34 = !DILocation(line: 26, column: 1, scope: !14)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c8c09ce0aee178f8:core.SubobjectDestroy.Core", scope: null, file: !4, line: 27, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !7)
// CHECK:STDOUT: !37 = !DILocation(line: 27, column: 10, scope: !35)
// CHECK:STDOUT: !38 = !{!36}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp.167f1efaf2876080:core.SubobjectDestroy.Core", scope: null, file: !4, line: 29, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !7)
// CHECK:STDOUT: !41 = !DILocation(line: 29, column: 7, scope: !39)
// CHECK:STDOUT: !42 = !{!40}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Op", linkageName: "_COp.29d802bf9e968066:core.SubobjectDestroy.Core", scope: null, file: !4, line: 29, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !7)
// CHECK:STDOUT: !45 = !DILocation(line: 29, column: 7, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f3c675338240c218:core.SubobjectDestroy.Core", scope: null, file: !4, line: 29, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !7)
// CHECK:STDOUT: !49 = !DILocation(line: 29, column: 7, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.EmptyRange.96a698c2240e6755.Main:Iterate.Core.894714a7cfcffc26", scope: null, file: !4, line: 15, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !57)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !7)
// CHECK:STDOUT: !56 = !DILocation(line: 16, column: 7, scope: !51)
// CHECK:STDOUT: !57 = !{!52}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.EmptyRange.96a698c2240e6755.Main:Iterate.Core.894714a7cfcffc26", scope: null, file: !4, line: 18, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !69)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !7)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 2, scope: !58, type: !7)
// CHECK:STDOUT: !65 = !DILocation(line: 19, column: 14, scope: !58)
// CHECK:STDOUT: !66 = !{!7}
// CHECK:STDOUT: !67 = !DISubroutineType(types: !66)
// CHECK:STDOUT: !68 = !DILocation(line: 19, column: 7, scope: !58)
// CHECK:STDOUT: !69 = !{!59, !60}
// CHECK:STDOUT: !70 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.6a8897b1e38f8c5b", scope: null, file: !70, line: 36, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !79)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !7)
// CHECK:STDOUT: !77 = !DILocation(line: 37, column: 12, scope: !71)
// CHECK:STDOUT: !78 = !DILocation(line: 37, column: 5, scope: !71)
// CHECK:STDOUT: !79 = !{!72}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.6a8897b1e38f8c5b", scope: null, file: !70, line: 39, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !88)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !7)
// CHECK:STDOUT: !86 = !DILocation(line: 40, column: 12, scope: !80)
// CHECK:STDOUT: !87 = !DILocation(line: 40, column: 5, scope: !80)
// CHECK:STDOUT: !88 = !{!81}
// CHECK:STDOUT: !89 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e75049736efa12f6", scope: null, file: !89, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !96)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !7)
// CHECK:STDOUT: !94 = !DILocation(line: 36, column: 5, scope: !90)
// CHECK:STDOUT: !95 = !DILocation(line: 34, column: 1, scope: !90)
// CHECK:STDOUT: !96 = !{!91}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.46435643ebb2ee3e", scope: null, file: !89, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !103)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !7)
// CHECK:STDOUT: !102 = !DILocation(line: 34, column: 1, scope: !97)
// CHECK:STDOUT: !103 = !{!98}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7f8dce37f94a83bf", scope: null, file: !89, line: 34, type: !32, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !110)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !6)
// CHECK:STDOUT: !109 = !DILocation(line: 34, column: 1, scope: !104)
// CHECK:STDOUT: !110 = !{!105}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.82f9f5598387b9ac", scope: null, file: !89, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !117)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !7)
// CHECK:STDOUT: !116 = !DILocation(line: 34, column: 1, scope: !111)
// CHECK:STDOUT: !117 = !{!112}
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4405b7abc1074bea", scope: null, file: !89, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !124)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !7)
// CHECK:STDOUT: !123 = !DILocation(line: 34, column: 1, scope: !118)
// CHECK:STDOUT: !124 = !{!119}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.27eb7fea99346bf0", scope: null, file: !89, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !131)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !7)
// CHECK:STDOUT: !130 = !DILocation(line: 34, column: 1, scope: !125)
// CHECK:STDOUT: !131 = !{!126}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c28b24c1e5f4c659", scope: null, file: !89, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !138)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !7)
// CHECK:STDOUT: !137 = !DILocation(line: 34, column: 1, scope: !132)
// CHECK:STDOUT: !138 = !{!133}
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.6a8897b1e38f8c5b", scope: null, file: !70, line: 30, type: !67, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !142 = !DILocation(line: 31, column: 12, scope: !139)
// CHECK:STDOUT: !143 = !DILocation(line: 31, column: 5, scope: !139)
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.81d99cc9206a1cd8:OptionalStorage.Core.894714a7cfcffc26", scope: null, file: !70, line: 141, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !151)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !7)
// CHECK:STDOUT: !149 = !DILocation(line: 142, column: 12, scope: !144)
// CHECK:STDOUT: !150 = !DILocation(line: 142, column: 5, scope: !144)
// CHECK:STDOUT: !151 = !{!145}
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.81d99cc9206a1cd8:OptionalStorage.Core.894714a7cfcffc26", scope: null, file: !70, line: 144, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !159)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !152, type: !7)
// CHECK:STDOUT: !157 = !DILocation(line: 145, column: 12, scope: !152)
// CHECK:STDOUT: !158 = !DILocation(line: 145, column: 5, scope: !152)
// CHECK:STDOUT: !159 = !{!153}
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "None", linkageName: "_CNone.81d99cc9206a1cd8:OptionalStorage.Core.894714a7cfcffc26", scope: null, file: !70, line: 125, type: !67, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !167 = !DILocation(line: 128, column: 5, scope: !160)
// CHECK:STDOUT: !168 = !DILocation(line: 129, column: 5, scope: !160)
// CHECK:STDOUT: !169 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4e4edddc7580c79b:Copy.Core.166c3bb67ee40138", scope: null, file: !169, line: 62, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !179)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !7)
// CHECK:STDOUT: !175 = !DILocation(line: 63, column: 13, scope: !170)
// CHECK:STDOUT: !176 = !DILocation(line: 63, column: 12, scope: !170)
// CHECK:STDOUT: !177 = !DILocation(line: 63, column: 26, scope: !170)
// CHECK:STDOUT: !178 = !DILocation(line: 63, column: 5, scope: !170)
// CHECK:STDOUT: !179 = !{!171}
// CHECK:STDOUT:
