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

class EmptyRange(T:! Core.Copy & Core.Destroy) {
  impl as Core.Iterate where .CursorType = {} and .ElementType = T {
    fn NewCursor(unused self) -> {} {
      return {};
    }
    fn Next(unused self, _: {}*) -> Core.Optional(T) {
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

// CHECK:STDOUT: ; ModuleID = 'bindings.carbon'
// CHECK:STDOUT: source_filename = "bindings.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @EmptyRange.val.loc27_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(i32, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %r.var = alloca {}, align 1, !dbg !7
// CHECK:STDOUT:   %var = alloca {}, align 1, !dbg !8
// CHECK:STDOUT:   %.loc29_33.1.temp = alloca <{ { i32, i32 }, i1 }>, align 4, !dbg !8
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !9
// CHECK:STDOUT:   %.loc29_33.8.temp = alloca { i32, i32 }, align 4, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %r.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %r.var, ptr align 1 @EmptyRange.val.loc27_3, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   call void @"_CNewCursor.EmptyRange.a081b942542726db.Main:Iterate.Core.c9f733d9bf35d2e5"(ptr %r.var), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.EmptyRange.a081b942542726db.Main:Iterate.Core.c9f733d9bf35d2e5"(ptr %.loc29_33.1.temp, ptr %r.var, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.66b735e0a187b4e4(ptr %.loc29_33.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.8.temp), !dbg !8
// CHECK:STDOUT:   call void @_CGet.Optional.Core.66b735e0a187b4e4(ptr %.loc29_33.8.temp, ptr %.loc29_33.1.temp), !dbg !8
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc29_33.8.temp, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc29_33.8.temp, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   %.loc29_33.11 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !8
// CHECK:STDOUT:   %.loc29_33.12 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !8
// CHECK:STDOUT:   store i32 %.loc29_33.12, ptr %n.var, align 4, !dbg !9
// CHECK:STDOUT:   call void @_CF.Main(i32 %.loc29_33.11, ptr %n.var), !dbg !10
// CHECK:STDOUT:   br label %for.next, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @"_COp.eff30dc40f4a53d2:core.Destroy.Core"(ptr %.loc29_33.8.temp), !dbg !8
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %n.var), !dbg !9
// CHECK:STDOUT:   call void @"_COp.72fec7d6e0df4c57:core.Destroy.Core"(ptr %.loc29_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_COp.88c97397fb274b67:core.Destroy.Core"(ptr %r.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.eff30dc40f4a53d2:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.1155b488c23b5086:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5172fb5622df6926:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.938741932578d142:core.Destroy.Core"(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b935d6615b90a01b:core.Destroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.854750c92287bb79:core.Destroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.72fec7d6e0df4c57:core.Destroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.88c97397fb274b67:core.Destroy.Core"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNewCursor.EmptyRange.a081b942542726db.Main:Iterate.Core.c9f733d9bf35d2e5"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.EmptyRange.a081b942542726db.Main:Iterate.Core.c9f733d9bf35d2e5"(ptr sret(<{ { i32, i32 }, i1 }>) %return, ptr %self, ptr %_) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CNone.Optional.Core.66b735e0a187b4e4(ptr %return), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.66b735e0a187b4e4(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.c9f733d9bf35d2e5"(ptr %self), !dbg !69
// CHECK:STDOUT:   ret i1 %1, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGet.Optional.Core.66b735e0a187b4e4(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !71 {
// CHECK:STDOUT:   call void @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.c9f733d9bf35d2e5"(ptr %return, ptr %self), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.66b735e0a187b4e4(ptr sret(<{ { i32, i32 }, i1 }>) %return) #0 !dbg !76 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.c9f733d9bf35d2e5"(ptr %return), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.c9f733d9bf35d2e5"(ptr %value) #0 !dbg !81 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %value, i32 0, i32 1, !dbg !84
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !84
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !84
// CHECK:STDOUT:   ret i1 %2, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.c9f733d9bf35d2e5"(ptr sret({ i32, i32 }) %return, ptr %value) #0 !dbg !86 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %value, i32 0, i32 0, !dbg !89
// CHECK:STDOUT:   call void @"_COp.4e4edddc7580c79b:Copy.Core.58bedf1100cb91a8"(ptr %return, ptr %value1), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.c9f733d9bf35d2e5"(ptr sret(<{ { i32, i32 }, i1 }>) %return) #0 !dbg !91 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %return, i32 0, i32 1, !dbg !92
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !92
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.4e4edddc7580c79b:Copy.Core.58bedf1100cb91a8"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !94 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !98
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !98
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !99
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !100
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !100
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !99
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !99
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !99
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "bindings.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !1, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 27, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 29, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 29, column: 17, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 30, column: 5, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 29, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 26, column: 1, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !1, line: 27, type: !14, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !17)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{null, !16}
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !13, type: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 27, column: 10, scope: !13)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Op", linkageName: "_COp.eff30dc40f4a53d2:core.Destroy.Core", scope: null, file: !1, line: 27, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !23}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 27, column: 10, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5172fb5622df6926:core.Destroy.Core", scope: null, file: !1, line: 29, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 29, column: 7, scope: !27)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Op", linkageName: "_COp.938741932578d142:core.Destroy.Core", scope: null, file: !1, line: 29, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !32)
// CHECK:STDOUT: !32 = !{!33}
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !31, type: !23)
// CHECK:STDOUT: !34 = !DILocation(line: 29, column: 7, scope: !31)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b935d6615b90a01b:core.Destroy.Core", scope: null, file: !1, line: 29, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !36)
// CHECK:STDOUT: !36 = !{!37}
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !35, type: !23)
// CHECK:STDOUT: !38 = !DILocation(line: 29, column: 7, scope: !35)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp.854750c92287bb79:core.Destroy.Core", scope: null, file: !1, line: 29, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !40)
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !39, type: !23)
// CHECK:STDOUT: !42 = !DILocation(line: 29, column: 7, scope: !39)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Op", linkageName: "_COp.72fec7d6e0df4c57:core.Destroy.Core", scope: null, file: !1, line: 29, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !44)
// CHECK:STDOUT: !44 = !{!45}
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !43, type: !23)
// CHECK:STDOUT: !46 = !DILocation(line: 29, column: 7, scope: !43)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Op", linkageName: "_COp.88c97397fb274b67:core.Destroy.Core", scope: null, file: !1, line: 27, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !48)
// CHECK:STDOUT: !48 = !{!49}
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !47, type: !23)
// CHECK:STDOUT: !50 = !DILocation(line: 27, column: 3, scope: !47)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.EmptyRange.a081b942542726db.Main:Iterate.Core.c9f733d9bf35d2e5", scope: null, file: !1, line: 15, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !52)
// CHECK:STDOUT: !52 = !{!53}
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !51, type: !23)
// CHECK:STDOUT: !54 = !DILocation(line: 16, column: 7, scope: !51)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.EmptyRange.a081b942542726db.Main:Iterate.Core.c9f733d9bf35d2e5", scope: null, file: !1, line: 18, type: !56, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !57 = !{!23, !23, !23}
// CHECK:STDOUT: !58 = !{!59, !60}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !55, type: !23)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 2, scope: !55, type: !23)
// CHECK:STDOUT: !61 = !DILocation(line: 19, column: 14, scope: !55)
// CHECK:STDOUT: !62 = !DILocation(line: 19, column: 7, scope: !55)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.66b735e0a187b4e4", scope: null, file: !64, line: 36, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !67)
// CHECK:STDOUT: !64 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !65 = !DISubroutineType(types: !66)
// CHECK:STDOUT: !66 = !{!23, !23}
// CHECK:STDOUT: !67 = !{!68}
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !63, type: !23)
// CHECK:STDOUT: !69 = !DILocation(line: 37, column: 12, scope: !63)
// CHECK:STDOUT: !70 = !DILocation(line: 37, column: 5, scope: !63)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.66b735e0a187b4e4", scope: null, file: !64, line: 39, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !72)
// CHECK:STDOUT: !72 = !{!73}
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !71, type: !23)
// CHECK:STDOUT: !74 = !DILocation(line: 40, column: 12, scope: !71)
// CHECK:STDOUT: !75 = !DILocation(line: 40, column: 5, scope: !71)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.66b735e0a187b4e4", scope: null, file: !64, line: 30, type: !77, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !77 = !DISubroutineType(types: !78)
// CHECK:STDOUT: !78 = !{!23}
// CHECK:STDOUT: !79 = !DILocation(line: 31, column: 12, scope: !76)
// CHECK:STDOUT: !80 = !DILocation(line: 31, column: 5, scope: !76)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.c9f733d9bf35d2e5", scope: null, file: !64, line: 141, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !82)
// CHECK:STDOUT: !82 = !{!83}
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !81, type: !23)
// CHECK:STDOUT: !84 = !DILocation(line: 142, column: 12, scope: !81)
// CHECK:STDOUT: !85 = !DILocation(line: 142, column: 5, scope: !81)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.c9f733d9bf35d2e5", scope: null, file: !64, line: 144, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !87)
// CHECK:STDOUT: !87 = !{!88}
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !86, type: !23)
// CHECK:STDOUT: !89 = !DILocation(line: 145, column: 12, scope: !86)
// CHECK:STDOUT: !90 = !DILocation(line: 145, column: 5, scope: !86)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.c9f733d9bf35d2e5", scope: null, file: !64, line: 125, type: !77, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !92 = !DILocation(line: 128, column: 5, scope: !91)
// CHECK:STDOUT: !93 = !DILocation(line: 129, column: 5, scope: !91)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4e4edddc7580c79b:Copy.Core.58bedf1100cb91a8", scope: null, file: !95, line: 62, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !96)
// CHECK:STDOUT: !95 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !96 = !{!97}
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !94, type: !23)
// CHECK:STDOUT: !98 = !DILocation(line: 63, column: 13, scope: !94)
// CHECK:STDOUT: !99 = !DILocation(line: 63, column: 12, scope: !94)
// CHECK:STDOUT: !100 = !DILocation(line: 63, column: 26, scope: !94)
// CHECK:STDOUT: !101 = !DILocation(line: 63, column: 5, scope: !94)
