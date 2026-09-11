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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'bindings.carbon'
// CHECK:STDOUT: source_filename = "bindings.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @EmptyRange.val.loc27_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(i32, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %r.var = alloca {}, align 1, !dbg !7
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %var = alloca {}, align 1, !dbg !9
// CHECK:STDOUT:   %.loc29_33.1.temp = alloca <{ { i32, i32 }, i1 }>, align 4, !dbg !9
// CHECK:STDOUT:   %.loc29_33.8.temp = alloca { i32, i32 }, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %r.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %r.var, ptr align 1 @EmptyRange.val.loc27_3.1, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !8
// CHECK:STDOUT:   call void @"_CNewCursor.EmptyRange.a081b942542726db.Main:Iterate.Core.46b2ecf387c25deb"(ptr %r.var), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !9
// CHECK:STDOUT:   br label %for.next, !dbg !9
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.1.temp), !dbg !9
// CHECK:STDOUT:   call void @"_CNext.EmptyRange.a081b942542726db.Main:Iterate.Core.46b2ecf387c25deb"(ptr %.loc29_33.1.temp, ptr %r.var, ptr %var), !dbg !9
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.b2facdec5eee8049(ptr %.loc29_33.1.temp), !dbg !9
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !9
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.8.temp), !dbg !9
// CHECK:STDOUT:   call void @_CGet.Optional.Core.b2facdec5eee8049(ptr %.loc29_33.8.temp, ptr %.loc29_33.1.temp), !dbg !9
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc29_33.8.temp, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc29_33.8.temp, i32 0, i32 1, !dbg !9
// CHECK:STDOUT:   %.loc29_33.11 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !9
// CHECK:STDOUT:   %.loc29_33.12 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !9
// CHECK:STDOUT:   store i32 %.loc29_33.12, ptr %n.var, align 4, !dbg !8
// CHECK:STDOUT:   call void @_CF.Main(i32 %.loc29_33.11, ptr %n.var), !dbg !10
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c6bc553020e9306c(ptr %.loc29_33.8.temp), !dbg !9
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.72043207be93cf84(ptr %.loc29_33.1.temp), !dbg !9
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %n.var), !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.72043207be93cf84(ptr %.loc29_33.1.temp), !dbg !9
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %n.var), !dbg !8
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd(ptr %var), !dbg !9
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.73b928a00c3036a8(ptr %r.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7ebc50e8bd9eb2f1:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.6f4dee545ed23f91:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.927f3a4d5771ff29:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e7bd1d379ebc3076:core.Destroy.Core"(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ecf07f982ed29041:core.Destroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3ef490caafd5a665:core.Destroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.14c548d378df4c37:core.Destroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.cba888e825ff669c:core.Destroy.Core"(ptr %self) #0 !dbg !47 {
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
// CHECK:STDOUT: define linkonce_odr void @"_CNewCursor.EmptyRange.a081b942542726db.Main:Iterate.Core.46b2ecf387c25deb"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.EmptyRange.a081b942542726db.Main:Iterate.Core.46b2ecf387c25deb"(ptr sret(<{ { i32, i32 }, i1 }>) %return, ptr %self, ptr %_) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CNone.Optional.Core.b2facdec5eee8049(ptr %return), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.b2facdec5eee8049(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b2ecf387c25deb"(ptr %self), !dbg !69
// CHECK:STDOUT:   ret i1 %1, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGet.Optional.Core.b2facdec5eee8049(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !71 {
// CHECK:STDOUT:   call void @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b2ecf387c25deb"(ptr %return, ptr %self), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.c6bc553020e9306c(ptr %self) #0 !dbg !76 {
// CHECK:STDOUT:   call void @"_COp.7ebc50e8bd9eb2f1:core.Destroy.Core"(ptr %self), !dbg !80
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.c6bc553020e9306c(ptr %self), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.72043207be93cf84(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT:   call void @"_COp.14c548d378df4c37:core.Destroy.Core"(ptr %self), !dbg !86
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.c6bc553020e9306c(ptr %self), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT:   call void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self), !dbg !92
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.c6bc553020e9306c(ptr %self), !dbg !93
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.c6bc553020e9306c(ptr %self), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.73b928a00c3036a8(ptr %self) #0 !dbg !100 {
// CHECK:STDOUT:   call void @"_COp.cba888e825ff669c:core.Destroy.Core"(ptr %self), !dbg !103
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.c6bc553020e9306c(ptr %self), !dbg !104
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.b2facdec5eee8049(ptr sret(<{ { i32, i32 }, i1 }>) %return) #0 !dbg !106 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b2ecf387c25deb"(ptr %return), !dbg !109
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b2ecf387c25deb"(ptr %value) #0 !dbg !111 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %value, i32 0, i32 1, !dbg !114
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !114
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !114
// CHECK:STDOUT:   ret i1 %2, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b2ecf387c25deb"(ptr sret({ i32, i32 }) %return, ptr %value) #0 !dbg !116 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %value, i32 0, i32 0, !dbg !119
// CHECK:STDOUT:   call void @"_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb"(ptr %return, ptr %value1), !dbg !119
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSubobjectDestroy.Destroy.Core.c6bc553020e9306c(ptr %self) #0 !dbg !121 {
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b2ecf387c25deb"(ptr sret(<{ { i32, i32 }, i1 }>) %return) #0 !dbg !125 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %return, i32 0, i32 1, !dbg !126
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !128 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !132
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !132
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !133
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !134
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !134
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !133
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !133
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !133
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.72043207be93cf84, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSubobjectDestroy.Destroy.Core.c6bc553020e9306c, { 4, 3, 0, 1, 2 }
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
// CHECK:STDOUT: !8 = !DILocation(line: 29, column: 17, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 29, column: 7, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 30, column: 5, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 29, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 26, column: 1, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 27, type: !14, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !17)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{null, !16}
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !13, type: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 27, column: 10, scope: !13)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7ebc50e8bd9eb2f1:core.Destroy.Core", scope: null, file: !1, line: 27, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !23}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 27, column: 10, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.927f3a4d5771ff29:core.Destroy.Core", scope: null, file: !1, line: 29, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 29, column: 7, scope: !27)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e7bd1d379ebc3076:core.Destroy.Core", scope: null, file: !1, line: 29, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !32)
// CHECK:STDOUT: !32 = !{!33}
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !31, type: !23)
// CHECK:STDOUT: !34 = !DILocation(line: 29, column: 7, scope: !31)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ecf07f982ed29041:core.Destroy.Core", scope: null, file: !1, line: 29, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !36)
// CHECK:STDOUT: !36 = !{!37}
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !35, type: !23)
// CHECK:STDOUT: !38 = !DILocation(line: 29, column: 7, scope: !35)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3ef490caafd5a665:core.Destroy.Core", scope: null, file: !1, line: 29, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !40)
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !39, type: !23)
// CHECK:STDOUT: !42 = !DILocation(line: 29, column: 7, scope: !39)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Op", linkageName: "_COp.14c548d378df4c37:core.Destroy.Core", scope: null, file: !1, line: 29, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !44)
// CHECK:STDOUT: !44 = !{!45}
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !43, type: !23)
// CHECK:STDOUT: !46 = !DILocation(line: 29, column: 7, scope: !43)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cba888e825ff669c:core.Destroy.Core", scope: null, file: !1, line: 27, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !48)
// CHECK:STDOUT: !48 = !{!49}
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !47, type: !23)
// CHECK:STDOUT: !50 = !DILocation(line: 27, column: 3, scope: !47)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.EmptyRange.a081b942542726db.Main:Iterate.Core.46b2ecf387c25deb", scope: null, file: !1, line: 15, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !52)
// CHECK:STDOUT: !52 = !{!53}
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !51, type: !23)
// CHECK:STDOUT: !54 = !DILocation(line: 16, column: 7, scope: !51)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.EmptyRange.a081b942542726db.Main:Iterate.Core.46b2ecf387c25deb", scope: null, file: !1, line: 18, type: !56, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !57 = !{!23, !23, !23}
// CHECK:STDOUT: !58 = !{!59, !60}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !55, type: !23)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 2, scope: !55, type: !23)
// CHECK:STDOUT: !61 = !DILocation(line: 19, column: 14, scope: !55)
// CHECK:STDOUT: !62 = !DILocation(line: 19, column: 7, scope: !55)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.b2facdec5eee8049", scope: null, file: !64, line: 36, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !67)
// CHECK:STDOUT: !64 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !65 = !DISubroutineType(types: !66)
// CHECK:STDOUT: !66 = !{!23, !23}
// CHECK:STDOUT: !67 = !{!68}
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !63, type: !23)
// CHECK:STDOUT: !69 = !DILocation(line: 37, column: 12, scope: !63)
// CHECK:STDOUT: !70 = !DILocation(line: 37, column: 5, scope: !63)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.b2facdec5eee8049", scope: null, file: !64, line: 39, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !72)
// CHECK:STDOUT: !72 = !{!73}
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !71, type: !23)
// CHECK:STDOUT: !74 = !DILocation(line: 40, column: 12, scope: !71)
// CHECK:STDOUT: !75 = !DILocation(line: 40, column: 5, scope: !71)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.c6bc553020e9306c", scope: null, file: !77, line: 26, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !78)
// CHECK:STDOUT: !77 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !78 = !{!79}
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !76, type: !23)
// CHECK:STDOUT: !80 = !DILocation(line: 27, column: 5, scope: !76)
// CHECK:STDOUT: !81 = !DILocation(line: 28, column: 5, scope: !76)
// CHECK:STDOUT: !82 = !DILocation(line: 26, column: 3, scope: !76)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.72043207be93cf84", scope: null, file: !77, line: 26, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !84)
// CHECK:STDOUT: !84 = !{!85}
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !83, type: !23)
// CHECK:STDOUT: !86 = !DILocation(line: 27, column: 5, scope: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 28, column: 5, scope: !83)
// CHECK:STDOUT: !88 = !DILocation(line: 26, column: 3, scope: !83)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !77, line: 26, type: !14, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !90)
// CHECK:STDOUT: !90 = !{!91}
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !89, type: !16)
// CHECK:STDOUT: !92 = !DILocation(line: 27, column: 5, scope: !89)
// CHECK:STDOUT: !93 = !DILocation(line: 28, column: 5, scope: !89)
// CHECK:STDOUT: !94 = !DILocation(line: 26, column: 3, scope: !89)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd", scope: null, file: !77, line: 26, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !96)
// CHECK:STDOUT: !96 = !{!97}
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !95, type: !23)
// CHECK:STDOUT: !98 = !DILocation(line: 28, column: 5, scope: !95)
// CHECK:STDOUT: !99 = !DILocation(line: 26, column: 3, scope: !95)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.73b928a00c3036a8", scope: null, file: !77, line: 26, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !101)
// CHECK:STDOUT: !101 = !{!102}
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !100, type: !23)
// CHECK:STDOUT: !103 = !DILocation(line: 27, column: 5, scope: !100)
// CHECK:STDOUT: !104 = !DILocation(line: 28, column: 5, scope: !100)
// CHECK:STDOUT: !105 = !DILocation(line: 26, column: 3, scope: !100)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.b2facdec5eee8049", scope: null, file: !64, line: 30, type: !107, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !107 = !DISubroutineType(types: !108)
// CHECK:STDOUT: !108 = !{!23}
// CHECK:STDOUT: !109 = !DILocation(line: 31, column: 12, scope: !106)
// CHECK:STDOUT: !110 = !DILocation(line: 31, column: 5, scope: !106)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b2ecf387c25deb", scope: null, file: !64, line: 141, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !112)
// CHECK:STDOUT: !112 = !{!113}
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !111, type: !23)
// CHECK:STDOUT: !114 = !DILocation(line: 142, column: 12, scope: !111)
// CHECK:STDOUT: !115 = !DILocation(line: 142, column: 5, scope: !111)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b2ecf387c25deb", scope: null, file: !64, line: 144, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !117)
// CHECK:STDOUT: !117 = !{!118}
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !116, type: !23)
// CHECK:STDOUT: !119 = !DILocation(line: 145, column: 12, scope: !116)
// CHECK:STDOUT: !120 = !DILocation(line: 145, column: 5, scope: !116)
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.Destroy.Core.c6bc553020e9306c", scope: null, file: !77, line: 19, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !122)
// CHECK:STDOUT: !122 = !{!123}
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !121, type: !23)
// CHECK:STDOUT: !124 = !DILocation(line: 19, column: 3, scope: !121)
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b2ecf387c25deb", scope: null, file: !64, line: 125, type: !107, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !126 = !DILocation(line: 128, column: 5, scope: !125)
// CHECK:STDOUT: !127 = !DILocation(line: 129, column: 5, scope: !125)
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb", scope: null, file: !129, line: 62, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !130)
// CHECK:STDOUT: !129 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !130 = !{!131}
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !128, type: !23)
// CHECK:STDOUT: !132 = !DILocation(line: 63, column: 13, scope: !128)
// CHECK:STDOUT: !133 = !DILocation(line: 63, column: 12, scope: !128)
// CHECK:STDOUT: !134 = !DILocation(line: 63, column: 26, scope: !128)
// CHECK:STDOUT: !135 = !DILocation(line: 63, column: 5, scope: !128)
// CHECK:STDOUT:
