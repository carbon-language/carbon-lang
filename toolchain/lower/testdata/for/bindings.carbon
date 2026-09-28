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
// CHECK:STDOUT: @EmptyRange.val.loc27_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(i32, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %r.var = alloca {}, align 1, !dbg !21
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !23
// CHECK:STDOUT:   %var = alloca {}, align 1, !dbg !26
// CHECK:STDOUT:   %.loc29_33.1.temp = alloca <{ { i32, i32 }, i1 }>, align 4, !dbg !26
// CHECK:STDOUT:   %.loc29_33.8.temp = alloca { i32, i32 }, align 4, !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %r.var), !dbg !21
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %r.var, ptr align 1 @EmptyRange.val.loc27_3, i64 0, i1 false), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !23
// CHECK:STDOUT:   call void @"_CNewCursor.EmptyRange.a081b942542726db.Main:Iterate.Core.2e30d2d7a87a6cd3"(ptr %r.var), !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !26
// CHECK:STDOUT:   br label %for.next, !dbg !26
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.1.temp), !dbg !26
// CHECK:STDOUT:   call void @"_CNext.EmptyRange.a081b942542726db.Main:Iterate.Core.2e30d2d7a87a6cd3"(ptr %.loc29_33.1.temp, ptr %r.var, ptr %var), !dbg !26
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.b20cf0d1c9daa8c2(ptr %.loc29_33.1.temp), !dbg !26
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !26
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.8.temp), !dbg !26
// CHECK:STDOUT:   call void @_CGet.Optional.Core.b20cf0d1c9daa8c2(ptr %.loc29_33.8.temp, ptr %.loc29_33.1.temp), !dbg !26
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc29_33.8.temp, i32 0, i32 0, !dbg !26
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc29_33.8.temp, i32 0, i32 1, !dbg !26
// CHECK:STDOUT:   %.loc29_33.11 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !26
// CHECK:STDOUT:   %.loc29_33.12 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !26
// CHECK:STDOUT:   store i32 %.loc29_33.12, ptr %n.var, align 4, !dbg !23
// CHECK:STDOUT:   call void @_CF.Main(i32 %.loc29_33.11, ptr %n.var), !dbg !36
// CHECK:STDOUT:   call void @"_CSelfDestruct.e554695bf635d9c1:core.Destroy.Core"(ptr %.loc29_33.8.temp), !dbg !26
// CHECK:STDOUT:   call void @"_CSelfDestruct.e1dbefa6145042b5:core.Destroy.Core"(ptr %.loc29_33.1.temp), !dbg !26
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %n.var), !dbg !23
// CHECK:STDOUT:   br label %for.next, !dbg !37
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @"_CSelfDestruct.e1dbefa6145042b5:core.Destroy.Core"(ptr %.loc29_33.1.temp), !dbg !26
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %n.var), !dbg !23
// CHECK:STDOUT:   call void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %var), !dbg !26
// CHECK:STDOUT:   call void @"_CSelfDestruct.454683f3b8639b20:core.Destroy.Core"(ptr %r.var), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !45
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %tuple.elem1.tuple.elem), !dbg !45
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !45
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %tuple.elem0.tuple.elem), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.e554695bf635d9c1:core.Destroy.Core"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core"(ptr %self), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.6b9c90d9dd82cebb:core.Destroy.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6b9c90d9dd82cebb:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.6b9c90d9dd82cebb:core.Destroy.Core"(ptr %self), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.a69fcb7116d040b0:core.Destroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.47c342be31d1679f:core.Destroy.Core"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.47c342be31d1679f:core.Destroy.Core"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.47c342be31d1679f:core.Destroy.Core"(ptr %self), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b61f4dca88958e98:core.Destroy.Core"(ptr %self) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b61f4dca88958e98:core.Destroy.Core"(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b61f4dca88958e98:core.Destroy.Core"(ptr %self), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.e1dbefa6145042b5:core.Destroy.Core"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454683f3b8639b20:core.Destroy.Core"(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNewCursor.EmptyRange.a081b942542726db.Main:Iterate.Core.2e30d2d7a87a6cd3"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.EmptyRange.a081b942542726db.Main:Iterate.Core.2e30d2d7a87a6cd3"(ptr sret(<{ { i32, i32 }, i1 }>) %return, ptr %self, ptr %_) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CNone.Optional.Core.b20cf0d1c9daa8c2(ptr %return), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.b20cf0d1c9daa8c2(ptr %self) #0 !dbg !111 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.2e30d2d7a87a6cd3"(ptr %self), !dbg !117
// CHECK:STDOUT:   ret i1 %1, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGet.Optional.Core.b20cf0d1c9daa8c2(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !120 {
// CHECK:STDOUT:   call void @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.2e30d2d7a87a6cd3"(ptr %return, ptr %self), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.b20cf0d1c9daa8c2(ptr sret(<{ { i32, i32 }, i1 }>) %return) #0 !dbg !129 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.2e30d2d7a87a6cd3"(ptr %return), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.2e30d2d7a87a6cd3"(ptr %value) #0 !dbg !134 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %value, i32 0, i32 1, !dbg !139
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !139
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !139
// CHECK:STDOUT:   ret i1 %2, !dbg !140
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.2e30d2d7a87a6cd3"(ptr sret({ i32, i32 }) %return, ptr %value) #0 !dbg !142 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %value, i32 0, i32 0, !dbg !147
// CHECK:STDOUT:   call void @"_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb"(ptr %return, ptr %value1), !dbg !147
// CHECK:STDOUT:   ret void, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.2e30d2d7a87a6cd3"(ptr sret(<{ { i32, i32 }, i1 }>) %return) #0 !dbg !150 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %return, i32 0, i32 1, !dbg !157
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !157
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !160 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !165
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !165
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !166
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !167
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !167
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !166
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !166
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !166
// CHECK:STDOUT:   ret void, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core", { 0, 1, 3, 2 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.e1dbefa6145042b5:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
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
// CHECK:STDOUT: !8 = !{null, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !12 = !{null}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{null, !6}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !4, line: 14, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !19)
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !16, type: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 14, column: 44, scope: !16)
// CHECK:STDOUT: !19 = !{!17}
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !4, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !21 = !DILocation(line: 27, column: 3, scope: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 29, column: 17, scope: !20)
// CHECK:STDOUT: !26 = !DILocation(line: 29, column: 7, scope: !20)
// CHECK:STDOUT: !27 = !{!7, !7, !7}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !29 = !{!7, !7}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !36 = !DILocation(line: 30, column: 5, scope: !20)
// CHECK:STDOUT: !37 = !DILocation(line: 29, column: 3, scope: !20)
// CHECK:STDOUT: !38 = !DILocation(line: 26, column: 1, scope: !20)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 27, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !6)
// CHECK:STDOUT: !41 = !DILocation(line: 27, column: 10, scope: !39)
// CHECK:STDOUT: !42 = !{!40}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core", scope: null, file: !4, line: 27, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !7)
// CHECK:STDOUT: !45 = !DILocation(line: 27, column: 10, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.e554695bf635d9c1:core.Destroy.Core", scope: null, file: !4, line: 27, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !7)
// CHECK:STDOUT: !49 = !DILocation(line: 27, column: 10, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.6b9c90d9dd82cebb:core.Destroy.Core", scope: null, file: !4, line: 29, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !54)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !7)
// CHECK:STDOUT: !53 = !DILocation(line: 29, column: 7, scope: !51)
// CHECK:STDOUT: !54 = !{!52}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6b9c90d9dd82cebb:core.Destroy.Core", scope: null, file: !4, line: 29, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !7)
// CHECK:STDOUT: !57 = !DILocation(line: 29, column: 7, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.a69fcb7116d040b0:core.Destroy.Core", scope: null, file: !4, line: 29, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !7)
// CHECK:STDOUT: !61 = !DILocation(line: 29, column: 7, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core", scope: null, file: !4, line: 29, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !7)
// CHECK:STDOUT: !65 = !DILocation(line: 29, column: 7, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.47c342be31d1679f:core.Destroy.Core", scope: null, file: !4, line: 29, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !7)
// CHECK:STDOUT: !69 = !DILocation(line: 29, column: 7, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.47c342be31d1679f:core.Destroy.Core", scope: null, file: !4, line: 29, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !7)
// CHECK:STDOUT: !73 = !DILocation(line: 29, column: 7, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b61f4dca88958e98:core.Destroy.Core", scope: null, file: !4, line: 29, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !78)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !7)
// CHECK:STDOUT: !77 = !DILocation(line: 29, column: 7, scope: !75)
// CHECK:STDOUT: !78 = !{!76}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b61f4dca88958e98:core.Destroy.Core", scope: null, file: !4, line: 29, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !82)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !7)
// CHECK:STDOUT: !81 = !DILocation(line: 29, column: 7, scope: !79)
// CHECK:STDOUT: !82 = !{!80}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.e1dbefa6145042b5:core.Destroy.Core", scope: null, file: !4, line: 29, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !86)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !7)
// CHECK:STDOUT: !85 = !DILocation(line: 29, column: 7, scope: !83)
// CHECK:STDOUT: !86 = !{!84}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454683f3b8639b20:core.Destroy.Core", scope: null, file: !4, line: 27, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !7)
// CHECK:STDOUT: !89 = !DILocation(line: 27, column: 3, scope: !87)
// CHECK:STDOUT: !90 = !{!88}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.EmptyRange.a081b942542726db.Main:Iterate.Core.2e30d2d7a87a6cd3", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !97)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !7)
// CHECK:STDOUT: !96 = !DILocation(line: 16, column: 7, scope: !91)
// CHECK:STDOUT: !97 = !{!92}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.EmptyRange.a081b942542726db.Main:Iterate.Core.2e30d2d7a87a6cd3", scope: null, file: !4, line: 18, type: !28, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !109)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !7)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 2, scope: !98, type: !7)
// CHECK:STDOUT: !105 = !DILocation(line: 19, column: 14, scope: !98)
// CHECK:STDOUT: !106 = !{!7}
// CHECK:STDOUT: !107 = !DISubroutineType(types: !106)
// CHECK:STDOUT: !108 = !DILocation(line: 19, column: 7, scope: !98)
// CHECK:STDOUT: !109 = !{!99, !100}
// CHECK:STDOUT: !110 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.b20cf0d1c9daa8c2", scope: null, file: !110, line: 36, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !119)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !7)
// CHECK:STDOUT: !117 = !DILocation(line: 37, column: 12, scope: !111)
// CHECK:STDOUT: !118 = !DILocation(line: 37, column: 5, scope: !111)
// CHECK:STDOUT: !119 = !{!112}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.b20cf0d1c9daa8c2", scope: null, file: !110, line: 39, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !128)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !7)
// CHECK:STDOUT: !126 = !DILocation(line: 40, column: 12, scope: !120)
// CHECK:STDOUT: !127 = !DILocation(line: 40, column: 5, scope: !120)
// CHECK:STDOUT: !128 = !{!121}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.b20cf0d1c9daa8c2", scope: null, file: !110, line: 30, type: !107, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !132 = !DILocation(line: 31, column: 12, scope: !129)
// CHECK:STDOUT: !133 = !DILocation(line: 31, column: 5, scope: !129)
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.2e30d2d7a87a6cd3", scope: null, file: !110, line: 141, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !141)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !7)
// CHECK:STDOUT: !139 = !DILocation(line: 142, column: 12, scope: !134)
// CHECK:STDOUT: !140 = !DILocation(line: 142, column: 5, scope: !134)
// CHECK:STDOUT: !141 = !{!135}
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.2e30d2d7a87a6cd3", scope: null, file: !110, line: 144, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !149)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 1, scope: !142, type: !7)
// CHECK:STDOUT: !147 = !DILocation(line: 145, column: 12, scope: !142)
// CHECK:STDOUT: !148 = !DILocation(line: 145, column: 5, scope: !142)
// CHECK:STDOUT: !149 = !{!143}
// CHECK:STDOUT: !150 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.2e30d2d7a87a6cd3", scope: null, file: !110, line: 125, type: !107, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !157 = !DILocation(line: 128, column: 5, scope: !150)
// CHECK:STDOUT: !158 = !DILocation(line: 129, column: 5, scope: !150)
// CHECK:STDOUT: !159 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb", scope: null, file: !159, line: 62, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !169)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !7)
// CHECK:STDOUT: !165 = !DILocation(line: 63, column: 13, scope: !160)
// CHECK:STDOUT: !166 = !DILocation(line: 63, column: 12, scope: !160)
// CHECK:STDOUT: !167 = !DILocation(line: 63, column: 26, scope: !160)
// CHECK:STDOUT: !168 = !DILocation(line: 63, column: 5, scope: !160)
// CHECK:STDOUT: !169 = !{!161}
// CHECK:STDOUT:
