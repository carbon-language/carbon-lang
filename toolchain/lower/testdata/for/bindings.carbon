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
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %r.var = alloca {}, align 1, !dbg !17
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !19
// CHECK:STDOUT:   %var = alloca {}, align 1, !dbg !22
// CHECK:STDOUT:   %.loc29_33.1.temp = alloca <{ { i32, i32 }, i1 }>, align 4, !dbg !22
// CHECK:STDOUT:   %.loc29_33.8.temp = alloca { i32, i32 }, align 4, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %r.var), !dbg !17
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %r.var, ptr align 1 @EmptyRange.val.loc27_3.1, i64 0, i1 false), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !19
// CHECK:STDOUT:   call void @"_CNewCursor.EmptyRange.a081b942542726db.Main:Iterate.Core.9d362ec27e917256"(ptr %r.var), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !22
// CHECK:STDOUT:   br label %for.next, !dbg !22
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.1.temp), !dbg !22
// CHECK:STDOUT:   call void @"_CNext.EmptyRange.a081b942542726db.Main:Iterate.Core.9d362ec27e917256"(ptr %.loc29_33.1.temp, ptr %r.var, ptr %var), !dbg !22
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.4f44af021e5d7006(ptr %.loc29_33.1.temp), !dbg !22
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !22
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.8.temp), !dbg !22
// CHECK:STDOUT:   call void @_CGet.Optional.Core.4f44af021e5d7006(ptr %.loc29_33.8.temp, ptr %.loc29_33.1.temp), !dbg !22
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc29_33.8.temp, i32 0, i32 0, !dbg !22
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc29_33.8.temp, i32 0, i32 1, !dbg !22
// CHECK:STDOUT:   %.loc29_33.11 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !22
// CHECK:STDOUT:   %.loc29_33.12 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !22
// CHECK:STDOUT:   store i32 %.loc29_33.12, ptr %n.var, align 4, !dbg !19
// CHECK:STDOUT:   call void @_CF.Main(i32 %.loc29_33.11, ptr %n.var), !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c6bc553020e9306c(ptr %.loc29_33.8.temp), !dbg !22
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.43ab251ed50630f4(ptr %.loc29_33.1.temp), !dbg !22
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %n.var), !dbg !19
// CHECK:STDOUT:   br label %for.next, !dbg !33
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.43ab251ed50630f4(ptr %.loc29_33.1.temp), !dbg !22
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %n.var), !dbg !19
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd(ptr %var), !dbg !22
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c20f9316c913ca39(ptr %r.var), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.6b9c90d9dd82cebb:core.Destroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.63f8452b0e65db7c:core.Destroy.Core"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f1878342e4d848d0:core.Destroy.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a73f0899c6bb0b51:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a35747328ec5c4bc:core.Destroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.5d8a292fae653ad6:core.Destroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNewCursor.EmptyRange.a081b942542726db.Main:Iterate.Core.9d362ec27e917256"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.EmptyRange.a081b942542726db.Main:Iterate.Core.9d362ec27e917256"(ptr sret(<{ { i32, i32 }, i1 }>) %return, ptr %self, ptr %_) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CNone.Optional.Core.4f44af021e5d7006(ptr %return), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.4f44af021e5d7006(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.9d362ec27e917256"(ptr %self), !dbg !93
// CHECK:STDOUT:   ret i1 %1, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGet.Optional.Core.4f44af021e5d7006(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !96 {
// CHECK:STDOUT:   call void @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.9d362ec27e917256"(ptr %return, ptr %self), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.c6bc553020e9306c(ptr %self) #0 !dbg !106 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core"(ptr %self), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.43ab251ed50630f4(ptr %self) #0 !dbg !113 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.a35747328ec5c4bc:core.Destroy.Core"(ptr %self), !dbg !117
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !120 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !124
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd(ptr %self) #0 !dbg !127 {
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.c20f9316c913ca39(ptr %self) #0 !dbg !134 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.5d8a292fae653ad6:core.Destroy.Core"(ptr %self), !dbg !138
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.4f44af021e5d7006(ptr sret(<{ { i32, i32 }, i1 }>) %return) #0 !dbg !141 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.9d362ec27e917256"(ptr %return), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.9d362ec27e917256"(ptr %value) #0 !dbg !146 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %value, i32 0, i32 1, !dbg !151
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !151
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !151
// CHECK:STDOUT:   ret i1 %2, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.9d362ec27e917256"(ptr sret({ i32, i32 }) %return, ptr %value) #0 !dbg !154 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %value, i32 0, i32 0, !dbg !159
// CHECK:STDOUT:   call void @"_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb"(ptr %return, ptr %value1), !dbg !159
// CHECK:STDOUT:   ret void, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.9d362ec27e917256"(ptr sret(<{ { i32, i32 }, i1 }>) %return) #0 !dbg !162 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %return, i32 0, i32 1, !dbg !169
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !169
// CHECK:STDOUT:   ret void, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !172 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !177
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !177
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !178
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !179
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !179
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !178
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !178
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !178
// CHECK:STDOUT:   ret void, !dbg !180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.43ab251ed50630f4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48, { 1, 0 }
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
// CHECK:STDOUT: !12 = !{null, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{null, !7}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !4, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !17 = !DILocation(line: 27, column: 3, scope: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 29, column: 17, scope: !16)
// CHECK:STDOUT: !22 = !DILocation(line: 29, column: 7, scope: !16)
// CHECK:STDOUT: !23 = !{!7, !7, !7}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !{!7, !7}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 30, column: 5, scope: !16)
// CHECK:STDOUT: !33 = !DILocation(line: 29, column: 3, scope: !16)
// CHECK:STDOUT: !34 = !DILocation(line: 26, column: 1, scope: !16)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 27, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !6)
// CHECK:STDOUT: !37 = !DILocation(line: 27, column: 10, scope: !35)
// CHECK:STDOUT: !38 = !{!36}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core", scope: null, file: !4, line: 27, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !7)
// CHECK:STDOUT: !41 = !DILocation(line: 27, column: 10, scope: !39)
// CHECK:STDOUT: !42 = !{!40}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.6b9c90d9dd82cebb:core.Destroy.Core", scope: null, file: !4, line: 29, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !7)
// CHECK:STDOUT: !45 = !DILocation(line: 29, column: 7, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.63f8452b0e65db7c:core.Destroy.Core", scope: null, file: !4, line: 29, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !7)
// CHECK:STDOUT: !49 = !DILocation(line: 29, column: 7, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f1878342e4d848d0:core.Destroy.Core", scope: null, file: !4, line: 29, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !54)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !7)
// CHECK:STDOUT: !53 = !DILocation(line: 29, column: 7, scope: !51)
// CHECK:STDOUT: !54 = !{!52}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a73f0899c6bb0b51:core.Destroy.Core", scope: null, file: !4, line: 29, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !7)
// CHECK:STDOUT: !57 = !DILocation(line: 29, column: 7, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a35747328ec5c4bc:core.Destroy.Core", scope: null, file: !4, line: 29, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !7)
// CHECK:STDOUT: !61 = !DILocation(line: 29, column: 7, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.5d8a292fae653ad6:core.Destroy.Core", scope: null, file: !4, line: 27, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !7)
// CHECK:STDOUT: !65 = !DILocation(line: 27, column: 3, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.EmptyRange.a081b942542726db.Main:Iterate.Core.9d362ec27e917256", scope: null, file: !4, line: 15, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !73)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !7)
// CHECK:STDOUT: !72 = !DILocation(line: 16, column: 7, scope: !67)
// CHECK:STDOUT: !73 = !{!68}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.EmptyRange.a081b942542726db.Main:Iterate.Core.9d362ec27e917256", scope: null, file: !4, line: 18, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !85)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !7)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 2, scope: !74, type: !7)
// CHECK:STDOUT: !81 = !DILocation(line: 19, column: 14, scope: !74)
// CHECK:STDOUT: !82 = !{!7}
// CHECK:STDOUT: !83 = !DISubroutineType(types: !82)
// CHECK:STDOUT: !84 = !DILocation(line: 19, column: 7, scope: !74)
// CHECK:STDOUT: !85 = !{!75, !76}
// CHECK:STDOUT: !86 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.4f44af021e5d7006", scope: null, file: !86, line: 36, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !95)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !7)
// CHECK:STDOUT: !93 = !DILocation(line: 37, column: 12, scope: !87)
// CHECK:STDOUT: !94 = !DILocation(line: 37, column: 5, scope: !87)
// CHECK:STDOUT: !95 = !{!88}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.4f44af021e5d7006", scope: null, file: !86, line: 39, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !104)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !7)
// CHECK:STDOUT: !102 = !DILocation(line: 40, column: 12, scope: !96)
// CHECK:STDOUT: !103 = !DILocation(line: 40, column: 5, scope: !96)
// CHECK:STDOUT: !104 = !{!97}
// CHECK:STDOUT: !105 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.c6bc553020e9306c", scope: null, file: !105, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !112)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !7)
// CHECK:STDOUT: !110 = !DILocation(line: 28, column: 5, scope: !106)
// CHECK:STDOUT: !111 = !DILocation(line: 26, column: 3, scope: !106)
// CHECK:STDOUT: !112 = !{!107}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.43ab251ed50630f4", scope: null, file: !105, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !119)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !7)
// CHECK:STDOUT: !117 = !DILocation(line: 28, column: 5, scope: !113)
// CHECK:STDOUT: !118 = !DILocation(line: 26, column: 3, scope: !113)
// CHECK:STDOUT: !119 = !{!114}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !105, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !126)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !6)
// CHECK:STDOUT: !124 = !DILocation(line: 28, column: 5, scope: !120)
// CHECK:STDOUT: !125 = !DILocation(line: 26, column: 3, scope: !120)
// CHECK:STDOUT: !126 = !{!121}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd", scope: null, file: !105, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !133)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !7)
// CHECK:STDOUT: !132 = !DILocation(line: 26, column: 3, scope: !127)
// CHECK:STDOUT: !133 = !{!128}
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.c20f9316c913ca39", scope: null, file: !105, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !140)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !7)
// CHECK:STDOUT: !138 = !DILocation(line: 28, column: 5, scope: !134)
// CHECK:STDOUT: !139 = !DILocation(line: 26, column: 3, scope: !134)
// CHECK:STDOUT: !140 = !{!135}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.4f44af021e5d7006", scope: null, file: !86, line: 30, type: !83, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !144 = !DILocation(line: 31, column: 12, scope: !141)
// CHECK:STDOUT: !145 = !DILocation(line: 31, column: 5, scope: !141)
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.9d362ec27e917256", scope: null, file: !86, line: 141, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !153)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !7)
// CHECK:STDOUT: !151 = !DILocation(line: 142, column: 12, scope: !146)
// CHECK:STDOUT: !152 = !DILocation(line: 142, column: 5, scope: !146)
// CHECK:STDOUT: !153 = !{!147}
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.9d362ec27e917256", scope: null, file: !86, line: 144, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !161)
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 1, scope: !154, type: !7)
// CHECK:STDOUT: !159 = !DILocation(line: 145, column: 12, scope: !154)
// CHECK:STDOUT: !160 = !DILocation(line: 145, column: 5, scope: !154)
// CHECK:STDOUT: !161 = !{!155}
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.9d362ec27e917256", scope: null, file: !86, line: 125, type: !83, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !169 = !DILocation(line: 128, column: 5, scope: !162)
// CHECK:STDOUT: !170 = !DILocation(line: 129, column: 5, scope: !162)
// CHECK:STDOUT: !171 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb", scope: null, file: !171, line: 62, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !181)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 1, scope: !172, type: !7)
// CHECK:STDOUT: !177 = !DILocation(line: 63, column: 13, scope: !172)
// CHECK:STDOUT: !178 = !DILocation(line: 63, column: 12, scope: !172)
// CHECK:STDOUT: !179 = !DILocation(line: 63, column: 26, scope: !172)
// CHECK:STDOUT: !180 = !DILocation(line: 63, column: 5, scope: !172)
// CHECK:STDOUT: !181 = !{!173}
// CHECK:STDOUT:
