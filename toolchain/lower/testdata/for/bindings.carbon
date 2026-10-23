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
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %r.var = alloca {}, align 1, !dbg !13
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !15
// CHECK:STDOUT:   %var = alloca {}, align 1, !dbg !18
// CHECK:STDOUT:   %.loc29_33.1.temp = alloca <{ { i32, i32 }, i1 }>, align 4, !dbg !18
// CHECK:STDOUT:   %.loc29_33.8.temp = alloca { i32, i32 }, align 4, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %r.var), !dbg !13
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %r.var, ptr align 1 @EmptyRange.val.loc27_3.1, i64 0, i1 false), !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !15
// CHECK:STDOUT:   call void @"_CNewCursor.EmptyRange.b30481ddd22ae53e.Main:Iterate.Core.3c6fb8ac8f4bd886"(ptr %r.var), !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !18
// CHECK:STDOUT:   br label %for.next, !dbg !18
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.1.temp), !dbg !18
// CHECK:STDOUT:   call void @"_CNext.EmptyRange.b30481ddd22ae53e.Main:Iterate.Core.3c6fb8ac8f4bd886"(ptr %.loc29_33.1.temp, ptr %r.var, ptr %var), !dbg !18
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.dc5ffaddab9fc53e(ptr %.loc29_33.1.temp), !dbg !18
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !18
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.8.temp), !dbg !18
// CHECK:STDOUT:   call void @_CGet.Optional.Core.dc5ffaddab9fc53e(ptr %.loc29_33.8.temp, ptr %.loc29_33.1.temp), !dbg !18
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc29_33.8.temp, i32 0, i32 0, !dbg !18
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc29_33.8.temp, i32 0, i32 1, !dbg !18
// CHECK:STDOUT:   %.loc29_33.11 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !18
// CHECK:STDOUT:   %.loc29_33.12 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !18
// CHECK:STDOUT:   store i32 %.loc29_33.12, ptr %n.var, align 4, !dbg !15
// CHECK:STDOUT:   call void @_CF.Main(i32 %.loc29_33.11, ptr %n.var), !dbg !30
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ffbc82e0019b7bf2(ptr %.loc29_33.8.temp), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.decc97f51bbf9647(ptr %.loc29_33.1.temp), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %n.var), !dbg !15
// CHECK:STDOUT:   br label %for.next, !dbg !33
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.decc97f51bbf9647(ptr %.loc29_33.1.temp), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %n.var), !dbg !15
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.61b5bb2079973306(ptr %var), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.feee70e0690b5123(ptr %r.var), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNewCursor.EmptyRange.b30481ddd22ae53e.Main:Iterate.Core.3c6fb8ac8f4bd886"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.EmptyRange.b30481ddd22ae53e.Main:Iterate.Core.3c6fb8ac8f4bd886"(ptr sret(<{ { i32, i32 }, i1 }>) %return, ptr %self, ptr %_) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CNone.Optional.Core.dc5ffaddab9fc53e(ptr %return), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.dc5ffaddab9fc53e(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.3c6fb8ac8f4bd886"(ptr %self), !dbg !61
// CHECK:STDOUT:   ret i1 %1, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGet.Optional.Core.dc5ffaddab9fc53e(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !64 {
// CHECK:STDOUT:   call void @"_CGet.c1cc3e41344c17ec:OptionalStorage.Core.3c6fb8ac8f4bd886"(ptr %return, ptr %self), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ffbc82e0019b7bf2(ptr %t) #0 !dbg !74 {
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.decc97f51bbf9647(ptr %t) #0 !dbg !81 {
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !88 {
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.61b5bb2079973306(ptr %t) #0 !dbg !95 {
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.feee70e0690b5123(ptr %t) #0 !dbg !102 {
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.dc5ffaddab9fc53e(ptr sret(<{ { i32, i32 }, i1 }>) %return) #0 !dbg !109 {
// CHECK:STDOUT:   call void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.3c6fb8ac8f4bd886"(ptr %return), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.3c6fb8ac8f4bd886"(ptr %value) #0 !dbg !114 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %value, i32 0, i32 1, !dbg !119
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !119
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !119
// CHECK:STDOUT:   ret i1 %2, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CGet.c1cc3e41344c17ec:OptionalStorage.Core.3c6fb8ac8f4bd886"(ptr sret({ i32, i32 }) %return, ptr %value) #0 !dbg !122 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %value, i32 0, i32 0, !dbg !127
// CHECK:STDOUT:   call void @"_COp.4e4edddc7580c79b:Copy.Core.3eda18978e8f48e1"(ptr %return, ptr %value1), !dbg !127
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.3c6fb8ac8f4bd886"(ptr sret(<{ { i32, i32 }, i1 }>) %return) #0 !dbg !130 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ { i32, i32 }, i1 }>, ptr %return, i32 0, i32 1, !dbg !137
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !137
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.4e4edddc7580c79b:Copy.Core.3eda18978e8f48e1"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !140 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !145
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !145
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !146
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !147
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !147
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !146
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !146
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !146
// CHECK:STDOUT:   ret void, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.decc97f51bbf9647, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 1, 0 }
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
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !4, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !13 = !DILocation(line: 27, column: 3, scope: !12)
// CHECK:STDOUT: !15 = !DILocation(line: 29, column: 17, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 29, column: 7, scope: !12)
// CHECK:STDOUT: !19 = !{null, !7}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !{!7, !7, !7}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{!7, !7}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 30, column: 5, scope: !12)
// CHECK:STDOUT: !31 = !{null, !6}
// CHECK:STDOUT: !32 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !33 = !DILocation(line: 29, column: 3, scope: !12)
// CHECK:STDOUT: !34 = !DILocation(line: 26, column: 1, scope: !12)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.EmptyRange.b30481ddd22ae53e.Main:Iterate.Core.3c6fb8ac8f4bd886", scope: null, file: !4, line: 15, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !7)
// CHECK:STDOUT: !40 = !DILocation(line: 16, column: 7, scope: !35)
// CHECK:STDOUT: !41 = !{!36}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.EmptyRange.b30481ddd22ae53e.Main:Iterate.Core.3c6fb8ac8f4bd886", scope: null, file: !4, line: 18, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !53)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !7)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 2, scope: !42, type: !7)
// CHECK:STDOUT: !49 = !DILocation(line: 19, column: 14, scope: !42)
// CHECK:STDOUT: !50 = !{!7}
// CHECK:STDOUT: !51 = !DISubroutineType(types: !50)
// CHECK:STDOUT: !52 = !DILocation(line: 19, column: 7, scope: !42)
// CHECK:STDOUT: !53 = !{!43, !44}
// CHECK:STDOUT: !54 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.dc5ffaddab9fc53e", scope: null, file: !54, line: 36, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !63)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !7)
// CHECK:STDOUT: !61 = !DILocation(line: 37, column: 12, scope: !55)
// CHECK:STDOUT: !62 = !DILocation(line: 37, column: 5, scope: !55)
// CHECK:STDOUT: !63 = !{!56}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.dc5ffaddab9fc53e", scope: null, file: !54, line: 39, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !72)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !7)
// CHECK:STDOUT: !70 = !DILocation(line: 40, column: 12, scope: !64)
// CHECK:STDOUT: !71 = !DILocation(line: 40, column: 5, scope: !64)
// CHECK:STDOUT: !72 = !{!65}
// CHECK:STDOUT: !73 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ffbc82e0019b7bf2", scope: null, file: !73, line: 35, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !7)
// CHECK:STDOUT: !79 = !DILocation(line: 35, column: 1, scope: !74)
// CHECK:STDOUT: !80 = !{!75}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.decc97f51bbf9647", scope: null, file: !73, line: 35, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !87)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !7)
// CHECK:STDOUT: !86 = !DILocation(line: 35, column: 1, scope: !81)
// CHECK:STDOUT: !87 = !{!82}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !73, line: 35, type: !32, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !94)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !6)
// CHECK:STDOUT: !93 = !DILocation(line: 35, column: 1, scope: !88)
// CHECK:STDOUT: !94 = !{!89}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.61b5bb2079973306", scope: null, file: !73, line: 35, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !101)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !7)
// CHECK:STDOUT: !100 = !DILocation(line: 35, column: 1, scope: !95)
// CHECK:STDOUT: !101 = !{!96}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.feee70e0690b5123", scope: null, file: !73, line: 35, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !108)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !7)
// CHECK:STDOUT: !107 = !DILocation(line: 35, column: 1, scope: !102)
// CHECK:STDOUT: !108 = !{!103}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.dc5ffaddab9fc53e", scope: null, file: !54, line: 30, type: !51, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !112 = !DILocation(line: 31, column: 12, scope: !109)
// CHECK:STDOUT: !113 = !DILocation(line: 31, column: 5, scope: !109)
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.c1cc3e41344c17ec:OptionalStorage.Core.3c6fb8ac8f4bd886", scope: null, file: !54, line: 141, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !121)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !114, type: !7)
// CHECK:STDOUT: !119 = !DILocation(line: 142, column: 12, scope: !114)
// CHECK:STDOUT: !120 = !DILocation(line: 142, column: 5, scope: !114)
// CHECK:STDOUT: !121 = !{!115}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.c1cc3e41344c17ec:OptionalStorage.Core.3c6fb8ac8f4bd886", scope: null, file: !54, line: 144, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !129)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !7)
// CHECK:STDOUT: !127 = !DILocation(line: 145, column: 12, scope: !122)
// CHECK:STDOUT: !128 = !DILocation(line: 145, column: 5, scope: !122)
// CHECK:STDOUT: !129 = !{!123}
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "None", linkageName: "_CNone.c1cc3e41344c17ec:OptionalStorage.Core.3c6fb8ac8f4bd886", scope: null, file: !54, line: 125, type: !51, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !137 = !DILocation(line: 128, column: 5, scope: !130)
// CHECK:STDOUT: !138 = !DILocation(line: 129, column: 5, scope: !130)
// CHECK:STDOUT: !139 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !140 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4e4edddc7580c79b:Copy.Core.3eda18978e8f48e1", scope: null, file: !139, line: 62, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !149)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !140, type: !7)
// CHECK:STDOUT: !145 = !DILocation(line: 63, column: 13, scope: !140)
// CHECK:STDOUT: !146 = !DILocation(line: 63, column: 12, scope: !140)
// CHECK:STDOUT: !147 = !DILocation(line: 63, column: 26, scope: !140)
// CHECK:STDOUT: !148 = !DILocation(line: 63, column: 5, scope: !140)
// CHECK:STDOUT: !149 = !{!141}
// CHECK:STDOUT:
