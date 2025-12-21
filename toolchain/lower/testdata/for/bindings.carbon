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
    fn NewCursor[self: Self]() -> {} {
      return {};
    }
    fn Next[self: Self](_: {}*) -> Core.Optional(T) {
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
// CHECK:STDOUT:   %r.var = alloca {}, align 8, !dbg !7
// CHECK:STDOUT:   %var = alloca {}, align 8, !dbg !8
// CHECK:STDOUT:   %.loc29_33.1.temp = alloca { { i32, i32 }, i1 }, align 8, !dbg !8
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !9
// CHECK:STDOUT:   %.loc29_33.8.temp = alloca { i32, i32 }, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %r.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %r.var, ptr align 1 @EmptyRange.val.loc27_3, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   call void @"_CNewCursor.EmptyRange.1ccc388edee5a786.Main:Iterate.Core.c6c37881d2ed7d9d"(ptr %r.var), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.EmptyRange.1ccc388edee5a786.Main:Iterate.Core.c6c37881d2ed7d9d"(ptr %.loc29_33.1.temp, ptr %r.var, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.eeebe1ada0072aea(ptr %.loc29_33.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.8.temp), !dbg !8
// CHECK:STDOUT:   call void @_CGet.Optional.Core.eeebe1ada0072aea(ptr %.loc29_33.8.temp, ptr %.loc29_33.1.temp), !dbg !8
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc29_33.8.temp, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc29_33.8.temp, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   %.loc29_33.11 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !8
// CHECK:STDOUT:   %.loc29_33.12 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !8
// CHECK:STDOUT:   store i32 %.loc29_33.12, ptr %n.var, align 4, !dbg !9
// CHECK:STDOUT:   call void @_CF.Main(i32 %.loc29_33.11, ptr %n.var), !dbg !10
// CHECK:STDOUT:   br label %for.next, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNewCursor.EmptyRange.1ccc388edee5a786.Main:Iterate.Core.c6c37881d2ed7d9d"(ptr %self) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.EmptyRange.1ccc388edee5a786.Main:Iterate.Core.c6c37881d2ed7d9d"(ptr sret({ { i32, i32 }, i1 }) %return, ptr %self, ptr %_) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CNone.Optional.Core.eeebe1ada0072aea(ptr %return), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.eeebe1ada0072aea(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.3e8267224c5dc9c2:OptionalStorage.Core.c6c37881d2ed7d9d"(ptr %self), !dbg !32
// CHECK:STDOUT:   ret i1 %1, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGet.Optional.Core.eeebe1ada0072aea(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !34 {
// CHECK:STDOUT:   call void @"_CGet.3e8267224c5dc9c2:OptionalStorage.Core.c6c37881d2ed7d9d"(ptr %return, ptr %self), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.eeebe1ada0072aea(ptr sret({ { i32, i32 }, i1 }) %return) #0 !dbg !39 {
// CHECK:STDOUT:   call void @"_CNone.3e8267224c5dc9c2:OptionalStorage.Core.c6c37881d2ed7d9d"(ptr %return), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.3e8267224c5dc9c2:OptionalStorage.Core.c6c37881d2ed7d9d"(ptr %value) #0 !dbg !44 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { { i32, i32 }, i1 }, ptr %value, i32 0, i32 1, !dbg !47
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !47
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !47
// CHECK:STDOUT:   ret i1 %2, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CGet.3e8267224c5dc9c2:OptionalStorage.Core.c6c37881d2ed7d9d"(ptr sret({ i32, i32 }) %return, ptr %value) #0 !dbg !49 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { { i32, i32 }, i1 }, ptr %value, i32 0, i32 0, !dbg !52
// CHECK:STDOUT:   call void @"_COp.e4daa70d026fdea2:Copy.Core.3e36085cb869f630"(ptr %return, ptr %value1), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.3e8267224c5dc9c2:OptionalStorage.Core.c6c37881d2ed7d9d"(ptr sret({ { i32, i32 }, i1 }) %return) #0 !dbg !54 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { { i32, i32 }, i1 }, ptr %return, i32 0, i32 1, !dbg !55
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !55
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.e4daa70d026fdea2:Copy.Core.3e36085cb869f630"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !57 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !61
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !61
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !62
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !63
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !63
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !62
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !62
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !62
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "bindings.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 27, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 29, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 29, column: 17, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 30, column: 5, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 29, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 26, column: 1, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.EmptyRange.1ccc388edee5a786.Main:Iterate.Core.c6c37881d2ed7d9d", scope: null, file: !3, line: 15, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !17)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{!16, !16}
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !13, type: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 16, column: 7, scope: !13)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.EmptyRange.1ccc388edee5a786.Main:Iterate.Core.c6c37881d2ed7d9d", scope: null, file: !3, line: 18, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{!16, !16, !16}
// CHECK:STDOUT: !23 = !{!24, !25}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !20, type: !16)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 2, scope: !20, type: !16)
// CHECK:STDOUT: !26 = !DILocation(line: 19, column: 14, scope: !20)
// CHECK:STDOUT: !27 = !DILocation(line: 19, column: 7, scope: !20)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.eeebe1ada0072aea", scope: null, file: !29, line: 32, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !30)
// CHECK:STDOUT: !29 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !28, type: !16)
// CHECK:STDOUT: !32 = !DILocation(line: 33, column: 12, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 33, column: 5, scope: !28)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.eeebe1ada0072aea", scope: null, file: !29, line: 35, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !16)
// CHECK:STDOUT: !37 = !DILocation(line: 36, column: 12, scope: !34)
// CHECK:STDOUT: !38 = !DILocation(line: 36, column: 5, scope: !34)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.eeebe1ada0072aea", scope: null, file: !29, line: 26, type: !40, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !40 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !41 = !{!16}
// CHECK:STDOUT: !42 = !DILocation(line: 27, column: 12, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 27, column: 5, scope: !39)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.3e8267224c5dc9c2:OptionalStorage.Core.c6c37881d2ed7d9d", scope: null, file: !29, line: 112, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !45)
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !44, type: !16)
// CHECK:STDOUT: !47 = !DILocation(line: 113, column: 12, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 113, column: 5, scope: !44)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.3e8267224c5dc9c2:OptionalStorage.Core.c6c37881d2ed7d9d", scope: null, file: !29, line: 115, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !16)
// CHECK:STDOUT: !52 = !DILocation(line: 116, column: 12, scope: !49)
// CHECK:STDOUT: !53 = !DILocation(line: 116, column: 5, scope: !49)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "None", linkageName: "_CNone.3e8267224c5dc9c2:OptionalStorage.Core.c6c37881d2ed7d9d", scope: null, file: !29, line: 99, type: !40, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !55 = !DILocation(line: 101, column: 5, scope: !54)
// CHECK:STDOUT: !56 = !DILocation(line: 102, column: 5, scope: !54)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e4daa70d026fdea2:Copy.Core.3e36085cb869f630", scope: null, file: !58, line: 58, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !59)
// CHECK:STDOUT: !58 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !59 = !{!60}
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !57, type: !16)
// CHECK:STDOUT: !61 = !DILocation(line: 59, column: 13, scope: !57)
// CHECK:STDOUT: !62 = !DILocation(line: 59, column: 12, scope: !57)
// CHECK:STDOUT: !63 = !DILocation(line: 59, column: 26, scope: !57)
// CHECK:STDOUT: !64 = !DILocation(line: 59, column: 5, scope: !57)
