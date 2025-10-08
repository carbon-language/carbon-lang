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

class EmptyRange(T:! Core.Copy) {
  impl as Core.Iterate where .CursorType = {} and .ElementType = T {
    fn NewCursor[self: Self]() -> {} {
      return {};
    }
    fn Next[self: Self](cursor: {}*) -> Core.Optional(T) {
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
// CHECK:STDOUT: @EmptyRange.val.loc27_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(i32, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CFor.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %r.var = alloca {}, align 8, !dbg !7
// CHECK:STDOUT:   %var = alloca {}, align 8, !dbg !8
// CHECK:STDOUT:   %.loc29_33.1.temp = alloca { i1, { i32, i32 } }, align 8, !dbg !8
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !9
// CHECK:STDOUT:   %.loc29_33.8.temp = alloca { i32, i32 }, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %r.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %r.var, ptr align 1 @EmptyRange.val.loc27_3.1, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   call void @"_CNewCursor.EmptyRange.Main:Iterate.Core.54ec9394d5df5464"(ptr %r.var), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.EmptyRange.Main:Iterate.Core.54ec9394d5df5464"(ptr %.loc29_33.1.temp, ptr %r.var, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.54ec9394d5df5464(ptr %.loc29_33.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_33.8.temp), !dbg !8
// CHECK:STDOUT:   call void @_CGet.Optional.Core.54ec9394d5df5464(ptr %.loc29_33.8.temp, ptr %.loc29_33.1.temp), !dbg !8
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
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_CNewCursor.EmptyRange.Main:Iterate.Core.54ec9394d5df5464"(ptr %self) !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_CNext.EmptyRange.Main:Iterate.Core.54ec9394d5df5464"(ptr sret({ i1, { i32, i32 } }) %return, ptr %self, ptr %cursor) !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CNone.Optional.Core.54ec9394d5df5464(ptr %return), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.54ec9394d5df5464(ptr %self) !dbg !18 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i1, { i32, i32 } }, ptr %self, i32 0, i32 0, !dbg !20
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !20
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !20
// CHECK:STDOUT:   ret i1 %2, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CGet.Optional.Core.54ec9394d5df5464(ptr sret({ i32, i32 }) %return, ptr %self) !dbg !22 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i1, { i32, i32 } }, ptr %self, i32 0, i32 1, !dbg !23
// CHECK:STDOUT:   call void @"_COp.0f13b78cd2788a9b:Copy.Core.58c6957df82926a9"(ptr %return, ptr %value), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.54ec9394d5df5464(ptr sret({ i1, { i32, i32 } }) %return) !dbg !25 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i1, { i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !26
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_COp.0f13b78cd2788a9b:Copy.Core.58c6957df82926a9"(ptr sret({ i32, i32 }) %return, ptr %self) !dbg !28 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !30
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !30
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !32
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !32
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !31
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !31
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !31
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "bindings.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 27, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 29, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 29, column: 17, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 30, column: 5, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 29, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 26, column: 1, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.EmptyRange.Main:Iterate.Core.54ec9394d5df5464", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !14 = !DILocation(line: 16, column: 7, scope: !13)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.EmptyRange.Main:Iterate.Core.54ec9394d5df5464", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 19, column: 14, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 19, column: 7, scope: !15)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.54ec9394d5df5464", scope: null, file: !19, line: 30, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !20 = !DILocation(line: 30, column: 46, scope: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 30, column: 39, scope: !18)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.54ec9394d5df5464", scope: null, file: !19, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !23 = !DILocation(line: 31, column: 38, scope: !22)
// CHECK:STDOUT: !24 = !DILocation(line: 31, column: 31, scope: !22)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.54ec9394d5df5464", scope: null, file: !19, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !26 = !DILocation(line: 22, column: 5, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 23, column: 5, scope: !25)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0f13b78cd2788a9b:Copy.Core.58c6957df82926a9", scope: null, file: !29, line: 58, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !29 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !30 = !DILocation(line: 59, column: 13, scope: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 59, column: 12, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 59, column: 26, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 59, column: 5, scope: !28)
