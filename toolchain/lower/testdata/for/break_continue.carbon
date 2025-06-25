// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/for/break_continue.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/for/break_continue.carbon

class EmptyRange(T:! type) {
  impl as Core.Iterate where .CursorType = {} and .ElementType = T {
    fn NewCursor[self: Self]() -> {} {
      return {};
    }
    fn Next[self: Self](cursor: {}*) -> Core.Optional(T) {
      return Core.Optional(T).None();
    }
  }
}

fn F() -> bool;
fn G() -> bool;
fn H();

fn For() {
  var r: EmptyRange(i32) = {};
  for (n: i32 in r) {
    if (F()) { break; }
    if (G()) { continue; }
    H();
  }
}

// CHECK:STDOUT: ; ModuleID = 'break_continue.carbon'
// CHECK:STDOUT: source_filename = "break_continue.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @EmptyRange.val.loc27_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CH.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CFor.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %r.var = alloca {}, align 8, !dbg !7
// CHECK:STDOUT:   %var = alloca {}, align 8, !dbg !8
// CHECK:STDOUT:   %.loc28_19.1.temp = alloca { i1, i32 }, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 0, ptr %r.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %r.var, ptr align 1 @EmptyRange.val.loc27_3, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   call void @"_CNewCursor.EmptyRange.Main:Iterate.Core.b88d1103f417c6d4"(ptr %r.var), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 0, ptr %var), !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %if.else.loc30, %if.then.loc30, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %.loc28_19.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.EmptyRange.Main:Iterate.Core.b88d1103f417c6d4"(ptr %.loc28_19.1.temp, ptr %r.var, ptr %var), !dbg !8
// CHECK:STDOUT:   %HasValue.call = call i1 @_CHasValue.Optional.Core.b88d1103f417c6d4(ptr %.loc28_19.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Get.call = call i32 @_CGet.Optional.Core.b88d1103f417c6d4(ptr %.loc28_19.1.temp), !dbg !8
// CHECK:STDOUT:   %F.call = call i1 @_CF.Main(), !dbg !9
// CHECK:STDOUT:   br i1 %F.call, label %if.then.loc29, label %if.else.loc29, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc29:                                    ; preds = %for.body
// CHECK:STDOUT:   br label %for.done, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc29:                                    ; preds = %for.body
// CHECK:STDOUT:   %G.call = call i1 @_CG.Main(), !dbg !12
// CHECK:STDOUT:   br i1 %G.call, label %if.then.loc30, label %if.else.loc30, !dbg !13
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc30:                                    ; preds = %if.else.loc29
// CHECK:STDOUT:   br label %for.next, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc30:                                    ; preds = %if.else.loc29
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !15
// CHECK:STDOUT:   br label %for.next, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %if.then.loc29, %for.next
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_CNewCursor.EmptyRange.Main:Iterate.Core.b88d1103f417c6d4"(ptr %self) !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_CNext.EmptyRange.Main:Iterate.Core.b88d1103f417c6d4"(ptr sret({ i1, i32 }) %return, ptr %self, ptr %cursor) !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CNone.Optional.Core.b88d1103f417c6d4(ptr %return), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.b88d1103f417c6d4(ptr %self) !dbg !23 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 0, !dbg !25
// CHECK:STDOUT:   %1 = load i1, ptr %has_value, align 1, !dbg !25
// CHECK:STDOUT:   ret i1 %1, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.b88d1103f417c6d4(ptr %self) !dbg !27 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 1, !dbg !28
// CHECK:STDOUT:   %1 = load i32, ptr %value, align 4, !dbg !28
// CHECK:STDOUT:   ret i32 %1, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.b88d1103f417c6d4(ptr sret({ i1, i32 }) %return) !dbg !30 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i1, i32 }, ptr %return, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   store i1 false, ptr %has_value, align 1, !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
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
// CHECK:STDOUT: !3 = !DIFile(filename: "break_continue.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 27, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 28, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 29, column: 9, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 29, column: 8, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 29, column: 16, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 30, column: 9, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 30, column: 8, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 30, column: 16, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 31, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 28, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 26, column: 1, scope: !4)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.EmptyRange.Main:Iterate.Core.b88d1103f417c6d4", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 14, column: 7, scope: !18)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.EmptyRange.Main:Iterate.Core.b88d1103f417c6d4", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !21 = !DILocation(line: 17, column: 14, scope: !20)
// CHECK:STDOUT: !22 = !DILocation(line: 17, column: 7, scope: !20)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.b88d1103f417c6d4", scope: null, file: !24, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !24 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !25 = !DILocation(line: 28, column: 46, scope: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 28, column: 39, scope: !23)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.b88d1103f417c6d4", scope: null, file: !24, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 29, column: 38, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 29, column: 31, scope: !27)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.b88d1103f417c6d4", scope: null, file: !24, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !31 = !DILocation(line: 20, column: 5, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 21, column: 5, scope: !30)
