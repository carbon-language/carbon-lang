// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/tuple/access/return_value_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/tuple/access/return_value_access.carbon

fn F() -> (i32, i32) { return (12, 24); }

fn Run() {
  var t: i32 = F().1;
}

// CHECK:STDOUT: ; ModuleID = 'return_value_access.carbon'
// CHECK:STDOUT: source_filename = "return_value_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.loc11_39 = internal constant { i32, i32 } { i32 12, i32 24 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ i32, i32 }) %return) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @tuple.loc11_39, i64 8, i1 false), !dbg !8
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @main() !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %t.var = alloca i32, align 4, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %t.var), !dbg !10
// CHECK:STDOUT:   %.loc14_18.1.temp = alloca { i32, i32 }, align 8, !dbg !11
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc14_18.1.temp), !dbg !11
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc14_18.1.temp, i32 0, i32 1, !dbg !11
// CHECK:STDOUT:   %.loc14_19 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !11
// CHECK:STDOUT:   store i32 %.loc14_19, ptr %t.var, align 4, !dbg !10
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "return_value_access.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 11, column: 31, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 11, column: 24, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 14, column: 3, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 14, column: 16, scope: !9)
// CHECK:STDOUT: !12 = !DILocation(line: 13, column: 1, scope: !9)
