// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call.carbon

fn F[T:! type](x: T) {
}

class C {}
class D {}

fn G() {
  var c: C = {};
  var d: D = {};
  var n: i32 = 0;

  F(c);
  F(d);
  F(n);
  F(i32);
}

// CHECK:STDOUT: ; ModuleID = 'call.carbon'
// CHECK:STDOUT: source_filename = "call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: @struct.1.loc18_16 = internal constant {} zeroinitializer
// CHECK:STDOUT: @struct.2.loc19_16 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CG.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @struct.1.loc18_16, i64 0, i1 false), !dbg !8
// CHECK:STDOUT:   %d.var = alloca {}, align 8, !dbg !9
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %d.var, ptr align 1 @struct.2.loc19_16, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !11
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !12
// CHECK:STDOUT:   call void @_CF.Main.46(ptr %c.var), !dbg !13
// CHECK:STDOUT:   call void @_CF.Main.47(ptr %d.var), !dbg !14
// CHECK:STDOUT:   %.loc24_5 = load i32, ptr %n.var, align 4, !dbg !15
// CHECK:STDOUT:   call void @_CF.Main.48(i32 %.loc24_5), !dbg !16
// CHECK:STDOUT:   call void @_CF.Main.49(%type zeroinitializer), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main.46(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main.47(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main.48(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main.49(%type)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 18, column: 7, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 19, column: 7, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 19, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 20, column: 7, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 22, column: 3, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 23, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 24, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 24, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 25, column: 3, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 17, column: 1, scope: !4)
