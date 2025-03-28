// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/base.carbon

base class Base {
  var b: i32;
}

class Derived {
  extend base: Base;

  var d: i32;
}

fn Make() -> Derived {
  return {.base = {.b = 4}, .d = 7};
}

fn Access(d: Derived) -> (i32, i32) {
  return (d.d, d.base.b);
}

fn Convert(p: Derived*) -> Base* {
  return p;
}

// CHECK:STDOUT: ; ModuleID = 'base.carbon'
// CHECK:STDOUT: source_filename = "base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Derived.val.loc22_36 = internal constant { { i32 }, i32 } { { i32 } { i32 4 }, i32 7 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMake.Main(ptr sret({ { i32 }, i32 }) %return) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_35.2.base = getelementptr inbounds nuw { { i32 }, i32 }, ptr %return, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %.loc22_26.3.b = getelementptr inbounds nuw { i32 }, ptr %.loc22_35.2.base, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc22_35.5.d = getelementptr inbounds nuw { { i32 }, i32 }, ptr %return, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @Derived.val.loc22_36, i64 8, i1 false), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CAccess.Main(ptr sret({ i32, i32 }) %return, ptr %d) !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26_12.1.d = getelementptr inbounds nuw { { i32 }, i32 }, ptr %d, i32 0, i32 1, !dbg !11
// CHECK:STDOUT:   %.loc26_12.2 = load i32, ptr %.loc26_12.1.d, align 4, !dbg !11
// CHECK:STDOUT:   %.loc26_17.1.base = getelementptr inbounds nuw { { i32 }, i32 }, ptr %d, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   %.loc26_22.1.b = getelementptr inbounds nuw { i32 }, ptr %.loc26_17.1.base, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   %.loc26_22.2 = load i32, ptr %.loc26_22.1.b, align 4, !dbg !12
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !13
// CHECK:STDOUT:   store i32 %.loc26_12.2, ptr %tuple.elem0.tuple.elem, align 4, !dbg !13
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !13
// CHECK:STDOUT:   store i32 %.loc26_22.2, ptr %tuple.elem1.tuple.elem, align 4, !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CConvert.Main(ptr %p) !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc30_11.2.base = getelementptr inbounds nuw { { i32 }, i32 }, ptr %p, i32 0, i32 0, !dbg !16
// CHECK:STDOUT:   ret ptr %.loc30_11.2.base, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "base.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 22, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 22, column: 19, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 22, column: 3, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Access", linkageName: "_CAccess.Main", scope: null, file: !3, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 26, column: 11, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 26, column: 16, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 26, column: 10, scope: !10)
// CHECK:STDOUT: !14 = !DILocation(line: 26, column: 3, scope: !10)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !3, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 30, column: 3, scope: !15)
