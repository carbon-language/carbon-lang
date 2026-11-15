// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'base.carbon'
// CHECK:STDOUT: source_filename = "base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Derived.val.loc24_36 = internal constant { { i32 }, i32 } { { i32 } { i32 4 }, i32 7 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.Main(ptr sret({ { i32 }, i32 }) %return) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc24_35.2.base = getelementptr inbounds nuw { { i32 }, i32 }, ptr %return, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %.loc24_26.3.b = getelementptr inbounds nuw { i32 }, ptr %.loc24_35.2.base, i32 0, i32 0, !dbg !32
// CHECK:STDOUT:   %.loc24_35.6.d = getelementptr inbounds nuw { { i32 }, i32 }, ptr %return, i32 0, i32 1, !dbg !31
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @Derived.val.loc24_36, i64 8, i1 false), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAccess.Main(ptr sret({ i32, i32 }) %return, ptr %d) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc28_12.1.d = getelementptr inbounds nuw { { i32 }, i32 }, ptr %d, i32 0, i32 1, !dbg !39
// CHECK:STDOUT:   %.loc28_12.2 = load i32, ptr %.loc28_12.1.d, align 4, !dbg !39
// CHECK:STDOUT:   %.loc28_17.1.base = getelementptr inbounds nuw { { i32 }, i32 }, ptr %d, i32 0, i32 0, !dbg !40
// CHECK:STDOUT:   %.loc28_22.1.b = getelementptr inbounds nuw { i32 }, ptr %.loc28_17.1.base, i32 0, i32 0, !dbg !40
// CHECK:STDOUT:   %.loc28_22.2 = load i32, ptr %.loc28_22.1.b, align 4, !dbg !40
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !41
// CHECK:STDOUT:   store i32 %.loc28_12.2, ptr %tuple.elem0.tuple.elem, align 4, !dbg !41
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !41
// CHECK:STDOUT:   store i32 %.loc28_22.2, ptr %tuple.elem1.tuple.elem, align 4, !dbg !41
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvert.Main(ptr %p) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_11.2.base = getelementptr inbounds nuw { { i32 }, i32 }, ptr %p, i32 0, i32 0, !dbg !50
// CHECK:STDOUT:   ret ptr %.loc32_11.2.base, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "base.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!23, !23}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !20, line: 23, type: !25, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !31 = !DILocation(line: 24, column: 10, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 24, column: 19, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 24, column: 3, scope: !28)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Access", linkageName: "_CAccess.Main", scope: null, file: !20, line: 27, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !43)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !23)
// CHECK:STDOUT: !39 = !DILocation(line: 28, column: 11, scope: !34)
// CHECK:STDOUT: !40 = !DILocation(line: 28, column: 16, scope: !34)
// CHECK:STDOUT: !41 = !DILocation(line: 28, column: 10, scope: !34)
// CHECK:STDOUT: !42 = !DILocation(line: 28, column: 3, scope: !34)
// CHECK:STDOUT: !43 = !{!35}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !20, line: 31, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !51)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !23)
// CHECK:STDOUT: !50 = !DILocation(line: 32, column: 3, scope: !44)
// CHECK:STDOUT: !51 = !{!45}
// CHECK:STDOUT:
