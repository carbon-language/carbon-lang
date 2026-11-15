// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/value_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/value_access.carbon

class C {
  var a: (i32, i32, i32);
}

fn F(c: C) -> i32 {
  // TODO: `c.a` is a value expression here, which forces a value acquisition as
  // part of the member access, creating a tuple value temporary. We could
  // defer performing the value acquisition to avoid creating this temporary.
  return c.a.1;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'value_access.carbon'
// CHECK:STDOUT: source_filename = "value_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main(ptr %c) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple = alloca { i32, i32, i32 }, align 4, !dbg !31
// CHECK:STDOUT:   %.loc21_11.1.a = getelementptr inbounds nuw { { i32, i32, i32 } }, ptr %c, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc21_11.1.a, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %.loc21_11.2 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !31
// CHECK:STDOUT:   %tuple.elem1.loc21_11.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc21_11.1.a, i32 0, i32 1, !dbg !31
// CHECK:STDOUT:   %.loc21_11.3 = load i32, ptr %tuple.elem1.loc21_11.tuple.elem, align 4, !dbg !31
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc21_11.1.a, i32 0, i32 2, !dbg !31
// CHECK:STDOUT:   %.loc21_11.4 = load i32, ptr %tuple.elem2.tuple.elem, align 4, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple), !dbg !31
// CHECK:STDOUT:   %tuple1 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   store i32 %.loc21_11.2, ptr %tuple1, align 4, !dbg !31
// CHECK:STDOUT:   %tuple2 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple, i32 0, i32 1, !dbg !31
// CHECK:STDOUT:   store i32 %.loc21_11.3, ptr %tuple2, align 4, !dbg !31
// CHECK:STDOUT:   %tuple3 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple, i32 0, i32 2, !dbg !31
// CHECK:STDOUT:   store i32 %.loc21_11.4, ptr %tuple3, align 4, !dbg !31
// CHECK:STDOUT:   %tuple.elem1.loc21_13.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple, i32 0, i32 1, !dbg !31
// CHECK:STDOUT:   %tuple.elem1.loc21_13.tuple.elem.load = load i32, ptr %tuple.elem1.loc21_13.tuple.elem, align 4, !dbg !31
// CHECK:STDOUT:   ret i32 %tuple.elem1.loc21_13.tuple.elem.load, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "value_access.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!22, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !20, line: 17, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !33)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !23)
// CHECK:STDOUT: !31 = !DILocation(line: 21, column: 10, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 21, column: 3, scope: !26)
// CHECK:STDOUT: !33 = !{!27}
// CHECK:STDOUT:
