// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/pointer/pointer_to_pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/pointer/pointer_to_pointer.carbon

fn F(p: i32**) -> i32 {
  var _: i32** = p;
  var b: i32* = *p;
  var c: i32** = &b;
  return **c;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'pointer_to_pointer.carbon'
// CHECK:STDOUT: source_filename = "pointer_to_pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main(ptr %p) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca ptr, align 8, !dbg !29
// CHECK:STDOUT:   %b.var = alloca ptr, align 8, !dbg !32
// CHECK:STDOUT:   %c.var = alloca ptr, align 8, !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !29
// CHECK:STDOUT:   store ptr %p, ptr %_.var, align 8, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !32
// CHECK:STDOUT:   %.loc15_17.2 = load ptr, ptr %p, align 8, !dbg !34
// CHECK:STDOUT:   store ptr %.loc15_17.2, ptr %b.var, align 8, !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !36
// CHECK:STDOUT:   store ptr %b.var, ptr %c.var, align 8, !dbg !36
// CHECK:STDOUT:   %.loc17_12 = load ptr, ptr %c.var, align 8, !dbg !40
// CHECK:STDOUT:   %.loc17_11.2 = load ptr, ptr %.loc17_12, align 8, !dbg !41
// CHECK:STDOUT:   %.loc17_10.2 = load i32, ptr %.loc17_11.2, align 4, !dbg !42
// CHECK:STDOUT:   ret i32 %.loc17_10.2, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "pointer_to_pointer.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 13, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !44)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 14, column: 3, scope: !24)
// CHECK:STDOUT: !32 = !DILocation(line: 15, column: 3, scope: !24)
// CHECK:STDOUT: !34 = !DILocation(line: 15, column: 17, scope: !24)
// CHECK:STDOUT: !36 = !DILocation(line: 16, column: 3, scope: !24)
// CHECK:STDOUT: !40 = !DILocation(line: 17, column: 12, scope: !24)
// CHECK:STDOUT: !41 = !DILocation(line: 17, column: 11, scope: !24)
// CHECK:STDOUT: !42 = !DILocation(line: 17, column: 10, scope: !24)
// CHECK:STDOUT: !43 = !DILocation(line: 17, column: 3, scope: !24)
// CHECK:STDOUT: !44 = !{!25}
// CHECK:STDOUT:
