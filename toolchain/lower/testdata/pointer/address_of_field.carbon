// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/pointer/address_of_field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/pointer/address_of_field.carbon

fn G(p: i32*);

fn F() {
  var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  G(&s.b);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'address_of_field.carbon'
// CHECK:STDOUT: source_filename = "address_of_field.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @struct.940.loc16_3.1 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %s.var = alloca { i32, i32 }, align 4, !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %s.var), !dbg !27
// CHECK:STDOUT:   %.loc16_46.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %s.var, i32 0, i32 0, !dbg !28
// CHECK:STDOUT:   %.loc16_46.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %s.var, i32 0, i32 1, !dbg !28
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %s.var, ptr align 4 @struct.940.loc16_3.1, i64 8, i1 false), !dbg !27
// CHECK:STDOUT:   %.loc17.b = getelementptr inbounds nuw { i32, i32 }, ptr %s.var, i32 0, i32 1, !dbg !30
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc17.b), !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ad01ed096e1f14e2(ptr %s.var), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ad01ed096e1f14e2(ptr %t) #0 !dbg !34 {
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "address_of_field.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !27 = !DILocation(line: 16, column: 3, scope: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 16, column: 31, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 17, column: 6, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 17, column: 3, scope: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 15, column: 1, scope: !26)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ad01ed096e1f14e2", scope: null, file: !10, line: 32, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !40)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !21)
// CHECK:STDOUT: !39 = !DILocation(line: 32, column: 1, scope: !34)
// CHECK:STDOUT: !40 = !{!35}
// CHECK:STDOUT:
