// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/field.carbon

class A {
  var v: array(i32, 2);

  fn Init() -> A { return {.v = (1, 2)}; }

  fn Access(self) -> i32 {
    return self.v[0];
  }

  fn Use(ref self) -> i32 {
    self.v[0] = 1;
    return self.v[1];
  }
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'field.carbon'
// CHECK:STDOUT: source_filename = "field.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @A.val.loc16_40 = internal constant { [2 x i32] } { [2 x i32] [i32 1, i32 2] }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInit.A.Main(ptr sret({ [2 x i32] }) %return) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_39.2.v = getelementptr inbounds nuw { [2 x i32] }, ptr %return, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %.loc16_38.3.array.index = getelementptr inbounds [2 x i32], ptr %.loc16_39.2.v, i32 0, i64 0, !dbg !32
// CHECK:STDOUT:   %.loc16_38.6.array.index = getelementptr inbounds [2 x i32], ptr %.loc16_39.2.v, i32 0, i64 1, !dbg !32
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @A.val.loc16_40, i64 8, i1 false), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccess.A.Main(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_16.1.v = getelementptr inbounds nuw { [2 x i32] }, ptr %self, i32 0, i32 0, !dbg !39
// CHECK:STDOUT:   %.loc19_20.2.array.index = getelementptr inbounds [2 x i32], ptr %.loc19_16.1.v, i32 0, i32 0, !dbg !39
// CHECK:STDOUT:   %.loc19_20.3 = load i32, ptr %.loc19_20.2.array.index, align 4, !dbg !39
// CHECK:STDOUT:   ret i32 %.loc19_20.3, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUse.A.Main(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_9.v = getelementptr inbounds nuw { [2 x i32] }, ptr %self, i32 0, i32 0, !dbg !47
// CHECK:STDOUT:   %.loc23_13.array.index = getelementptr inbounds [2 x i32], ptr %.loc23_9.v, i32 0, i32 0, !dbg !47
// CHECK:STDOUT:   store i32 1, ptr %.loc23_13.array.index, align 4, !dbg !47
// CHECK:STDOUT:   %.loc24_16.v = getelementptr inbounds nuw { [2 x i32] }, ptr %self, i32 0, i32 0, !dbg !48
// CHECK:STDOUT:   %.loc24_20.1.array.index = getelementptr inbounds [2 x i32], ptr %.loc24_16.v, i32 0, i32 1, !dbg !48
// CHECK:STDOUT:   %.loc24_20.2 = load i32, ptr %.loc24_20.1.array.index, align 4, !dbg !48
// CHECK:STDOUT:   ret i32 %.loc24_20.2, !dbg !49
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
// CHECK:STDOUT: !20 = !DIFile(filename: "field.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!22, !23}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Init", linkageName: "_CInit.A.Main", scope: null, file: !20, line: 16, type: !25, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !31 = !DILocation(line: 16, column: 27, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 16, column: 33, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 16, column: 20, scope: !28)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Access", linkageName: "_CAccess.A.Main", scope: null, file: !20, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !41)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !23)
// CHECK:STDOUT: !39 = !DILocation(line: 19, column: 12, scope: !34)
// CHECK:STDOUT: !40 = !DILocation(line: 19, column: 5, scope: !34)
// CHECK:STDOUT: !41 = !{!35}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.A.Main", scope: null, file: !20, line: 22, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !50)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !23)
// CHECK:STDOUT: !47 = !DILocation(line: 23, column: 5, scope: !42)
// CHECK:STDOUT: !48 = !DILocation(line: 24, column: 12, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 24, column: 5, scope: !42)
// CHECK:STDOUT: !50 = !{!43}
// CHECK:STDOUT:
