// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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
// CHECK:STDOUT: define void @_CInit.A.Main(ptr sret({ [2 x i32] }) %return) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_39.2.v = getelementptr inbounds nuw { [2 x i32] }, ptr %return, i32 0, i32 0, !dbg !29
// CHECK:STDOUT:   %.loc16_38.3.array.index = getelementptr inbounds [2 x i32], ptr %.loc16_39.2.v, i32 0, i64 0, !dbg !30
// CHECK:STDOUT:   %.loc16_38.6.array.index = getelementptr inbounds [2 x i32], ptr %.loc16_39.2.v, i32 0, i64 1, !dbg !30
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @A.val.loc16_40, i64 8, i1 false), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccess.A.Main(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_16.1.v = getelementptr inbounds nuw { [2 x i32] }, ptr %self, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   %.loc19_20.2.array.index = getelementptr inbounds [2 x i32], ptr %.loc19_16.1.v, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   %.loc19_20.3 = load i32, ptr %.loc19_20.2.array.index, align 4, !dbg !37
// CHECK:STDOUT:   ret i32 %.loc19_20.3, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUse.A.Main(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_9.v = getelementptr inbounds nuw { [2 x i32] }, ptr %self, i32 0, i32 0, !dbg !45
// CHECK:STDOUT:   %.loc23_13.array.index = getelementptr inbounds [2 x i32], ptr %.loc23_9.v, i32 0, i32 0, !dbg !45
// CHECK:STDOUT:   store i32 1, ptr %.loc23_13.array.index, align 4, !dbg !45
// CHECK:STDOUT:   %.loc24_16.v = getelementptr inbounds nuw { [2 x i32] }, ptr %self, i32 0, i32 0, !dbg !46
// CHECK:STDOUT:   %.loc24_20.1.array.index = getelementptr inbounds [2 x i32], ptr %.loc24_16.v, i32 0, i32 1, !dbg !46
// CHECK:STDOUT:   %.loc24_20.2 = load i32, ptr %.loc24_20.1.array.index, align 4, !dbg !46
// CHECK:STDOUT:   ret i32 %.loc24_20.2, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "field.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{!20, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Init", linkageName: "_CInit.A.Main", scope: null, file: !18, line: 16, type: !23, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 27, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 16, column: 33, scope: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 16, column: 20, scope: !26)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Access", linkageName: "_CAccess.A.Main", scope: null, file: !18, line: 18, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !39)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !21)
// CHECK:STDOUT: !37 = !DILocation(line: 19, column: 12, scope: !32)
// CHECK:STDOUT: !38 = !DILocation(line: 19, column: 5, scope: !32)
// CHECK:STDOUT: !39 = !{!33}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.A.Main", scope: null, file: !18, line: 22, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !48)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !21)
// CHECK:STDOUT: !45 = !DILocation(line: 23, column: 5, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 24, column: 12, scope: !40)
// CHECK:STDOUT: !47 = !DILocation(line: 24, column: 5, scope: !40)
// CHECK:STDOUT: !48 = !{!41}
// CHECK:STDOUT:
