// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/field.carbon

class A {
  var v: array(i32, 2);

  // TODO: The LLVM IR we create for this crashes LLVM instruction selection.
  // The gep indexes are completely bogus.
  fn Init() -> A { return {.v = (1, 2)}; }

  fn Access[self: Self]() -> i32 {
    return self.v[0];
  }

  fn Use[addr self: Self*]() -> i32 {
    self->v[0] = 1;
    return self->v[1];
  }
}

// CHECK:STDOUT: ; ModuleID = 'field.carbon'
// CHECK:STDOUT: source_filename = "field.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @A.val.loc16_40 = internal constant { [2 x i32] } { [2 x i32] [i32 1, i32 2] }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CInit.A.Main(ptr sret({ [2 x i32] }) %return) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_39.2.v = getelementptr inbounds nuw { [2 x i32] }, ptr %return, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %.loc16_38.3.array.index = getelementptr inbounds [2 x i32], ptr %.loc16_39.2.v, i32 0, i64 0, !dbg !8
// CHECK:STDOUT:   %.loc16_38.6.array.index = getelementptr inbounds [2 x i32], ptr %.loc16_39.2.v, i32 0, i64 1, !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @A.val.loc16_40, i64 8, i1 false), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CAccess.A.Main(ptr %self) !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_16.1.v = getelementptr inbounds nuw { [2 x i32] }, ptr %self, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   %.loc19_20.2.array.index = getelementptr inbounds [2 x i32], ptr %.loc19_16.1.v, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   %.loc19_20.3 = load i32, ptr %.loc19_20.2.array.index, align 4, !dbg !11
// CHECK:STDOUT:   ret i32 %.loc19_20.3, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CUse.A.Main(ptr %self) !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_9.2.v = getelementptr inbounds nuw { [2 x i32] }, ptr %self, i32 0, i32 0, !dbg !14
// CHECK:STDOUT:   %.loc23_14.array.index = getelementptr inbounds [2 x i32], ptr %.loc23_9.2.v, i32 0, i32 0, !dbg !14
// CHECK:STDOUT:   store i32 1, ptr %.loc23_14.array.index, align 4, !dbg !14
// CHECK:STDOUT:   %.loc24_16.2.v = getelementptr inbounds nuw { [2 x i32] }, ptr %self, i32 0, i32 0, !dbg !15
// CHECK:STDOUT:   %.loc24_21.1.array.index = getelementptr inbounds [2 x i32], ptr %.loc24_16.2.v, i32 0, i32 1, !dbg !15
// CHECK:STDOUT:   %.loc24_21.2 = load i32, ptr %.loc24_21.1.array.index, align 4, !dbg !15
// CHECK:STDOUT:   ret i32 %.loc24_21.2, !dbg !16
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
// CHECK:STDOUT: !3 = !DIFile(filename: "field.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Init", linkageName: "_CInit.A.Main", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 16, column: 27, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 16, column: 33, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 16, column: 20, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Access", linkageName: "_CAccess.A.Main", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 19, column: 12, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 19, column: 5, scope: !10)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.A.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !14 = !DILocation(line: 23, column: 5, scope: !13)
// CHECK:STDOUT: !15 = !DILocation(line: 24, column: 12, scope: !13)
// CHECK:STDOUT: !16 = !DILocation(line: 24, column: 5, scope: !13)
