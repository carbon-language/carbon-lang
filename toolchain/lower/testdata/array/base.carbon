// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/base.carbon

fn Run() {
  var _: array(i32, 1) = (1,);
  var _: array(f64, 2) = (11.1, 2.2,);
  var _: array((), 5) = ((), (), (), (), (),);
  var d: (i32, i32, i32) = (1, 2, 3);
  var _: array(i32, 3) = d;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'base.carbon'
// CHECK:STDOUT: source_filename = "base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array.7dc.loc14_3 = internal constant [1 x i32] [i32 1]
// CHECK:STDOUT: @array.38b.loc15_3 = internal constant [2 x double] [double 1.110000e+01, double 2.200000e+00]
// CHECK:STDOUT: @array.29b.loc16_3 = internal constant [5 x {}] zeroinitializer
// CHECK:STDOUT: @tuple.80e.loc17_3 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc14 = alloca [1 x i32], align 4, !dbg !40
// CHECK:STDOUT:   %_.var.loc15 = alloca [2 x double], align 8, !dbg !43
// CHECK:STDOUT:   %_.var.loc16 = alloca [5 x {}], align 1, !dbg !46
// CHECK:STDOUT:   %d.var = alloca { i32, i32, i32 }, align 4, !dbg !49
// CHECK:STDOUT:   %_.var.loc18 = alloca [3 x i32], align 4, !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !40
// CHECK:STDOUT:   %.loc14_29.3.array.index = getelementptr inbounds i8, ptr %_.var.loc14, i64 0, !dbg !41
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %_.var.loc14, ptr align 4 @array.7dc.loc14_3, i64 4, i1 false), !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !43
// CHECK:STDOUT:   %.loc15_37.3.array.index = getelementptr inbounds i8, ptr %_.var.loc15, i64 0, !dbg !44
// CHECK:STDOUT:   %.loc15_37.6.array.index = getelementptr inbounds i8, ptr %_.var.loc15, i64 8, !dbg !44
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc15, ptr align 8 @array.38b.loc15_3, i64 16, i1 false), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !46
// CHECK:STDOUT:   %.loc16_45.2.array.index = getelementptr inbounds i8, ptr %_.var.loc16, i64 0, !dbg !47
// CHECK:STDOUT:   %.loc16_45.5.array.index = getelementptr inbounds i8, ptr %_.var.loc16, i64 0, !dbg !47
// CHECK:STDOUT:   %.loc16_45.8.array.index = getelementptr inbounds i8, ptr %_.var.loc16, i64 0, !dbg !47
// CHECK:STDOUT:   %.loc16_45.11.array.index = getelementptr inbounds i8, ptr %_.var.loc16, i64 0, !dbg !47
// CHECK:STDOUT:   %.loc16_45.14.array.index = getelementptr inbounds i8, ptr %_.var.loc16, i64 0, !dbg !47
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc16, ptr align 1 @array.29b.loc16_3, i64 0, i1 false), !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !49
// CHECK:STDOUT:   %tuple.elem0.loc17.tuple.elem = getelementptr inbounds nuw i8, ptr %d.var, i64 0, !dbg !50
// CHECK:STDOUT:   %tuple.elem1.loc17.tuple.elem = getelementptr inbounds nuw i8, ptr %d.var, i64 4, !dbg !50
// CHECK:STDOUT:   %tuple.elem2.loc17.tuple.elem = getelementptr inbounds nuw i8, ptr %d.var, i64 8, !dbg !50
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %d.var, ptr align 4 @tuple.80e.loc17_3, i64 12, i1 false), !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !52
// CHECK:STDOUT:   %tuple.elem0.loc18.tuple.elem = getelementptr inbounds nuw i8, ptr %d.var, i64 0, !dbg !53
// CHECK:STDOUT:   %.loc18_26.1 = load i32, ptr %tuple.elem0.loc18.tuple.elem, align 4, !dbg !53
// CHECK:STDOUT:   %.loc18_26.2.array.index = getelementptr inbounds i8, ptr %_.var.loc18, i64 0, !dbg !53
// CHECK:STDOUT:   store i32 %.loc18_26.1, ptr %.loc18_26.2.array.index, align 4, !dbg !53
// CHECK:STDOUT:   %tuple.elem1.loc18.tuple.elem = getelementptr inbounds nuw i8, ptr %d.var, i64 4, !dbg !53
// CHECK:STDOUT:   %.loc18_26.4 = load i32, ptr %tuple.elem1.loc18.tuple.elem, align 4, !dbg !53
// CHECK:STDOUT:   %.loc18_26.5.array.index = getelementptr inbounds i8, ptr %_.var.loc18, i64 4, !dbg !53
// CHECK:STDOUT:   store i32 %.loc18_26.4, ptr %.loc18_26.5.array.index, align 4, !dbg !53
// CHECK:STDOUT:   %tuple.elem2.loc18.tuple.elem = getelementptr inbounds nuw i8, ptr %d.var, i64 8, !dbg !53
// CHECK:STDOUT:   %.loc18_26.7 = load i32, ptr %tuple.elem2.loc18.tuple.elem, align 4, !dbg !53
// CHECK:STDOUT:   %.loc18_26.8.array.index = getelementptr inbounds i8, ptr %_.var.loc18, i64 8, !dbg !53
// CHECK:STDOUT:   store i32 %.loc18_26.7, ptr %.loc18_26.8.array.index, align 4, !dbg !53
// CHECK:STDOUT:   ret i32 0, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "base.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{!36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !34, line: 13, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !40 = !DILocation(line: 14, column: 3, scope: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 14, column: 26, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 15, column: 3, scope: !39)
// CHECK:STDOUT: !44 = !DILocation(line: 15, column: 26, scope: !39)
// CHECK:STDOUT: !46 = !DILocation(line: 16, column: 3, scope: !39)
// CHECK:STDOUT: !47 = !DILocation(line: 16, column: 25, scope: !39)
// CHECK:STDOUT: !49 = !DILocation(line: 17, column: 3, scope: !39)
// CHECK:STDOUT: !50 = !DILocation(line: 17, column: 28, scope: !39)
// CHECK:STDOUT: !52 = !DILocation(line: 18, column: 3, scope: !39)
// CHECK:STDOUT: !53 = !DILocation(line: 18, column: 26, scope: !39)
// CHECK:STDOUT: !55 = !DILocation(line: 13, column: 1, scope: !39)
// CHECK:STDOUT:
