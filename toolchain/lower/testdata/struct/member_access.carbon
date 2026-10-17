// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/struct/member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/struct/member_access.carbon

fn Run() -> i32 {
  var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  var y: i32 = x.b;
  var _: i32 = y;
  return 0;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'member_access.carbon'
// CHECK:STDOUT: source_filename = "member_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @struct.242.loc14_3.1 = internal constant <{ double, i32 }> <{ double 0.000000e+00, i32 1 }>
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca <{ double, i32 }>, align 8, !dbg !42
// CHECK:STDOUT:   %y.var = alloca i32, align 4, !dbg !45
// CHECK:STDOUT:   %_.var = alloca i32, align 4, !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !42
// CHECK:STDOUT:   %.loc14_48.3.a = getelementptr inbounds nuw <{ double, i32 }>, ptr %x.var, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   %.loc14_48.6.b = getelementptr inbounds nuw <{ double, i32 }>, ptr %x.var, i32 0, i32 1, !dbg !43
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %x.var, ptr align 8 @struct.242.loc14_3.1, i64 12, i1 false), !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !45
// CHECK:STDOUT:   %.loc15_17.1.b = getelementptr inbounds nuw <{ double, i32 }>, ptr %x.var, i32 0, i32 1, !dbg !46
// CHECK:STDOUT:   %.loc15_17.2 = load i32, ptr %.loc15_17.1.b, align 4, !dbg !46
// CHECK:STDOUT:   store i32 %.loc15_17.2, ptr %y.var, align 4, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !48
// CHECK:STDOUT:   %.loc16_16 = load i32, ptr %y.var, align 4, !dbg !49
// CHECK:STDOUT:   store i32 %.loc16_16, ptr %_.var, align 4, !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %_.var), !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %y.var), !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.917afdc590dc83b9(ptr %x.var), !dbg !42
// CHECK:STDOUT:   ret i32 0, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %t) #0 !dbg !57 {
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.917afdc590dc83b9(ptr %t) #0 !dbg !64 {
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.dc5658e0ac74b2aa, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIFile(filename: "member_access.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{!36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !34, line: 13, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !42 = !DILocation(line: 14, column: 3, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 14, column: 31, scope: !39)
// CHECK:STDOUT: !45 = !DILocation(line: 15, column: 3, scope: !39)
// CHECK:STDOUT: !46 = !DILocation(line: 15, column: 16, scope: !39)
// CHECK:STDOUT: !48 = !DILocation(line: 16, column: 3, scope: !39)
// CHECK:STDOUT: !49 = !DILocation(line: 16, column: 16, scope: !39)
// CHECK:STDOUT: !51 = !{null, !36}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !53 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !54 = !{null, !53}
// CHECK:STDOUT: !55 = !DISubroutineType(types: !54)
// CHECK:STDOUT: !56 = !DILocation(line: 17, column: 3, scope: !39)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dc5658e0ac74b2aa", scope: null, file: !26, line: 30, type: !52, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !63)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !36)
// CHECK:STDOUT: !62 = !DILocation(line: 30, column: 1, scope: !57)
// CHECK:STDOUT: !63 = !{!58}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.917afdc590dc83b9", scope: null, file: !26, line: 30, type: !55, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !70)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !53)
// CHECK:STDOUT: !69 = !DILocation(line: 30, column: 1, scope: !64)
// CHECK:STDOUT: !70 = !{!65}
// CHECK:STDOUT:
