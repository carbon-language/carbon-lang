// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_method.carbon

class C {
  fn F[T: Core.Copy](unused self, x: T) -> T {
    return x;
  }
}

fn CallF() -> i32 {
  var c: C = {};
  var n: i32 = 0;
  return c.F(n);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_method.carbon'
// CHECK:STDOUT: source_filename = "call_method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc20_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallF.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !26
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !26
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc20_3.1, i64 0, i1 false), !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !28
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !28
// CHECK:STDOUT:   %.loc22_14 = load i32, ptr %n.var, align 4, !dbg !31
// CHECK:STDOUT:   %C.F.call = call i32 @_CF.C.Main.2aef1d52a6e1f957(ptr %c.var, i32 %.loc22_14), !dbg !30
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %n.var), !dbg !28
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.166956a092acbab5(ptr %c.var), !dbg !26
// CHECK:STDOUT:   ret i32 %C.F.call, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.C.Main.2aef1d52a6e1f957(ptr %self, i32 %x) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !50 {
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.166956a092acbab5(ptr %t) #0 !dbg !57 {
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
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
// CHECK:STDOUT: !18 = !DIFile(filename: "call_method.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !18, line: 19, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 20, column: 3, scope: !23)
// CHECK:STDOUT: !28 = !DILocation(line: 21, column: 3, scope: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 22, column: 10, scope: !23)
// CHECK:STDOUT: !31 = !DILocation(line: 22, column: 14, scope: !23)
// CHECK:STDOUT: !32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !33 = !{!20, !32, !20}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !35 = !{null, !20}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !{null, !32}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 22, column: 3, scope: !23)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main.2aef1d52a6e1f957", scope: null, file: !18, line: 14, type: !34, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !49)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !32)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 2, scope: !40, type: !20)
// CHECK:STDOUT: !48 = !DILocation(line: 15, column: 5, scope: !40)
// CHECK:STDOUT: !49 = !{!41, !42}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !36, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !56)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !20)
// CHECK:STDOUT: !55 = !DILocation(line: 26, column: 1, scope: !50)
// CHECK:STDOUT: !56 = !{!51}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.166956a092acbab5", scope: null, file: !10, line: 26, type: !38, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !63)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !32)
// CHECK:STDOUT: !62 = !DILocation(line: 26, column: 1, scope: !57)
// CHECK:STDOUT: !63 = !{!58}
// CHECK:STDOUT:
