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
// CHECK:STDOUT: @C.val.loc20_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallF.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !28
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !28
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc20_3, i64 0, i1 false), !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !30
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !30
// CHECK:STDOUT:   %.loc22_14 = load i32, ptr %n.var, align 4, !dbg !33
// CHECK:STDOUT:   %C.F.call = call i32 @_CF.C.Main.2fad9574e050607e(ptr %c.var, i32 %.loc22_14), !dbg !32
// CHECK:STDOUT:   ret i32 %C.F.call, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.C.Main.2fad9574e050607e(ptr %self, i32 %x) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "call_method.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !20, line: 19, type: !24, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 20, column: 3, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 21, column: 3, scope: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 22, column: 10, scope: !25)
// CHECK:STDOUT: !33 = !DILocation(line: 22, column: 14, scope: !25)
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{!22, !34, !22}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !DILocation(line: 22, column: 3, scope: !25)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main.2fad9574e050607e", scope: null, file: !20, line: 14, type: !36, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !47)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !34)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 2, scope: !38, type: !22)
// CHECK:STDOUT: !46 = !DILocation(line: 15, column: 5, scope: !38)
// CHECK:STDOUT: !47 = !{!39, !40}
// CHECK:STDOUT:
