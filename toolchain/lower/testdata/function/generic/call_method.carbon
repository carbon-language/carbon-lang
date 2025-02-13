// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_method.carbon

class C {
  fn F[self: Self, T:! type](x: T) -> T {
    return x;
  }
}

fn CallF() -> i32 {
  var c: C = {};
  var n: i32 = 0;
  return c.F(n);
}

// CHECK:STDOUT: ; ModuleID = 'call_method.carbon'
// CHECK:STDOUT: source_filename = "call_method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc18_3.2 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CCallF.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !7
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 0, ptr %c.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc18_3.2, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %n.var), !dbg !7
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !8
// CHECK:STDOUT:   %.loc20_14 = load i32, ptr %n.var, align 4, !dbg !9
// CHECK:STDOUT:   %F.call = call i32 @_CF.C.Main.b88d1103f417c6d4(ptr %c.var, i32 %.loc20_14), !dbg !10
// CHECK:STDOUT:   ret i32 %F.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CF.C.Main.b88d1103f417c6d4(ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_method.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 19, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 20, column: 14, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 20, column: 10, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 20, column: 3, scope: !4)
