// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_basic_depth.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_basic_depth.carbon

// Builds on `call_basic.carbon`. Checks definitions are emitted with deeper
// call stack, i.e., when functions can be type checked without looking at the
// specific calling context.
fn F[T: type](_: T) {
}

fn H[T: Core.Copy](x: T) -> T {
  F(x);
  return x;
}

fn G[T: Core.Copy & Core.Destroy](x: T) -> T {
  H(x);
  F(x);
  return x;
}

fn M() -> i32 {
  var n: i32 = 0;
  var m: i32;

  F(n);
  m = G(n);
  return m;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_basic_depth.carbon'
// CHECK:STDOUT: source_filename = "call_basic_depth.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CM.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !28
// CHECK:STDOUT:   %m.var = alloca i32, align 4, !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !28
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !30
// CHECK:STDOUT:   store i32 poison, ptr %m.var, align 4, !dbg !30
// CHECK:STDOUT:   %.loc34 = load i32, ptr %n.var, align 4, !dbg !32
// CHECK:STDOUT:   call void @_CF.Main.55456748a8a2eb70(i32 %.loc34), !dbg !33
// CHECK:STDOUT:   %.loc35_9 = load i32, ptr %n.var, align 4, !dbg !37
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.091764f9509e9823(i32 %.loc35_9), !dbg !38
// CHECK:STDOUT:   store i32 %G.call, ptr %m.var, align 4, !dbg !36
// CHECK:STDOUT:   %.loc36 = load i32, ptr %m.var, align 4, !dbg !41
// CHECK:STDOUT:   ret i32 %.loc36, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.55456748a8a2eb70(i32 %_) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.091764f9509e9823(i32 %x) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc25_6.3.temp = alloca i32, align 4, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_6.3.temp), !dbg !54
// CHECK:STDOUT:   %H.call = call i32 @_CH.Main.2fad9574e050607e(i32 %x), !dbg !54
// CHECK:STDOUT:   store i32 %H.call, ptr %.loc25_6.3.temp, align 4, !dbg !54
// CHECK:STDOUT:   call void @_CF.Main.55456748a8a2eb70(i32 %x), !dbg !56
// CHECK:STDOUT:   ret i32 %x, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CH.Main.2fad9574e050607e(i32 %x) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.55456748a8a2eb70(i32 %x), !dbg !66
// CHECK:STDOUT:   ret i32 %x, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "call_basic_depth.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !20, line: 30, type: !24, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 31, column: 3, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 32, column: 3, scope: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 34, column: 5, scope: !25)
// CHECK:STDOUT: !33 = !DILocation(line: 34, column: 3, scope: !25)
// CHECK:STDOUT: !34 = !{null, !22}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 35, column: 3, scope: !25)
// CHECK:STDOUT: !37 = !DILocation(line: 35, column: 9, scope: !25)
// CHECK:STDOUT: !38 = !DILocation(line: 35, column: 7, scope: !25)
// CHECK:STDOUT: !39 = !{!22, !22}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 36, column: 10, scope: !25)
// CHECK:STDOUT: !42 = !DILocation(line: 36, column: 3, scope: !25)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.55456748a8a2eb70", scope: null, file: !20, line: 16, type: !35, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !47)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !22)
// CHECK:STDOUT: !46 = !DILocation(line: 16, column: 1, scope: !43)
// CHECK:STDOUT: !47 = !{!44}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.091764f9509e9823", scope: null, file: !20, line: 24, type: !40, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !59)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !22)
// CHECK:STDOUT: !54 = !DILocation(line: 25, column: 3, scope: !48)
// CHECK:STDOUT: !56 = !DILocation(line: 26, column: 3, scope: !48)
// CHECK:STDOUT: !58 = !DILocation(line: 27, column: 3, scope: !48)
// CHECK:STDOUT: !59 = !{!49}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.2fad9574e050607e", scope: null, file: !20, line: 19, type: !40, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !69)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !22)
// CHECK:STDOUT: !66 = !DILocation(line: 20, column: 3, scope: !60)
// CHECK:STDOUT: !68 = !DILocation(line: 21, column: 3, scope: !60)
// CHECK:STDOUT: !69 = !{!61}
// CHECK:STDOUT:
