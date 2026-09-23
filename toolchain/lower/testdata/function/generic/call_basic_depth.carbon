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
// CHECK:STDOUT:   call void @_CF.Main.84588f41d61dafba(i32 %.loc34), !dbg !33
// CHECK:STDOUT:   %.loc35_9 = load i32, ptr %n.var, align 4, !dbg !35
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d9f279dbc2b9f6f(i32 %.loc35_9), !dbg !36
// CHECK:STDOUT:   store i32 %G.call, ptr %m.var, align 4, !dbg !34
// CHECK:STDOUT:   %.loc36 = load i32, ptr %m.var, align 4, !dbg !39
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %m.var), !dbg !30
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %n.var), !dbg !28
// CHECK:STDOUT:   ret i32 %.loc36, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84588f41d61dafba(i32 %_) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.4d9f279dbc2b9f6f(i32 %x) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc25_6.3.temp = alloca i32, align 4, !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_6.3.temp), !dbg !64
// CHECK:STDOUT:   %H.call = call i32 @_CH.Main.4d6ceefeb7f772f4(i32 %x), !dbg !64
// CHECK:STDOUT:   store i32 %H.call, ptr %.loc25_6.3.temp, align 4, !dbg !64
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc25_6.3.temp), !dbg !64
// CHECK:STDOUT:   call void @_CF.Main.84588f41d61dafba(i32 %x), !dbg !66
// CHECK:STDOUT:   ret i32 %x, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CH.Main.4d6ceefeb7f772f4(i32 %x) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.84588f41d61dafba(i32 %x), !dbg !76
// CHECK:STDOUT:   ret i32 %x, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core", { 0, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "call_basic_depth.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null, !20}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !18, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !28 = !DILocation(line: 31, column: 3, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 32, column: 3, scope: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 34, column: 5, scope: !25)
// CHECK:STDOUT: !33 = !DILocation(line: 34, column: 3, scope: !25)
// CHECK:STDOUT: !34 = !DILocation(line: 35, column: 3, scope: !25)
// CHECK:STDOUT: !35 = !DILocation(line: 35, column: 9, scope: !25)
// CHECK:STDOUT: !36 = !DILocation(line: 35, column: 7, scope: !25)
// CHECK:STDOUT: !37 = !{!20, !20}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 36, column: 10, scope: !25)
// CHECK:STDOUT: !40 = !DILocation(line: 36, column: 3, scope: !25)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !18, line: 35, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !44)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !20)
// CHECK:STDOUT: !43 = !DILocation(line: 35, column: 7, scope: !41)
// CHECK:STDOUT: !44 = !{!42}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 35, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !20)
// CHECK:STDOUT: !47 = !DILocation(line: 35, column: 7, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 35, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !20)
// CHECK:STDOUT: !51 = !DILocation(line: 35, column: 7, scope: !49)
// CHECK:STDOUT: !52 = !{!50}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84588f41d61dafba", scope: null, file: !18, line: 16, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !57)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !20)
// CHECK:STDOUT: !56 = !DILocation(line: 16, column: 1, scope: !53)
// CHECK:STDOUT: !57 = !{!54}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4d9f279dbc2b9f6f", scope: null, file: !18, line: 24, type: !38, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !69)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !20)
// CHECK:STDOUT: !64 = !DILocation(line: 25, column: 3, scope: !58)
// CHECK:STDOUT: !66 = !DILocation(line: 26, column: 3, scope: !58)
// CHECK:STDOUT: !68 = !DILocation(line: 27, column: 3, scope: !58)
// CHECK:STDOUT: !69 = !{!59}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.4d6ceefeb7f772f4", scope: null, file: !18, line: 19, type: !38, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !79)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !20)
// CHECK:STDOUT: !76 = !DILocation(line: 20, column: 3, scope: !70)
// CHECK:STDOUT: !78 = !DILocation(line: 21, column: 3, scope: !70)
// CHECK:STDOUT: !79 = !{!71}
// CHECK:STDOUT:
