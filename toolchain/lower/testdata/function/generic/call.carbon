// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call.carbon

fn F[T: type](_: T) {
}

class C {}
class D {}

fn G() {
  var c: C = {};
  var d: D = {};
  var n: i32 = 0;
  var m: f64 = 0.0;

  F(c);
  F(d);
  F(n);
  F(m);
  F(i32);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call.carbon'
// CHECK:STDOUT: source_filename = "call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type.28 = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc20_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT: @D.val.loc21_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !40
// CHECK:STDOUT:   %d.var = alloca {}, align 1, !dbg !42
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !44
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !40
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc20_3.1, i64 0, i1 false), !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !42
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %d.var, ptr align 1 @D.val.loc21_3.1, i64 0, i1 false), !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !44
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !46
// CHECK:STDOUT:   store double 0.000000e+00, ptr %m.var, align 8, !dbg !46
// CHECK:STDOUT:   call void @_CF.Main.9522ff908e97e267(ptr %c.var), !dbg !49
// CHECK:STDOUT:   call void @_CF.Main.9522ff908e97e267(ptr %d.var), !dbg !54
// CHECK:STDOUT:   %.loc27 = load i32, ptr %n.var, align 4, !dbg !55
// CHECK:STDOUT:   call void @_CF.Main.55456748a8a2eb70(i32 %.loc27), !dbg !56
// CHECK:STDOUT:   %.loc28 = load double, ptr %m.var, align 8, !dbg !59
// CHECK:STDOUT:   call void @_CF.Main.b709b3b9ccbae883(double %.loc28), !dbg !60
// CHECK:STDOUT:   call void @_CF.Main.99ea8cc237f84c8d(%type.28 zeroinitializer), !dbg !61
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %m.var), !dbg !46
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %n.var), !dbg !44
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0ee92c77788c7639(ptr %d.var), !dbg !42
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.356af3b6aef18564(ptr %c.var), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.9522ff908e97e267(ptr %_) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.55456748a8a2eb70(i32 %_) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.b709b3b9ccbae883(double %_) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.99ea8cc237f84c8d(%type.28 %_) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %t) #0 !dbg !88 {
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !95 {
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0ee92c77788c7639(ptr %t) #0 !dbg !102 {
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.356af3b6aef18564(ptr %t) #0 !dbg !109 {
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.9522ff908e97e267, { 1, 0 }
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
// CHECK:STDOUT: !34 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{null}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !34, line: 19, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !40 = !DILocation(line: 20, column: 3, scope: !39)
// CHECK:STDOUT: !42 = !DILocation(line: 21, column: 3, scope: !39)
// CHECK:STDOUT: !44 = !DILocation(line: 22, column: 3, scope: !39)
// CHECK:STDOUT: !46 = !DILocation(line: 23, column: 3, scope: !39)
// CHECK:STDOUT: !49 = !DILocation(line: 25, column: 3, scope: !39)
// CHECK:STDOUT: !50 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !51 = !{null, !50}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !54 = !DILocation(line: 26, column: 3, scope: !39)
// CHECK:STDOUT: !55 = !DILocation(line: 27, column: 5, scope: !39)
// CHECK:STDOUT: !56 = !DILocation(line: 27, column: 3, scope: !39)
// CHECK:STDOUT: !57 = !{null, !36}
// CHECK:STDOUT: !58 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 28, column: 5, scope: !39)
// CHECK:STDOUT: !60 = !DILocation(line: 28, column: 3, scope: !39)
// CHECK:STDOUT: !61 = !DILocation(line: 29, column: 3, scope: !39)
// CHECK:STDOUT: !62 = !DILocation(line: 19, column: 1, scope: !39)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9522ff908e97e267", scope: null, file: !34, line: 13, type: !52, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !67)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !50)
// CHECK:STDOUT: !66 = !DILocation(line: 13, column: 1, scope: !63)
// CHECK:STDOUT: !67 = !{!64}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.55456748a8a2eb70", scope: null, file: !34, line: 13, type: !58, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !77)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !36)
// CHECK:STDOUT: !76 = !DILocation(line: 13, column: 1, scope: !73)
// CHECK:STDOUT: !77 = !{!74}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b709b3b9ccbae883", scope: null, file: !34, line: 13, type: !52, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !82)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !50)
// CHECK:STDOUT: !81 = !DILocation(line: 13, column: 1, scope: !78)
// CHECK:STDOUT: !82 = !{!79}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.99ea8cc237f84c8d", scope: null, file: !34, line: 13, type: !52, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !87)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !50)
// CHECK:STDOUT: !86 = !DILocation(line: 13, column: 1, scope: !83)
// CHECK:STDOUT: !87 = !{!84}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2b38c2ca9f7f1a94", scope: null, file: !26, line: 32, type: !52, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !94)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !50)
// CHECK:STDOUT: !93 = !DILocation(line: 32, column: 1, scope: !88)
// CHECK:STDOUT: !94 = !{!89}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !26, line: 32, type: !58, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !101)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !36)
// CHECK:STDOUT: !100 = !DILocation(line: 32, column: 1, scope: !95)
// CHECK:STDOUT: !101 = !{!96}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0ee92c77788c7639", scope: null, file: !26, line: 32, type: !52, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !108)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !50)
// CHECK:STDOUT: !107 = !DILocation(line: 32, column: 1, scope: !102)
// CHECK:STDOUT: !108 = !{!103}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.356af3b6aef18564", scope: null, file: !26, line: 32, type: !52, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !115)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !50)
// CHECK:STDOUT: !114 = !DILocation(line: 32, column: 1, scope: !109)
// CHECK:STDOUT: !115 = !{!110}
// CHECK:STDOUT:
