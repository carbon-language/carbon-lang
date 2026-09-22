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
// CHECK:STDOUT: %type.26 = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc20_3 = internal constant {} zeroinitializer
// CHECK:STDOUT: @D.val.loc21_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !43
// CHECK:STDOUT:   %d.var = alloca {}, align 1, !dbg !45
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !47
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !43
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc20_3, i64 0, i1 false), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !45
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %d.var, ptr align 1 @D.val.loc21_3, i64 0, i1 false), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !47
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !49
// CHECK:STDOUT:   store double 0.000000e+00, ptr %m.var, align 8, !dbg !49
// CHECK:STDOUT:   call void @_CF.Main.9522ff908e97e267(ptr %c.var), !dbg !52
// CHECK:STDOUT:   call void @_CF.Main.9522ff908e97e267(ptr %d.var), !dbg !54
// CHECK:STDOUT:   %.loc27 = load i32, ptr %n.var, align 4, !dbg !55
// CHECK:STDOUT:   call void @_CF.Main.84588f41d61dafba(i32 %.loc27), !dbg !56
// CHECK:STDOUT:   %.loc28 = load double, ptr %m.var, align 8, !dbg !57
// CHECK:STDOUT:   call void @_CF.Main.073fbce68be6103e(double %.loc28), !dbg !58
// CHECK:STDOUT:   call void @_CF.Main.99ea8cc237f84c8d(%type.26 zeroinitializer), !dbg !59
// CHECK:STDOUT:   call void @"_COp.dc14d4dd21801791:core.Destroy.Core"(ptr %m.var), !dbg !49
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %n.var), !dbg !47
// CHECK:STDOUT:   call void @"_COp.a50fd899f969e609:core.Destroy.Core"(ptr %d.var), !dbg !45
// CHECK:STDOUT:   call void @"_COp.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a50fd899f969e609:core.Destroy.Core"(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.9522ff908e97e267(ptr %_) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84588f41d61dafba(i32 %_) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.073fbce68be6103e(double %_) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.99ea8cc237f84c8d(%type.26 %_) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !100
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{null}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null, !34}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !32, line: 19, type: !36, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !43 = !DILocation(line: 20, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !DILocation(line: 21, column: 3, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 22, column: 3, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 23, column: 3, scope: !42)
// CHECK:STDOUT: !52 = !DILocation(line: 25, column: 3, scope: !42)
// CHECK:STDOUT: !54 = !DILocation(line: 26, column: 3, scope: !42)
// CHECK:STDOUT: !55 = !DILocation(line: 27, column: 5, scope: !42)
// CHECK:STDOUT: !56 = !DILocation(line: 27, column: 3, scope: !42)
// CHECK:STDOUT: !57 = !DILocation(line: 28, column: 5, scope: !42)
// CHECK:STDOUT: !58 = !DILocation(line: 28, column: 3, scope: !42)
// CHECK:STDOUT: !59 = !DILocation(line: 29, column: 3, scope: !42)
// CHECK:STDOUT: !60 = !DILocation(line: 19, column: 1, scope: !42)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Op", linkageName: "_COp.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !32, line: 23, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !64)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !37)
// CHECK:STDOUT: !63 = !DILocation(line: 23, column: 3, scope: !61)
// CHECK:STDOUT: !64 = !{!62}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f58524831c37807e:core.Destroy.Core", scope: null, file: !32, line: 22, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !34)
// CHECK:STDOUT: !67 = !DILocation(line: 22, column: 3, scope: !65)
// CHECK:STDOUT: !68 = !{!66}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a50fd899f969e609:core.Destroy.Core", scope: null, file: !32, line: 21, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !72)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !37)
// CHECK:STDOUT: !71 = !DILocation(line: 21, column: 3, scope: !69)
// CHECK:STDOUT: !72 = !{!70}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Op", linkageName: "_COp.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !32, line: 20, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !76)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !37)
// CHECK:STDOUT: !75 = !DILocation(line: 20, column: 3, scope: !73)
// CHECK:STDOUT: !76 = !{!74}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9522ff908e97e267", scope: null, file: !32, line: 13, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !81)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !37)
// CHECK:STDOUT: !80 = !DILocation(line: 13, column: 1, scope: !77)
// CHECK:STDOUT: !81 = !{!78}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84588f41d61dafba", scope: null, file: !32, line: 13, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !91)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !34)
// CHECK:STDOUT: !90 = !DILocation(line: 13, column: 1, scope: !87)
// CHECK:STDOUT: !91 = !{!88}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.073fbce68be6103e", scope: null, file: !32, line: 13, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !96)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !37)
// CHECK:STDOUT: !95 = !DILocation(line: 13, column: 1, scope: !92)
// CHECK:STDOUT: !96 = !{!93}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.99ea8cc237f84c8d", scope: null, file: !32, line: 13, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !101)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !37)
// CHECK:STDOUT: !100 = !DILocation(line: 13, column: 1, scope: !97)
// CHECK:STDOUT: !101 = !{!98}
// CHECK:STDOUT:
