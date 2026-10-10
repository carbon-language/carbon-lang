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
// CHECK:STDOUT: @C.val.loc20_3 = internal constant {} zeroinitializer
// CHECK:STDOUT: @D.val.loc21_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !45
// CHECK:STDOUT:   %d.var = alloca {}, align 1, !dbg !47
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !49
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !45
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc20_3, i64 0, i1 false), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !47
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %d.var, ptr align 1 @D.val.loc21_3, i64 0, i1 false), !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !49
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !51
// CHECK:STDOUT:   store double 0.000000e+00, ptr %m.var, align 8, !dbg !51
// CHECK:STDOUT:   call void @_CF.Main.9522ff908e97e267(ptr %c.var), !dbg !54
// CHECK:STDOUT:   call void @_CF.Main.9522ff908e97e267(ptr %d.var), !dbg !56
// CHECK:STDOUT:   %.loc27 = load i32, ptr %n.var, align 4, !dbg !57
// CHECK:STDOUT:   call void @_CF.Main.d727653b20be66f9(i32 %.loc27), !dbg !58
// CHECK:STDOUT:   %.loc28 = load double, ptr %m.var, align 8, !dbg !59
// CHECK:STDOUT:   call void @_CF.Main.e0b820d0bf6546cf(double %.loc28), !dbg !60
// CHECK:STDOUT:   call void @_CF.Main.99ea8cc237f84c8d(%type.28 zeroinitializer), !dbg !61
// CHECK:STDOUT:   call void @"_CSelfDestruct.978c8bb64ae80ba3:core.Destroy.Core"(ptr %m.var), !dbg !51
// CHECK:STDOUT:   call void @"_CSelfDestruct.d19d32d3b3a297a1:core.Destroy.Core"(ptr %n.var), !dbg !49
// CHECK:STDOUT:   call void @"_CSelfDestruct.a50fd899f969e609:core.Destroy.Core"(ptr %d.var), !dbg !47
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.978c8bb64ae80ba3:core.Destroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d19d32d3b3a297a1:core.Destroy.Core"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.a50fd899f969e609:core.Destroy.Core"(ptr %self) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.9522ff908e97e267(ptr %_) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.d727653b20be66f9(i32 %_) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.e0b820d0bf6546cf(double %_) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.99ea8cc237f84c8d(%type.28 %_) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !106
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
// CHECK:STDOUT: !34 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{null}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{null, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{null, !36}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !34, line: 19, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !45 = !DILocation(line: 20, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !DILocation(line: 21, column: 3, scope: !44)
// CHECK:STDOUT: !49 = !DILocation(line: 22, column: 3, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 23, column: 3, scope: !44)
// CHECK:STDOUT: !54 = !DILocation(line: 25, column: 3, scope: !44)
// CHECK:STDOUT: !56 = !DILocation(line: 26, column: 3, scope: !44)
// CHECK:STDOUT: !57 = !DILocation(line: 27, column: 5, scope: !44)
// CHECK:STDOUT: !58 = !DILocation(line: 27, column: 3, scope: !44)
// CHECK:STDOUT: !59 = !DILocation(line: 28, column: 5, scope: !44)
// CHECK:STDOUT: !60 = !DILocation(line: 28, column: 3, scope: !44)
// CHECK:STDOUT: !61 = !DILocation(line: 29, column: 3, scope: !44)
// CHECK:STDOUT: !62 = !DILocation(line: 19, column: 1, scope: !44)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.978c8bb64ae80ba3:core.Destroy.Core", scope: null, file: !34, line: 23, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !39)
// CHECK:STDOUT: !65 = !DILocation(line: 23, column: 3, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d19d32d3b3a297a1:core.Destroy.Core", scope: null, file: !34, line: 22, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !36)
// CHECK:STDOUT: !69 = !DILocation(line: 22, column: 3, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !34, line: 21, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !39)
// CHECK:STDOUT: !73 = !DILocation(line: 21, column: 3, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.a50fd899f969e609:core.Destroy.Core", scope: null, file: !34, line: 21, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !78)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !39)
// CHECK:STDOUT: !77 = !DILocation(line: 21, column: 3, scope: !75)
// CHECK:STDOUT: !78 = !{!76}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !34, line: 20, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !82)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !39)
// CHECK:STDOUT: !81 = !DILocation(line: 20, column: 3, scope: !79)
// CHECK:STDOUT: !82 = !{!80}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9522ff908e97e267", scope: null, file: !34, line: 13, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !87)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !39)
// CHECK:STDOUT: !86 = !DILocation(line: 13, column: 1, scope: !83)
// CHECK:STDOUT: !87 = !{!84}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.d727653b20be66f9", scope: null, file: !34, line: 13, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !97)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !36)
// CHECK:STDOUT: !96 = !DILocation(line: 13, column: 1, scope: !93)
// CHECK:STDOUT: !97 = !{!94}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.e0b820d0bf6546cf", scope: null, file: !34, line: 13, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !102)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !39)
// CHECK:STDOUT: !101 = !DILocation(line: 13, column: 1, scope: !98)
// CHECK:STDOUT: !102 = !{!99}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.99ea8cc237f84c8d", scope: null, file: !34, line: 13, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !107)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !39)
// CHECK:STDOUT: !106 = !DILocation(line: 13, column: 1, scope: !103)
// CHECK:STDOUT: !107 = !{!104}
// CHECK:STDOUT:
