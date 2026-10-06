// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_specific_in_class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_specific_in_class.carbon

// Builds on `call_basic.carbon`. Checks different specific definitions are
// emitted for a generic function defined inside a class, when calling that
// generic function for an object, from a generic context.
// The specifics with a pointer type could be deduplicated here.
class C {
  fn Cfn[T: Core.Copy](unused self, x: T) -> T {
    return x;
  }
}

fn G[T: Core.Copy](x: T) -> T {
  var c: C;
  return c.Cfn(x);
}

fn F[T: Core.Copy](x: T) -> T {
  return G(x);
}

fn M() {
  var ptr_i32 : i32*;
  F(ptr_i32);
  var ptr_f64 : f64*;
  F(ptr_f64);
  var ptr_i8 : i8*;
  F(ptr_i8);
  var var_i32: i32 = 0;
  F(var_i32);
  var var_f64: f64 = 0.0;
  F(var_f64);
  var _: C;
  F(C);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_specific_in_class.carbon'
// CHECK:STDOUT: source_filename = "call_specific_in_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type.26 = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !51
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !57
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !61
// CHECK:STDOUT:   %var_i32.var = alloca i32, align 4, !dbg !65
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !71
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !75
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !51
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !51
// CHECK:STDOUT:   %.loc34_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !53
// CHECK:STDOUT:   %F.call.loc34 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc34_5), !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !57
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !57
// CHECK:STDOUT:   %.loc36_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !59
// CHECK:STDOUT:   %F.call.loc36 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc36_5), !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !61
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !61
// CHECK:STDOUT:   %.loc38_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !63
// CHECK:STDOUT:   %F.call.loc38 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc38_5), !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_i32.var), !dbg !65
// CHECK:STDOUT:   store i32 0, ptr %var_i32.var, align 4, !dbg !65
// CHECK:STDOUT:   %.loc40_5 = load i32, ptr %var_i32.var, align 4, !dbg !67
// CHECK:STDOUT:   %F.call.loc40 = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %.loc40_5), !dbg !68
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !71
// CHECK:STDOUT:   store double 0.000000e+00, ptr %var_f64.var, align 8, !dbg !71
// CHECK:STDOUT:   %.loc42_5 = load double, ptr %var_f64.var, align 8, !dbg !73
// CHECK:STDOUT:   %F.call.loc42 = call double @_CF.Main.c5e4ba542a56b71f(double %.loc42_5), !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !75
// CHECK:STDOUT:   %F.call.loc44 = call %type.26 @_CF.Main.c3359c2c217ad593(%type.26 zeroinitializer), !dbg !77
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %_.var), !dbg !75
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %var_f64.var), !dbg !71
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %var_i32.var), !dbg !65
// CHECK:STDOUT:   call void @"_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core"(ptr %ptr_i8.var), !dbg !61
// CHECK:STDOUT:   call void @"_CSelfDestruct.749b51755634ad17:core.Destroy.Core"(ptr %ptr_f64.var), !dbg !57
// CHECK:STDOUT:   call void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %ptr_i32.var), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core"(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.749b51755634ad17:core.Destroy.Core"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a76b78461416b4a8(ptr %x) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x), !dbg !105
// CHECK:STDOUT:   ret ptr %G.call, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x) #0 !dbg !126 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x), !dbg !132
// CHECK:STDOUT:   ret i32 %G.call, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.c5e4ba542a56b71f(double %x) #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x), !dbg !141
// CHECK:STDOUT:   ret double %G.call, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.26 @_CF.Main.c3359c2c217ad593(%type.26 %x) #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call %type.26 @_CG.Main.c3359c2c217ad593(%type.26 %x), !dbg !150
// CHECK:STDOUT:   ret %type.26 %G.call, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.a76b78461416b4a8(ptr %x) #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !158
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !158
// CHECK:STDOUT:   %C.Cfn.call = call ptr @_CCfn.C.Main.a76b78461416b4a8(ptr %c.var, ptr %x), !dbg !160
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !158
// CHECK:STDOUT:   ret ptr %C.Cfn.call, !dbg !164
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x) #0 !dbg !188 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !193
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !193
// CHECK:STDOUT:   %C.Cfn.call = call i32 @_CCfn.C.Main.4d6ceefeb7f772f4(ptr %c.var, i32 %x), !dbg !195
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !193
// CHECK:STDOUT:   ret i32 %C.Cfn.call, !dbg !199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.c5e4ba542a56b71f(double %x) #0 !dbg !201 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !206
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !206
// CHECK:STDOUT:   %C.Cfn.call = call double @_CCfn.C.Main.c5e4ba542a56b71f(ptr %c.var, double %x), !dbg !208
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !206
// CHECK:STDOUT:   ret double %C.Cfn.call, !dbg !210
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.26 @_CG.Main.c3359c2c217ad593(%type.26 %x) #0 !dbg !212 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !217
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !217
// CHECK:STDOUT:   %C.Cfn.call = call %type.26 @_CCfn.C.Main.c3359c2c217ad593(ptr %c.var, %type.26 %x), !dbg !219
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !217
// CHECK:STDOUT:   ret %type.26 %C.Cfn.call, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CCfn.C.Main.a76b78461416b4a8(ptr %self, ptr %x) #0 !dbg !223 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !231
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CCfn.C.Main.4d6ceefeb7f772f4(ptr %self, i32 %x) #0 !dbg !253 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CCfn.C.Main.c5e4ba542a56b71f(ptr %self, double %x) #0 !dbg !263 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !271
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.26 @_CCfn.C.Main.c3359c2c217ad593(ptr %self, %type.26 %x) #0 !dbg !273 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.26 %x, !dbg !281
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 9, 8, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.a76b78461416b4a8, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "call_specific_in_class.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{null, !35}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = !{null}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null, !34}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !32, line: 24, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !35)
// CHECK:STDOUT: !44 = !DILocation(line: 24, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !32, line: 24, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !49)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !35)
// CHECK:STDOUT: !48 = !DILocation(line: 24, column: 3, scope: !46)
// CHECK:STDOUT: !49 = !{!47}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !32, line: 32, type: !39, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !51 = !DILocation(line: 33, column: 3, scope: !50)
// CHECK:STDOUT: !53 = !DILocation(line: 34, column: 5, scope: !50)
// CHECK:STDOUT: !54 = !DILocation(line: 34, column: 3, scope: !50)
// CHECK:STDOUT: !55 = !{!35, !35}
// CHECK:STDOUT: !56 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !57 = !DILocation(line: 35, column: 3, scope: !50)
// CHECK:STDOUT: !59 = !DILocation(line: 36, column: 5, scope: !50)
// CHECK:STDOUT: !60 = !DILocation(line: 36, column: 3, scope: !50)
// CHECK:STDOUT: !61 = !DILocation(line: 37, column: 3, scope: !50)
// CHECK:STDOUT: !63 = !DILocation(line: 38, column: 5, scope: !50)
// CHECK:STDOUT: !64 = !DILocation(line: 38, column: 3, scope: !50)
// CHECK:STDOUT: !65 = !DILocation(line: 39, column: 3, scope: !50)
// CHECK:STDOUT: !67 = !DILocation(line: 40, column: 5, scope: !50)
// CHECK:STDOUT: !68 = !DILocation(line: 40, column: 3, scope: !50)
// CHECK:STDOUT: !69 = !{!34, !34}
// CHECK:STDOUT: !70 = !DISubroutineType(types: !69)
// CHECK:STDOUT: !71 = !DILocation(line: 41, column: 3, scope: !50)
// CHECK:STDOUT: !73 = !DILocation(line: 42, column: 5, scope: !50)
// CHECK:STDOUT: !74 = !DILocation(line: 42, column: 3, scope: !50)
// CHECK:STDOUT: !75 = !DILocation(line: 43, column: 3, scope: !50)
// CHECK:STDOUT: !77 = !DILocation(line: 44, column: 3, scope: !50)
// CHECK:STDOUT: !78 = !DILocation(line: 32, column: 1, scope: !50)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !32, line: 41, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !82)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !35)
// CHECK:STDOUT: !81 = !DILocation(line: 41, column: 3, scope: !79)
// CHECK:STDOUT: !82 = !{!80}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !32, line: 39, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !86)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !34)
// CHECK:STDOUT: !85 = !DILocation(line: 39, column: 3, scope: !83)
// CHECK:STDOUT: !86 = !{!84}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core", scope: null, file: !32, line: 37, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !35)
// CHECK:STDOUT: !89 = !DILocation(line: 37, column: 3, scope: !87)
// CHECK:STDOUT: !90 = !{!88}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.749b51755634ad17:core.Destroy.Core", scope: null, file: !32, line: 35, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !94)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !35)
// CHECK:STDOUT: !93 = !DILocation(line: 35, column: 3, scope: !91)
// CHECK:STDOUT: !94 = !{!92}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d088c7eb746488be:core.Destroy.Core", scope: null, file: !32, line: 33, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !35)
// CHECK:STDOUT: !97 = !DILocation(line: 33, column: 3, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a76b78461416b4a8", scope: null, file: !32, line: 28, type: !56, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !107)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !35)
// CHECK:STDOUT: !105 = !DILocation(line: 29, column: 10, scope: !99)
// CHECK:STDOUT: !106 = !DILocation(line: 29, column: 3, scope: !99)
// CHECK:STDOUT: !107 = !{!100}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 28, type: !70, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !134)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !34)
// CHECK:STDOUT: !132 = !DILocation(line: 29, column: 10, scope: !126)
// CHECK:STDOUT: !133 = !DILocation(line: 29, column: 3, scope: !126)
// CHECK:STDOUT: !134 = !{!127}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c5e4ba542a56b71f", scope: null, file: !32, line: 28, type: !56, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !143)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !35)
// CHECK:STDOUT: !141 = !DILocation(line: 29, column: 10, scope: !135)
// CHECK:STDOUT: !142 = !DILocation(line: 29, column: 3, scope: !135)
// CHECK:STDOUT: !143 = !{!136}
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c3359c2c217ad593", scope: null, file: !32, line: 28, type: !56, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !152)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !35)
// CHECK:STDOUT: !150 = !DILocation(line: 29, column: 10, scope: !144)
// CHECK:STDOUT: !151 = !DILocation(line: 29, column: 3, scope: !144)
// CHECK:STDOUT: !152 = !{!145}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.a76b78461416b4a8", scope: null, file: !32, line: 23, type: !56, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !165)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !35)
// CHECK:STDOUT: !158 = !DILocation(line: 24, column: 3, scope: !153)
// CHECK:STDOUT: !160 = !DILocation(line: 25, column: 10, scope: !153)
// CHECK:STDOUT: !162 = !{!35, !35, !35}
// CHECK:STDOUT: !163 = !DISubroutineType(types: !162)
// CHECK:STDOUT: !164 = !DILocation(line: 25, column: 3, scope: !153)
// CHECK:STDOUT: !165 = !{!154}
// CHECK:STDOUT: !188 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 23, type: !70, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !200)
// CHECK:STDOUT: !189 = !DILocalVariable(arg: 1, scope: !188, type: !34)
// CHECK:STDOUT: !193 = !DILocation(line: 24, column: 3, scope: !188)
// CHECK:STDOUT: !195 = !DILocation(line: 25, column: 10, scope: !188)
// CHECK:STDOUT: !197 = !{!34, !35, !34}
// CHECK:STDOUT: !198 = !DISubroutineType(types: !197)
// CHECK:STDOUT: !199 = !DILocation(line: 25, column: 3, scope: !188)
// CHECK:STDOUT: !200 = !{!189}
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c5e4ba542a56b71f", scope: null, file: !32, line: 23, type: !56, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !211)
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !201, type: !35)
// CHECK:STDOUT: !206 = !DILocation(line: 24, column: 3, scope: !201)
// CHECK:STDOUT: !208 = !DILocation(line: 25, column: 10, scope: !201)
// CHECK:STDOUT: !210 = !DILocation(line: 25, column: 3, scope: !201)
// CHECK:STDOUT: !211 = !{!202}
// CHECK:STDOUT: !212 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c3359c2c217ad593", scope: null, file: !32, line: 23, type: !56, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !222)
// CHECK:STDOUT: !213 = !DILocalVariable(arg: 1, scope: !212, type: !35)
// CHECK:STDOUT: !217 = !DILocation(line: 24, column: 3, scope: !212)
// CHECK:STDOUT: !219 = !DILocation(line: 25, column: 10, scope: !212)
// CHECK:STDOUT: !221 = !DILocation(line: 25, column: 3, scope: !212)
// CHECK:STDOUT: !222 = !{!213}
// CHECK:STDOUT: !223 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.a76b78461416b4a8", scope: null, file: !32, line: 18, type: !163, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !232)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 1, scope: !223, type: !35)
// CHECK:STDOUT: !225 = !DILocalVariable(arg: 2, scope: !223, type: !35)
// CHECK:STDOUT: !231 = !DILocation(line: 19, column: 5, scope: !223)
// CHECK:STDOUT: !232 = !{!224, !225}
// CHECK:STDOUT: !253 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 18, type: !198, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !262)
// CHECK:STDOUT: !254 = !DILocalVariable(arg: 1, scope: !253, type: !35)
// CHECK:STDOUT: !255 = !DILocalVariable(arg: 2, scope: !253, type: !34)
// CHECK:STDOUT: !261 = !DILocation(line: 19, column: 5, scope: !253)
// CHECK:STDOUT: !262 = !{!254, !255}
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.c5e4ba542a56b71f", scope: null, file: !32, line: 18, type: !163, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !272)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 1, scope: !263, type: !35)
// CHECK:STDOUT: !265 = !DILocalVariable(arg: 2, scope: !263, type: !35)
// CHECK:STDOUT: !271 = !DILocation(line: 19, column: 5, scope: !263)
// CHECK:STDOUT: !272 = !{!264, !265}
// CHECK:STDOUT: !273 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.c3359c2c217ad593", scope: null, file: !32, line: 18, type: !163, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !282)
// CHECK:STDOUT: !274 = !DILocalVariable(arg: 1, scope: !273, type: !35)
// CHECK:STDOUT: !275 = !DILocalVariable(arg: 2, scope: !273, type: !35)
// CHECK:STDOUT: !281 = !DILocation(line: 19, column: 5, scope: !273)
// CHECK:STDOUT: !282 = !{!274, !275}
// CHECK:STDOUT:
