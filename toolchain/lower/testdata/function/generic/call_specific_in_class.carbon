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
// CHECK:STDOUT: %type.28 = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !53
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !59
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !63
// CHECK:STDOUT:   %var_i32.var = alloca i32, align 4, !dbg !67
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !73
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !77
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !53
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !53
// CHECK:STDOUT:   %.loc34_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !55
// CHECK:STDOUT:   %F.call.loc34 = call ptr @_CF.Main.01c402613389278e(ptr %.loc34_5), !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !59
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !59
// CHECK:STDOUT:   %.loc36_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !61
// CHECK:STDOUT:   %F.call.loc36 = call ptr @_CF.Main.01c402613389278e(ptr %.loc36_5), !dbg !62
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !63
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !63
// CHECK:STDOUT:   %.loc38_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !65
// CHECK:STDOUT:   %F.call.loc38 = call ptr @_CF.Main.01c402613389278e(ptr %.loc38_5), !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_i32.var), !dbg !67
// CHECK:STDOUT:   store i32 0, ptr %var_i32.var, align 4, !dbg !67
// CHECK:STDOUT:   %.loc40_5 = load i32, ptr %var_i32.var, align 4, !dbg !69
// CHECK:STDOUT:   %F.call.loc40 = call i32 @_CF.Main.3094c9a3a4d84642(i32 %.loc40_5), !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !73
// CHECK:STDOUT:   store double 0.000000e+00, ptr %var_f64.var, align 8, !dbg !73
// CHECK:STDOUT:   %.loc42_5 = load double, ptr %var_f64.var, align 8, !dbg !75
// CHECK:STDOUT:   %F.call.loc42 = call double @_CF.Main.d30a9b8c414024b6(double %.loc42_5), !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !77
// CHECK:STDOUT:   %F.call.loc44 = call %type.28 @_CF.Main.c3359c2c217ad593(%type.28 zeroinitializer), !dbg !79
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %_.var), !dbg !77
// CHECK:STDOUT:   call void @"_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core"(ptr %var_f64.var), !dbg !73
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %var_i32.var), !dbg !67
// CHECK:STDOUT:   call void @"_CSelfDestruct.64a2114a2cb70470:core.Destroy.Core"(ptr %ptr_i8.var), !dbg !63
// CHECK:STDOUT:   call void @"_CSelfDestruct.3a7a2e5beb0d0fa7:core.Destroy.Core"(ptr %ptr_f64.var), !dbg !59
// CHECK:STDOUT:   call void @"_CSelfDestruct.1feff81df232a20a:core.Destroy.Core"(ptr %ptr_i32.var), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.64a2114a2cb70470:core.Destroy.Core"(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3a7a2e5beb0d0fa7:core.Destroy.Core"(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.1feff81df232a20a:core.Destroy.Core"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.01c402613389278e(ptr %x) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.01c402613389278e(ptr %x), !dbg !107
// CHECK:STDOUT:   ret ptr %G.call, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.3094c9a3a4d84642(i32 %x) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.3094c9a3a4d84642(i32 %x), !dbg !134
// CHECK:STDOUT:   ret i32 %G.call, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.d30a9b8c414024b6(double %x) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.d30a9b8c414024b6(double %x), !dbg !143
// CHECK:STDOUT:   ret double %G.call, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.28 @_CF.Main.c3359c2c217ad593(%type.28 %x) #0 !dbg !146 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call %type.28 @_CG.Main.c3359c2c217ad593(%type.28 %x), !dbg !152
// CHECK:STDOUT:   ret %type.28 %G.call, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.01c402613389278e(ptr %x) #0 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !160
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !160
// CHECK:STDOUT:   %C.Cfn.call = call ptr @_CCfn.C.Main.01c402613389278e(ptr %c.var, ptr %x), !dbg !162
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !160
// CHECK:STDOUT:   ret ptr %C.Cfn.call, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.3094c9a3a4d84642(i32 %x) #0 !dbg !190 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !195
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !195
// CHECK:STDOUT:   %C.Cfn.call = call i32 @_CCfn.C.Main.3094c9a3a4d84642(ptr %c.var, i32 %x), !dbg !197
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !195
// CHECK:STDOUT:   ret i32 %C.Cfn.call, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.d30a9b8c414024b6(double %x) #0 !dbg !203 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !208
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !208
// CHECK:STDOUT:   %C.Cfn.call = call double @_CCfn.C.Main.d30a9b8c414024b6(ptr %c.var, double %x), !dbg !210
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !208
// CHECK:STDOUT:   ret double %C.Cfn.call, !dbg !212
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.28 @_CG.Main.c3359c2c217ad593(%type.28 %x) #0 !dbg !214 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !219
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !219
// CHECK:STDOUT:   %C.Cfn.call = call %type.28 @_CCfn.C.Main.c3359c2c217ad593(ptr %c.var, %type.28 %x), !dbg !221
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !219
// CHECK:STDOUT:   ret %type.28 %C.Cfn.call, !dbg !223
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CCfn.C.Main.01c402613389278e(ptr %self, ptr %x) #0 !dbg !225 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CCfn.C.Main.3094c9a3a4d84642(ptr %self, i32 %x) #0 !dbg !255 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !263
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CCfn.C.Main.d30a9b8c414024b6(ptr %self, double %x) #0 !dbg !265 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !273
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.28 @_CCfn.C.Main.c3359c2c217ad593(ptr %self, %type.28 %x) #0 !dbg !275 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.28 %x, !dbg !283
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 9, 8, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.01c402613389278e, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "call_specific_in_class.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{null, !36}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !34, line: 24, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !47)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !37)
// CHECK:STDOUT: !46 = !DILocation(line: 24, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !{!45}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !34, line: 24, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !51)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !37)
// CHECK:STDOUT: !50 = !DILocation(line: 24, column: 3, scope: !48)
// CHECK:STDOUT: !51 = !{!49}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !34, line: 32, type: !41, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !53 = !DILocation(line: 33, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !DILocation(line: 34, column: 5, scope: !52)
// CHECK:STDOUT: !56 = !DILocation(line: 34, column: 3, scope: !52)
// CHECK:STDOUT: !57 = !{!37, !37}
// CHECK:STDOUT: !58 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 35, column: 3, scope: !52)
// CHECK:STDOUT: !61 = !DILocation(line: 36, column: 5, scope: !52)
// CHECK:STDOUT: !62 = !DILocation(line: 36, column: 3, scope: !52)
// CHECK:STDOUT: !63 = !DILocation(line: 37, column: 3, scope: !52)
// CHECK:STDOUT: !65 = !DILocation(line: 38, column: 5, scope: !52)
// CHECK:STDOUT: !66 = !DILocation(line: 38, column: 3, scope: !52)
// CHECK:STDOUT: !67 = !DILocation(line: 39, column: 3, scope: !52)
// CHECK:STDOUT: !69 = !DILocation(line: 40, column: 5, scope: !52)
// CHECK:STDOUT: !70 = !DILocation(line: 40, column: 3, scope: !52)
// CHECK:STDOUT: !71 = !{!36, !36}
// CHECK:STDOUT: !72 = !DISubroutineType(types: !71)
// CHECK:STDOUT: !73 = !DILocation(line: 41, column: 3, scope: !52)
// CHECK:STDOUT: !75 = !DILocation(line: 42, column: 5, scope: !52)
// CHECK:STDOUT: !76 = !DILocation(line: 42, column: 3, scope: !52)
// CHECK:STDOUT: !77 = !DILocation(line: 43, column: 3, scope: !52)
// CHECK:STDOUT: !79 = !DILocation(line: 44, column: 3, scope: !52)
// CHECK:STDOUT: !80 = !DILocation(line: 32, column: 1, scope: !52)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core", scope: null, file: !34, line: 41, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !84)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !37)
// CHECK:STDOUT: !83 = !DILocation(line: 41, column: 3, scope: !81)
// CHECK:STDOUT: !84 = !{!82}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core", scope: null, file: !34, line: 39, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !36)
// CHECK:STDOUT: !87 = !DILocation(line: 39, column: 3, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.64a2114a2cb70470:core.Destroy.Core", scope: null, file: !34, line: 37, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !92)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !37)
// CHECK:STDOUT: !91 = !DILocation(line: 37, column: 3, scope: !89)
// CHECK:STDOUT: !92 = !{!90}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3a7a2e5beb0d0fa7:core.Destroy.Core", scope: null, file: !34, line: 35, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !37)
// CHECK:STDOUT: !95 = !DILocation(line: 35, column: 3, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.1feff81df232a20a:core.Destroy.Core", scope: null, file: !34, line: 33, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !37)
// CHECK:STDOUT: !99 = !DILocation(line: 33, column: 3, scope: !97)
// CHECK:STDOUT: !100 = !{!98}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.01c402613389278e", scope: null, file: !34, line: 28, type: !58, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !109)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !37)
// CHECK:STDOUT: !107 = !DILocation(line: 29, column: 10, scope: !101)
// CHECK:STDOUT: !108 = !DILocation(line: 29, column: 3, scope: !101)
// CHECK:STDOUT: !109 = !{!102}
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.3094c9a3a4d84642", scope: null, file: !34, line: 28, type: !72, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !136)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !36)
// CHECK:STDOUT: !134 = !DILocation(line: 29, column: 10, scope: !128)
// CHECK:STDOUT: !135 = !DILocation(line: 29, column: 3, scope: !128)
// CHECK:STDOUT: !136 = !{!129}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.d30a9b8c414024b6", scope: null, file: !34, line: 28, type: !58, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !145)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !37)
// CHECK:STDOUT: !143 = !DILocation(line: 29, column: 10, scope: !137)
// CHECK:STDOUT: !144 = !DILocation(line: 29, column: 3, scope: !137)
// CHECK:STDOUT: !145 = !{!138}
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c3359c2c217ad593", scope: null, file: !34, line: 28, type: !58, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !154)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !37)
// CHECK:STDOUT: !152 = !DILocation(line: 29, column: 10, scope: !146)
// CHECK:STDOUT: !153 = !DILocation(line: 29, column: 3, scope: !146)
// CHECK:STDOUT: !154 = !{!147}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.01c402613389278e", scope: null, file: !34, line: 23, type: !58, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !167)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !37)
// CHECK:STDOUT: !160 = !DILocation(line: 24, column: 3, scope: !155)
// CHECK:STDOUT: !162 = !DILocation(line: 25, column: 10, scope: !155)
// CHECK:STDOUT: !164 = !{!37, !37, !37}
// CHECK:STDOUT: !165 = !DISubroutineType(types: !164)
// CHECK:STDOUT: !166 = !DILocation(line: 25, column: 3, scope: !155)
// CHECK:STDOUT: !167 = !{!156}
// CHECK:STDOUT: !190 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.3094c9a3a4d84642", scope: null, file: !34, line: 23, type: !72, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !202)
// CHECK:STDOUT: !191 = !DILocalVariable(arg: 1, scope: !190, type: !36)
// CHECK:STDOUT: !195 = !DILocation(line: 24, column: 3, scope: !190)
// CHECK:STDOUT: !197 = !DILocation(line: 25, column: 10, scope: !190)
// CHECK:STDOUT: !199 = !{!36, !37, !36}
// CHECK:STDOUT: !200 = !DISubroutineType(types: !199)
// CHECK:STDOUT: !201 = !DILocation(line: 25, column: 3, scope: !190)
// CHECK:STDOUT: !202 = !{!191}
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.d30a9b8c414024b6", scope: null, file: !34, line: 23, type: !58, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !213)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !203, type: !37)
// CHECK:STDOUT: !208 = !DILocation(line: 24, column: 3, scope: !203)
// CHECK:STDOUT: !210 = !DILocation(line: 25, column: 10, scope: !203)
// CHECK:STDOUT: !212 = !DILocation(line: 25, column: 3, scope: !203)
// CHECK:STDOUT: !213 = !{!204}
// CHECK:STDOUT: !214 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c3359c2c217ad593", scope: null, file: !34, line: 23, type: !58, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !224)
// CHECK:STDOUT: !215 = !DILocalVariable(arg: 1, scope: !214, type: !37)
// CHECK:STDOUT: !219 = !DILocation(line: 24, column: 3, scope: !214)
// CHECK:STDOUT: !221 = !DILocation(line: 25, column: 10, scope: !214)
// CHECK:STDOUT: !223 = !DILocation(line: 25, column: 3, scope: !214)
// CHECK:STDOUT: !224 = !{!215}
// CHECK:STDOUT: !225 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.01c402613389278e", scope: null, file: !34, line: 18, type: !165, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !234)
// CHECK:STDOUT: !226 = !DILocalVariable(arg: 1, scope: !225, type: !37)
// CHECK:STDOUT: !227 = !DILocalVariable(arg: 2, scope: !225, type: !37)
// CHECK:STDOUT: !233 = !DILocation(line: 19, column: 5, scope: !225)
// CHECK:STDOUT: !234 = !{!226, !227}
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.3094c9a3a4d84642", scope: null, file: !34, line: 18, type: !200, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !264)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !37)
// CHECK:STDOUT: !257 = !DILocalVariable(arg: 2, scope: !255, type: !36)
// CHECK:STDOUT: !263 = !DILocation(line: 19, column: 5, scope: !255)
// CHECK:STDOUT: !264 = !{!256, !257}
// CHECK:STDOUT: !265 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.d30a9b8c414024b6", scope: null, file: !34, line: 18, type: !165, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !274)
// CHECK:STDOUT: !266 = !DILocalVariable(arg: 1, scope: !265, type: !37)
// CHECK:STDOUT: !267 = !DILocalVariable(arg: 2, scope: !265, type: !37)
// CHECK:STDOUT: !273 = !DILocation(line: 19, column: 5, scope: !265)
// CHECK:STDOUT: !274 = !{!266, !267}
// CHECK:STDOUT: !275 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.c3359c2c217ad593", scope: null, file: !34, line: 18, type: !165, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !284)
// CHECK:STDOUT: !276 = !DILocalVariable(arg: 1, scope: !275, type: !37)
// CHECK:STDOUT: !277 = !DILocalVariable(arg: 2, scope: !275, type: !37)
// CHECK:STDOUT: !283 = !DILocation(line: 19, column: 5, scope: !275)
// CHECK:STDOUT: !284 = !{!276, !277}
// CHECK:STDOUT:
