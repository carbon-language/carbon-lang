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
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !40
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !47
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !51
// CHECK:STDOUT:   %var_i32.var = alloca i32, align 4, !dbg !55
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !61
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !40
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !40
// CHECK:STDOUT:   %.loc34_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !42
// CHECK:STDOUT:   %F.call.loc34 = call ptr @_CF.Main.c0480b580588c73f(ptr %.loc34_5), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !47
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !47
// CHECK:STDOUT:   %.loc36_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !49
// CHECK:STDOUT:   %F.call.loc36 = call ptr @_CF.Main.c0480b580588c73f(ptr %.loc36_5), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !51
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !51
// CHECK:STDOUT:   %.loc38_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !53
// CHECK:STDOUT:   %F.call.loc38 = call ptr @_CF.Main.c0480b580588c73f(ptr %.loc38_5), !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_i32.var), !dbg !55
// CHECK:STDOUT:   store i32 0, ptr %var_i32.var, align 4, !dbg !55
// CHECK:STDOUT:   %.loc40_5 = load i32, ptr %var_i32.var, align 4, !dbg !57
// CHECK:STDOUT:   %F.call.loc40 = call i32 @_CF.Main.2aef1d52a6e1f957(i32 %.loc40_5), !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !61
// CHECK:STDOUT:   store double 0.000000e+00, ptr %var_f64.var, align 8, !dbg !61
// CHECK:STDOUT:   %.loc42_5 = load double, ptr %var_f64.var, align 8, !dbg !63
// CHECK:STDOUT:   %F.call.loc42 = call double @_CF.Main.cff5c31fb9f8fd40(double %.loc42_5), !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !65
// CHECK:STDOUT:   %F.call.loc44 = call %type.28 @_CF.Main.c3359c2c217ad593(%type.28 zeroinitializer), !dbg !67
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a4388806629063ea(ptr %_.var), !dbg !65
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.236ebccb4c02fafd(ptr %var_f64.var), !dbg !61
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %var_i32.var), !dbg !55
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.611e1d5593a8abfd(ptr %ptr_i8.var), !dbg !51
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.81b4e30e2d9fb9f0(ptr %ptr_f64.var), !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.39f93332b5bec0ba(ptr %ptr_i32.var), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.c0480b580588c73f(ptr %x) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.c0480b580588c73f(ptr %x), !dbg !79
// CHECK:STDOUT:   ret ptr %G.call, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.2aef1d52a6e1f957(i32 %x) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.2aef1d52a6e1f957(i32 %x), !dbg !106
// CHECK:STDOUT:   ret i32 %G.call, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.cff5c31fb9f8fd40(double %x) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.cff5c31fb9f8fd40(double %x), !dbg !115
// CHECK:STDOUT:   ret double %G.call, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.28 @_CF.Main.c3359c2c217ad593(%type.28 %x) #0 !dbg !118 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call %type.28 @_CG.Main.c3359c2c217ad593(%type.28 %x), !dbg !124
// CHECK:STDOUT:   ret %type.28 %G.call, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a4388806629063ea(ptr %t) #0 !dbg !127 {
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.236ebccb4c02fafd(ptr %t) #0 !dbg !134 {
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %t) #0 !dbg !141 {
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.611e1d5593a8abfd(ptr %t) #0 !dbg !148 {
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.81b4e30e2d9fb9f0(ptr %t) #0 !dbg !155 {
// CHECK:STDOUT:   ret void, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.39f93332b5bec0ba(ptr %t) #0 !dbg !162 {
// CHECK:STDOUT:   ret void, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.c0480b580588c73f(ptr %x) #0 !dbg !169 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !174
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !174
// CHECK:STDOUT:   %C.Cfn.call = call ptr @_CCfn.C.Main.c0480b580588c73f(ptr %c.var, ptr %x), !dbg !176
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a4388806629063ea(ptr %c.var), !dbg !174
// CHECK:STDOUT:   ret ptr %C.Cfn.call, !dbg !180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.2aef1d52a6e1f957(i32 %x) #0 !dbg !204 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !209
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !209
// CHECK:STDOUT:   %C.Cfn.call = call i32 @_CCfn.C.Main.2aef1d52a6e1f957(ptr %c.var, i32 %x), !dbg !211
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a4388806629063ea(ptr %c.var), !dbg !209
// CHECK:STDOUT:   ret i32 %C.Cfn.call, !dbg !215
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.cff5c31fb9f8fd40(double %x) #0 !dbg !217 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !222
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !222
// CHECK:STDOUT:   %C.Cfn.call = call double @_CCfn.C.Main.cff5c31fb9f8fd40(ptr %c.var, double %x), !dbg !224
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a4388806629063ea(ptr %c.var), !dbg !222
// CHECK:STDOUT:   ret double %C.Cfn.call, !dbg !226
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.28 @_CG.Main.c3359c2c217ad593(%type.28 %x) #0 !dbg !228 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !233
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !233
// CHECK:STDOUT:   %C.Cfn.call = call %type.28 @_CCfn.C.Main.c3359c2c217ad593(ptr %c.var, %type.28 %x), !dbg !235
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a4388806629063ea(ptr %c.var), !dbg !233
// CHECK:STDOUT:   ret %type.28 %C.Cfn.call, !dbg !237
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CCfn.C.Main.c0480b580588c73f(ptr %self, ptr %x) #0 !dbg !239 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !247
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CCfn.C.Main.2aef1d52a6e1f957(ptr %self, i32 %x) #0 !dbg !269 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !277
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CCfn.C.Main.cff5c31fb9f8fd40(ptr %self, double %x) #0 !dbg !279 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !287
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.28 @_CCfn.C.Main.c3359c2c217ad593(ptr %self, %type.28 %x) #0 !dbg !289 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.28 %x, !dbg !297
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 9, 8, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.c0480b580588c73f, { 2, 1, 0 }
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
// CHECK:STDOUT: !34 = !DIFile(filename: "call_specific_in_class.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{null}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !34, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !40 = !DILocation(line: 33, column: 3, scope: !39)
// CHECK:STDOUT: !42 = !DILocation(line: 34, column: 5, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 34, column: 3, scope: !39)
// CHECK:STDOUT: !44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !45 = !{!44, !44}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 35, column: 3, scope: !39)
// CHECK:STDOUT: !49 = !DILocation(line: 36, column: 5, scope: !39)
// CHECK:STDOUT: !50 = !DILocation(line: 36, column: 3, scope: !39)
// CHECK:STDOUT: !51 = !DILocation(line: 37, column: 3, scope: !39)
// CHECK:STDOUT: !53 = !DILocation(line: 38, column: 5, scope: !39)
// CHECK:STDOUT: !54 = !DILocation(line: 38, column: 3, scope: !39)
// CHECK:STDOUT: !55 = !DILocation(line: 39, column: 3, scope: !39)
// CHECK:STDOUT: !57 = !DILocation(line: 40, column: 5, scope: !39)
// CHECK:STDOUT: !58 = !DILocation(line: 40, column: 3, scope: !39)
// CHECK:STDOUT: !59 = !{!36, !36}
// CHECK:STDOUT: !60 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !61 = !DILocation(line: 41, column: 3, scope: !39)
// CHECK:STDOUT: !63 = !DILocation(line: 42, column: 5, scope: !39)
// CHECK:STDOUT: !64 = !DILocation(line: 42, column: 3, scope: !39)
// CHECK:STDOUT: !65 = !DILocation(line: 43, column: 3, scope: !39)
// CHECK:STDOUT: !67 = !DILocation(line: 44, column: 3, scope: !39)
// CHECK:STDOUT: !68 = !{null, !44}
// CHECK:STDOUT: !69 = !DISubroutineType(types: !68)
// CHECK:STDOUT: !70 = !{null, !36}
// CHECK:STDOUT: !71 = !DISubroutineType(types: !70)
// CHECK:STDOUT: !72 = !DILocation(line: 32, column: 1, scope: !39)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c0480b580588c73f", scope: null, file: !34, line: 28, type: !46, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !81)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !44)
// CHECK:STDOUT: !79 = !DILocation(line: 29, column: 10, scope: !73)
// CHECK:STDOUT: !80 = !DILocation(line: 29, column: 3, scope: !73)
// CHECK:STDOUT: !81 = !{!74}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2aef1d52a6e1f957", scope: null, file: !34, line: 28, type: !60, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !108)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !36)
// CHECK:STDOUT: !106 = !DILocation(line: 29, column: 10, scope: !100)
// CHECK:STDOUT: !107 = !DILocation(line: 29, column: 3, scope: !100)
// CHECK:STDOUT: !108 = !{!101}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.cff5c31fb9f8fd40", scope: null, file: !34, line: 28, type: !46, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !117)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !44)
// CHECK:STDOUT: !115 = !DILocation(line: 29, column: 10, scope: !109)
// CHECK:STDOUT: !116 = !DILocation(line: 29, column: 3, scope: !109)
// CHECK:STDOUT: !117 = !{!110}
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c3359c2c217ad593", scope: null, file: !34, line: 28, type: !46, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !126)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !44)
// CHECK:STDOUT: !124 = !DILocation(line: 29, column: 10, scope: !118)
// CHECK:STDOUT: !125 = !DILocation(line: 29, column: 3, scope: !118)
// CHECK:STDOUT: !126 = !{!119}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a4388806629063ea", scope: null, file: !26, line: 30, type: !69, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !133)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !44)
// CHECK:STDOUT: !132 = !DILocation(line: 30, column: 1, scope: !127)
// CHECK:STDOUT: !133 = !{!128}
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.236ebccb4c02fafd", scope: null, file: !26, line: 30, type: !69, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !140)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !44)
// CHECK:STDOUT: !139 = !DILocation(line: 30, column: 1, scope: !134)
// CHECK:STDOUT: !140 = !{!135}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7f8dce37f94a83bf", scope: null, file: !26, line: 30, type: !71, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !147)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !36)
// CHECK:STDOUT: !146 = !DILocation(line: 30, column: 1, scope: !141)
// CHECK:STDOUT: !147 = !{!142}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.611e1d5593a8abfd", scope: null, file: !26, line: 30, type: !69, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !154)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !44)
// CHECK:STDOUT: !153 = !DILocation(line: 30, column: 1, scope: !148)
// CHECK:STDOUT: !154 = !{!149}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.81b4e30e2d9fb9f0", scope: null, file: !26, line: 30, type: !69, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !161)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !44)
// CHECK:STDOUT: !160 = !DILocation(line: 30, column: 1, scope: !155)
// CHECK:STDOUT: !161 = !{!156}
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.39f93332b5bec0ba", scope: null, file: !26, line: 30, type: !69, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !168)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !162, type: !44)
// CHECK:STDOUT: !167 = !DILocation(line: 30, column: 1, scope: !162)
// CHECK:STDOUT: !168 = !{!163}
// CHECK:STDOUT: !169 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c0480b580588c73f", scope: null, file: !34, line: 23, type: !46, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !181)
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 1, scope: !169, type: !44)
// CHECK:STDOUT: !174 = !DILocation(line: 24, column: 3, scope: !169)
// CHECK:STDOUT: !176 = !DILocation(line: 25, column: 10, scope: !169)
// CHECK:STDOUT: !178 = !{!44, !44, !44}
// CHECK:STDOUT: !179 = !DISubroutineType(types: !178)
// CHECK:STDOUT: !180 = !DILocation(line: 25, column: 3, scope: !169)
// CHECK:STDOUT: !181 = !{!170}
// CHECK:STDOUT: !204 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.2aef1d52a6e1f957", scope: null, file: !34, line: 23, type: !60, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !216)
// CHECK:STDOUT: !205 = !DILocalVariable(arg: 1, scope: !204, type: !36)
// CHECK:STDOUT: !209 = !DILocation(line: 24, column: 3, scope: !204)
// CHECK:STDOUT: !211 = !DILocation(line: 25, column: 10, scope: !204)
// CHECK:STDOUT: !213 = !{!36, !44, !36}
// CHECK:STDOUT: !214 = !DISubroutineType(types: !213)
// CHECK:STDOUT: !215 = !DILocation(line: 25, column: 3, scope: !204)
// CHECK:STDOUT: !216 = !{!205}
// CHECK:STDOUT: !217 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.cff5c31fb9f8fd40", scope: null, file: !34, line: 23, type: !46, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !227)
// CHECK:STDOUT: !218 = !DILocalVariable(arg: 1, scope: !217, type: !44)
// CHECK:STDOUT: !222 = !DILocation(line: 24, column: 3, scope: !217)
// CHECK:STDOUT: !224 = !DILocation(line: 25, column: 10, scope: !217)
// CHECK:STDOUT: !226 = !DILocation(line: 25, column: 3, scope: !217)
// CHECK:STDOUT: !227 = !{!218}
// CHECK:STDOUT: !228 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c3359c2c217ad593", scope: null, file: !34, line: 23, type: !46, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !238)
// CHECK:STDOUT: !229 = !DILocalVariable(arg: 1, scope: !228, type: !44)
// CHECK:STDOUT: !233 = !DILocation(line: 24, column: 3, scope: !228)
// CHECK:STDOUT: !235 = !DILocation(line: 25, column: 10, scope: !228)
// CHECK:STDOUT: !237 = !DILocation(line: 25, column: 3, scope: !228)
// CHECK:STDOUT: !238 = !{!229}
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.c0480b580588c73f", scope: null, file: !34, line: 18, type: !179, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !248)
// CHECK:STDOUT: !240 = !DILocalVariable(arg: 1, scope: !239, type: !44)
// CHECK:STDOUT: !241 = !DILocalVariable(arg: 2, scope: !239, type: !44)
// CHECK:STDOUT: !247 = !DILocation(line: 19, column: 5, scope: !239)
// CHECK:STDOUT: !248 = !{!240, !241}
// CHECK:STDOUT: !269 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.2aef1d52a6e1f957", scope: null, file: !34, line: 18, type: !214, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !278)
// CHECK:STDOUT: !270 = !DILocalVariable(arg: 1, scope: !269, type: !44)
// CHECK:STDOUT: !271 = !DILocalVariable(arg: 2, scope: !269, type: !36)
// CHECK:STDOUT: !277 = !DILocation(line: 19, column: 5, scope: !269)
// CHECK:STDOUT: !278 = !{!270, !271}
// CHECK:STDOUT: !279 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.cff5c31fb9f8fd40", scope: null, file: !34, line: 18, type: !179, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !288)
// CHECK:STDOUT: !280 = !DILocalVariable(arg: 1, scope: !279, type: !44)
// CHECK:STDOUT: !281 = !DILocalVariable(arg: 2, scope: !279, type: !44)
// CHECK:STDOUT: !287 = !DILocation(line: 19, column: 5, scope: !279)
// CHECK:STDOUT: !288 = !{!280, !281}
// CHECK:STDOUT: !289 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.c3359c2c217ad593", scope: null, file: !34, line: 18, type: !179, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !298)
// CHECK:STDOUT: !290 = !DILocalVariable(arg: 1, scope: !289, type: !44)
// CHECK:STDOUT: !291 = !DILocalVariable(arg: 2, scope: !289, type: !44)
// CHECK:STDOUT: !297 = !DILocation(line: 19, column: 5, scope: !289)
// CHECK:STDOUT: !298 = !{!290, !291}
// CHECK:STDOUT:
