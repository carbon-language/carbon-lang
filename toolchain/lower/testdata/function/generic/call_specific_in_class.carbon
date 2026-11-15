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
// CHECK:STDOUT:   %F.call.loc34 = call ptr @_CF.Main.417c9c4a651ab31b(ptr %.loc34_5), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !47
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !47
// CHECK:STDOUT:   %.loc36_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !49
// CHECK:STDOUT:   %F.call.loc36 = call ptr @_CF.Main.417c9c4a651ab31b(ptr %.loc36_5), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !51
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !51
// CHECK:STDOUT:   %.loc38_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !53
// CHECK:STDOUT:   %F.call.loc38 = call ptr @_CF.Main.417c9c4a651ab31b(ptr %.loc38_5), !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_i32.var), !dbg !55
// CHECK:STDOUT:   store i32 0, ptr %var_i32.var, align 4, !dbg !55
// CHECK:STDOUT:   %.loc40_5 = load i32, ptr %var_i32.var, align 4, !dbg !57
// CHECK:STDOUT:   %F.call.loc40 = call i32 @_CF.Main.2fad9574e050607e(i32 %.loc40_5), !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !61
// CHECK:STDOUT:   store double 0.000000e+00, ptr %var_f64.var, align 8, !dbg !61
// CHECK:STDOUT:   %.loc42_5 = load double, ptr %var_f64.var, align 8, !dbg !63
// CHECK:STDOUT:   %F.call.loc42 = call double @_CF.Main.21aefb7dfdebb8e4(double %.loc42_5), !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !65
// CHECK:STDOUT:   %F.call.loc44 = call %type.28 @_CF.Main.c3359c2c217ad593(%type.28 zeroinitializer), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.417c9c4a651ab31b(ptr %x) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.417c9c4a651ab31b(ptr %x), !dbg !75
// CHECK:STDOUT:   ret ptr %G.call, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.2fad9574e050607e(i32 %x) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.2fad9574e050607e(i32 %x), !dbg !102
// CHECK:STDOUT:   ret i32 %G.call, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.21aefb7dfdebb8e4(double %x) #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.21aefb7dfdebb8e4(double %x), !dbg !111
// CHECK:STDOUT:   ret double %G.call, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.28 @_CF.Main.c3359c2c217ad593(%type.28 %x) #0 !dbg !114 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call %type.28 @_CG.Main.c3359c2c217ad593(%type.28 %x), !dbg !120
// CHECK:STDOUT:   ret %type.28 %G.call, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.417c9c4a651ab31b(ptr %x) #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !128
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !128
// CHECK:STDOUT:   %C.Cfn.call = call ptr @_CCfn.C.Main.417c9c4a651ab31b(ptr %c.var, ptr %x), !dbg !130
// CHECK:STDOUT:   ret ptr %C.Cfn.call, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.2fad9574e050607e(i32 %x) #0 !dbg !158 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !163
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !163
// CHECK:STDOUT:   %C.Cfn.call = call i32 @_CCfn.C.Main.2fad9574e050607e(ptr %c.var, i32 %x), !dbg !165
// CHECK:STDOUT:   ret i32 %C.Cfn.call, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.21aefb7dfdebb8e4(double %x) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !176
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !176
// CHECK:STDOUT:   %C.Cfn.call = call double @_CCfn.C.Main.21aefb7dfdebb8e4(ptr %c.var, double %x), !dbg !178
// CHECK:STDOUT:   ret double %C.Cfn.call, !dbg !180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.28 @_CG.Main.c3359c2c217ad593(%type.28 %x) #0 !dbg !182 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !187
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !187
// CHECK:STDOUT:   %C.Cfn.call = call %type.28 @_CCfn.C.Main.c3359c2c217ad593(ptr %c.var, %type.28 %x), !dbg !189
// CHECK:STDOUT:   ret %type.28 %C.Cfn.call, !dbg !191
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CCfn.C.Main.417c9c4a651ab31b(ptr %self, ptr %x) #0 !dbg !193 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CCfn.C.Main.2fad9574e050607e(ptr %self, i32 %x) #0 !dbg !223 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !231
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CCfn.C.Main.21aefb7dfdebb8e4(ptr %self, double %x) #0 !dbg !233 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !241
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.28 @_CCfn.C.Main.c3359c2c217ad593(ptr %self, %type.28 %x) #0 !dbg !243 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.28 %x, !dbg !251
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 9, 8, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.417c9c4a651ab31b, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!36}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !DIFile(filename: "call_specific_in_class.carbon", directory: "")
// CHECK:STDOUT: !36 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !35, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !37 = !{null}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !35, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !36)
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
// CHECK:STDOUT: !59 = !{!6, !6}
// CHECK:STDOUT: !60 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !61 = !DILocation(line: 41, column: 3, scope: !39)
// CHECK:STDOUT: !63 = !DILocation(line: 42, column: 5, scope: !39)
// CHECK:STDOUT: !64 = !DILocation(line: 42, column: 3, scope: !39)
// CHECK:STDOUT: !65 = !DILocation(line: 43, column: 3, scope: !39)
// CHECK:STDOUT: !67 = !DILocation(line: 44, column: 3, scope: !39)
// CHECK:STDOUT: !68 = !DILocation(line: 32, column: 1, scope: !39)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.417c9c4a651ab31b", scope: null, file: !35, line: 28, type: !46, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !77)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !44)
// CHECK:STDOUT: !75 = !DILocation(line: 29, column: 10, scope: !69)
// CHECK:STDOUT: !76 = !DILocation(line: 29, column: 3, scope: !69)
// CHECK:STDOUT: !77 = !{!70}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2fad9574e050607e", scope: null, file: !35, line: 28, type: !60, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !104)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !6)
// CHECK:STDOUT: !102 = !DILocation(line: 29, column: 10, scope: !96)
// CHECK:STDOUT: !103 = !DILocation(line: 29, column: 3, scope: !96)
// CHECK:STDOUT: !104 = !{!97}
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.21aefb7dfdebb8e4", scope: null, file: !35, line: 28, type: !46, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !113)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !44)
// CHECK:STDOUT: !111 = !DILocation(line: 29, column: 10, scope: !105)
// CHECK:STDOUT: !112 = !DILocation(line: 29, column: 3, scope: !105)
// CHECK:STDOUT: !113 = !{!106}
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c3359c2c217ad593", scope: null, file: !35, line: 28, type: !46, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !122)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !114, type: !44)
// CHECK:STDOUT: !120 = !DILocation(line: 29, column: 10, scope: !114)
// CHECK:STDOUT: !121 = !DILocation(line: 29, column: 3, scope: !114)
// CHECK:STDOUT: !122 = !{!115}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.417c9c4a651ab31b", scope: null, file: !35, line: 23, type: !46, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !135)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !44)
// CHECK:STDOUT: !128 = !DILocation(line: 24, column: 3, scope: !123)
// CHECK:STDOUT: !130 = !DILocation(line: 25, column: 10, scope: !123)
// CHECK:STDOUT: !132 = !{!44, !44, !44}
// CHECK:STDOUT: !133 = !DISubroutineType(types: !132)
// CHECK:STDOUT: !134 = !DILocation(line: 25, column: 3, scope: !123)
// CHECK:STDOUT: !135 = !{!124}
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.2fad9574e050607e", scope: null, file: !35, line: 23, type: !60, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !170)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 1, scope: !158, type: !6)
// CHECK:STDOUT: !163 = !DILocation(line: 24, column: 3, scope: !158)
// CHECK:STDOUT: !165 = !DILocation(line: 25, column: 10, scope: !158)
// CHECK:STDOUT: !167 = !{!6, !44, !6}
// CHECK:STDOUT: !168 = !DISubroutineType(types: !167)
// CHECK:STDOUT: !169 = !DILocation(line: 25, column: 3, scope: !158)
// CHECK:STDOUT: !170 = !{!159}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.21aefb7dfdebb8e4", scope: null, file: !35, line: 23, type: !46, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !181)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !44)
// CHECK:STDOUT: !176 = !DILocation(line: 24, column: 3, scope: !171)
// CHECK:STDOUT: !178 = !DILocation(line: 25, column: 10, scope: !171)
// CHECK:STDOUT: !180 = !DILocation(line: 25, column: 3, scope: !171)
// CHECK:STDOUT: !181 = !{!172}
// CHECK:STDOUT: !182 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c3359c2c217ad593", scope: null, file: !35, line: 23, type: !46, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !192)
// CHECK:STDOUT: !183 = !DILocalVariable(arg: 1, scope: !182, type: !44)
// CHECK:STDOUT: !187 = !DILocation(line: 24, column: 3, scope: !182)
// CHECK:STDOUT: !189 = !DILocation(line: 25, column: 10, scope: !182)
// CHECK:STDOUT: !191 = !DILocation(line: 25, column: 3, scope: !182)
// CHECK:STDOUT: !192 = !{!183}
// CHECK:STDOUT: !193 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.417c9c4a651ab31b", scope: null, file: !35, line: 18, type: !133, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !202)
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 1, scope: !193, type: !44)
// CHECK:STDOUT: !195 = !DILocalVariable(arg: 2, scope: !193, type: !44)
// CHECK:STDOUT: !201 = !DILocation(line: 19, column: 5, scope: !193)
// CHECK:STDOUT: !202 = !{!194, !195}
// CHECK:STDOUT: !223 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.2fad9574e050607e", scope: null, file: !35, line: 18, type: !168, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !232)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 1, scope: !223, type: !44)
// CHECK:STDOUT: !225 = !DILocalVariable(arg: 2, scope: !223, type: !6)
// CHECK:STDOUT: !231 = !DILocation(line: 19, column: 5, scope: !223)
// CHECK:STDOUT: !232 = !{!224, !225}
// CHECK:STDOUT: !233 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.21aefb7dfdebb8e4", scope: null, file: !35, line: 18, type: !133, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !242)
// CHECK:STDOUT: !234 = !DILocalVariable(arg: 1, scope: !233, type: !44)
// CHECK:STDOUT: !235 = !DILocalVariable(arg: 2, scope: !233, type: !44)
// CHECK:STDOUT: !241 = !DILocation(line: 19, column: 5, scope: !233)
// CHECK:STDOUT: !242 = !{!234, !235}
// CHECK:STDOUT: !243 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.c3359c2c217ad593", scope: null, file: !35, line: 18, type: !133, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !252)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 1, scope: !243, type: !44)
// CHECK:STDOUT: !245 = !DILocalVariable(arg: 2, scope: !243, type: !44)
// CHECK:STDOUT: !251 = !DILocation(line: 19, column: 5, scope: !243)
// CHECK:STDOUT: !252 = !{!244, !245}
// CHECK:STDOUT:
