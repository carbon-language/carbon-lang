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
// CHECK:STDOUT: define weak_odr void @"_COp.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !47
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !53
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !57
// CHECK:STDOUT:   %var_i32.var = alloca i32, align 4, !dbg !61
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !67
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !71
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !47
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !47
// CHECK:STDOUT:   %.loc34_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !49
// CHECK:STDOUT:   %F.call.loc34 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc34_5), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !53
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !53
// CHECK:STDOUT:   %.loc36_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !55
// CHECK:STDOUT:   %F.call.loc36 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc36_5), !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !57
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !57
// CHECK:STDOUT:   %.loc38_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !59
// CHECK:STDOUT:   %F.call.loc38 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc38_5), !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_i32.var), !dbg !61
// CHECK:STDOUT:   store i32 0, ptr %var_i32.var, align 4, !dbg !61
// CHECK:STDOUT:   %.loc40_5 = load i32, ptr %var_i32.var, align 4, !dbg !63
// CHECK:STDOUT:   %F.call.loc40 = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %.loc40_5), !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !67
// CHECK:STDOUT:   store double 0.000000e+00, ptr %var_f64.var, align 8, !dbg !67
// CHECK:STDOUT:   %.loc42_5 = load double, ptr %var_f64.var, align 8, !dbg !69
// CHECK:STDOUT:   %F.call.loc42 = call double @_CF.Main.c5e4ba542a56b71f(double %.loc42_5), !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !71
// CHECK:STDOUT:   %F.call.loc44 = call %type.26 @_CF.Main.c3359c2c217ad593(%type.26 zeroinitializer), !dbg !73
// CHECK:STDOUT:   call void @"_COp.08d9e38b928ba575:core.Destroy.Core"(ptr %_.var), !dbg !71
// CHECK:STDOUT:   call void @"_COp.dc14d4dd21801791:core.Destroy.Core"(ptr %var_f64.var), !dbg !67
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %var_i32.var), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a76b78461416b4a8(ptr %x) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x), !dbg !89
// CHECK:STDOUT:   ret ptr %G.call, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x), !dbg !116
// CHECK:STDOUT:   ret i32 %G.call, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.c5e4ba542a56b71f(double %x) #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x), !dbg !125
// CHECK:STDOUT:   ret double %G.call, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.26 @_CF.Main.c3359c2c217ad593(%type.26 %x) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call %type.26 @_CG.Main.c3359c2c217ad593(%type.26 %x), !dbg !134
// CHECK:STDOUT:   ret %type.26 %G.call, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.a76b78461416b4a8(ptr %x) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !142
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !142
// CHECK:STDOUT:   %C.Cfn.call = call ptr @_CCfn.C.Main.a76b78461416b4a8(ptr %c.var, ptr %x), !dbg !144
// CHECK:STDOUT:   call void @"_COp.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !142
// CHECK:STDOUT:   ret ptr %C.Cfn.call, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x) #0 !dbg !172 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !177
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !177
// CHECK:STDOUT:   %C.Cfn.call = call i32 @_CCfn.C.Main.4d6ceefeb7f772f4(ptr %c.var, i32 %x), !dbg !179
// CHECK:STDOUT:   call void @"_COp.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !177
// CHECK:STDOUT:   ret i32 %C.Cfn.call, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.c5e4ba542a56b71f(double %x) #0 !dbg !185 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !190
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !190
// CHECK:STDOUT:   %C.Cfn.call = call double @_CCfn.C.Main.c5e4ba542a56b71f(ptr %c.var, double %x), !dbg !192
// CHECK:STDOUT:   call void @"_COp.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !190
// CHECK:STDOUT:   ret double %C.Cfn.call, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.26 @_CG.Main.c3359c2c217ad593(%type.26 %x) #0 !dbg !196 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !201
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !201
// CHECK:STDOUT:   %C.Cfn.call = call %type.26 @_CCfn.C.Main.c3359c2c217ad593(ptr %c.var, %type.26 %x), !dbg !203
// CHECK:STDOUT:   call void @"_COp.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !201
// CHECK:STDOUT:   ret %type.26 %C.Cfn.call, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CCfn.C.Main.a76b78461416b4a8(ptr %self, ptr %x) #0 !dbg !207 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !215
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CCfn.C.Main.4d6ceefeb7f772f4(ptr %self, i32 %x) #0 !dbg !237 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !245
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CCfn.C.Main.c5e4ba542a56b71f(ptr %self, double %x) #0 !dbg !247 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !255
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.26 @_CCfn.C.Main.c3359c2c217ad593(ptr %self, %type.26 %x) #0 !dbg !257 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.26 %x, !dbg !265
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
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Op", linkageName: "_COp.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !32, line: 24, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !35)
// CHECK:STDOUT: !44 = !DILocation(line: 24, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !32, line: 32, type: !39, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !47 = !DILocation(line: 33, column: 3, scope: !46)
// CHECK:STDOUT: !49 = !DILocation(line: 34, column: 5, scope: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 34, column: 3, scope: !46)
// CHECK:STDOUT: !51 = !{!35, !35}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !53 = !DILocation(line: 35, column: 3, scope: !46)
// CHECK:STDOUT: !55 = !DILocation(line: 36, column: 5, scope: !46)
// CHECK:STDOUT: !56 = !DILocation(line: 36, column: 3, scope: !46)
// CHECK:STDOUT: !57 = !DILocation(line: 37, column: 3, scope: !46)
// CHECK:STDOUT: !59 = !DILocation(line: 38, column: 5, scope: !46)
// CHECK:STDOUT: !60 = !DILocation(line: 38, column: 3, scope: !46)
// CHECK:STDOUT: !61 = !DILocation(line: 39, column: 3, scope: !46)
// CHECK:STDOUT: !63 = !DILocation(line: 40, column: 5, scope: !46)
// CHECK:STDOUT: !64 = !DILocation(line: 40, column: 3, scope: !46)
// CHECK:STDOUT: !65 = !{!34, !34}
// CHECK:STDOUT: !66 = !DISubroutineType(types: !65)
// CHECK:STDOUT: !67 = !DILocation(line: 41, column: 3, scope: !46)
// CHECK:STDOUT: !69 = !DILocation(line: 42, column: 5, scope: !46)
// CHECK:STDOUT: !70 = !DILocation(line: 42, column: 3, scope: !46)
// CHECK:STDOUT: !71 = !DILocation(line: 43, column: 3, scope: !46)
// CHECK:STDOUT: !73 = !DILocation(line: 44, column: 3, scope: !46)
// CHECK:STDOUT: !74 = !DILocation(line: 32, column: 1, scope: !46)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Op", linkageName: "_COp.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !32, line: 41, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !78)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !35)
// CHECK:STDOUT: !77 = !DILocation(line: 41, column: 3, scope: !75)
// CHECK:STDOUT: !78 = !{!76}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f58524831c37807e:core.Destroy.Core", scope: null, file: !32, line: 39, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !82)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !34)
// CHECK:STDOUT: !81 = !DILocation(line: 39, column: 3, scope: !79)
// CHECK:STDOUT: !82 = !{!80}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a76b78461416b4a8", scope: null, file: !32, line: 28, type: !52, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !91)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !35)
// CHECK:STDOUT: !89 = !DILocation(line: 29, column: 10, scope: !83)
// CHECK:STDOUT: !90 = !DILocation(line: 29, column: 3, scope: !83)
// CHECK:STDOUT: !91 = !{!84}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 28, type: !66, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !118)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !34)
// CHECK:STDOUT: !116 = !DILocation(line: 29, column: 10, scope: !110)
// CHECK:STDOUT: !117 = !DILocation(line: 29, column: 3, scope: !110)
// CHECK:STDOUT: !118 = !{!111}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c5e4ba542a56b71f", scope: null, file: !32, line: 28, type: !52, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !127)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !35)
// CHECK:STDOUT: !125 = !DILocation(line: 29, column: 10, scope: !119)
// CHECK:STDOUT: !126 = !DILocation(line: 29, column: 3, scope: !119)
// CHECK:STDOUT: !127 = !{!120}
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c3359c2c217ad593", scope: null, file: !32, line: 28, type: !52, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !136)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !35)
// CHECK:STDOUT: !134 = !DILocation(line: 29, column: 10, scope: !128)
// CHECK:STDOUT: !135 = !DILocation(line: 29, column: 3, scope: !128)
// CHECK:STDOUT: !136 = !{!129}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.a76b78461416b4a8", scope: null, file: !32, line: 23, type: !52, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !149)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !35)
// CHECK:STDOUT: !142 = !DILocation(line: 24, column: 3, scope: !137)
// CHECK:STDOUT: !144 = !DILocation(line: 25, column: 10, scope: !137)
// CHECK:STDOUT: !146 = !{!35, !35, !35}
// CHECK:STDOUT: !147 = !DISubroutineType(types: !146)
// CHECK:STDOUT: !148 = !DILocation(line: 25, column: 3, scope: !137)
// CHECK:STDOUT: !149 = !{!138}
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 23, type: !66, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !184)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 1, scope: !172, type: !34)
// CHECK:STDOUT: !177 = !DILocation(line: 24, column: 3, scope: !172)
// CHECK:STDOUT: !179 = !DILocation(line: 25, column: 10, scope: !172)
// CHECK:STDOUT: !181 = !{!34, !35, !34}
// CHECK:STDOUT: !182 = !DISubroutineType(types: !181)
// CHECK:STDOUT: !183 = !DILocation(line: 25, column: 3, scope: !172)
// CHECK:STDOUT: !184 = !{!173}
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c5e4ba542a56b71f", scope: null, file: !32, line: 23, type: !52, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !195)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 1, scope: !185, type: !35)
// CHECK:STDOUT: !190 = !DILocation(line: 24, column: 3, scope: !185)
// CHECK:STDOUT: !192 = !DILocation(line: 25, column: 10, scope: !185)
// CHECK:STDOUT: !194 = !DILocation(line: 25, column: 3, scope: !185)
// CHECK:STDOUT: !195 = !{!186}
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c3359c2c217ad593", scope: null, file: !32, line: 23, type: !52, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !206)
// CHECK:STDOUT: !197 = !DILocalVariable(arg: 1, scope: !196, type: !35)
// CHECK:STDOUT: !201 = !DILocation(line: 24, column: 3, scope: !196)
// CHECK:STDOUT: !203 = !DILocation(line: 25, column: 10, scope: !196)
// CHECK:STDOUT: !205 = !DILocation(line: 25, column: 3, scope: !196)
// CHECK:STDOUT: !206 = !{!197}
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.a76b78461416b4a8", scope: null, file: !32, line: 18, type: !147, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !216)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !207, type: !35)
// CHECK:STDOUT: !209 = !DILocalVariable(arg: 2, scope: !207, type: !35)
// CHECK:STDOUT: !215 = !DILocation(line: 19, column: 5, scope: !207)
// CHECK:STDOUT: !216 = !{!208, !209}
// CHECK:STDOUT: !237 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 18, type: !182, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !246)
// CHECK:STDOUT: !238 = !DILocalVariable(arg: 1, scope: !237, type: !35)
// CHECK:STDOUT: !239 = !DILocalVariable(arg: 2, scope: !237, type: !34)
// CHECK:STDOUT: !245 = !DILocation(line: 19, column: 5, scope: !237)
// CHECK:STDOUT: !246 = !{!238, !239}
// CHECK:STDOUT: !247 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.c5e4ba542a56b71f", scope: null, file: !32, line: 18, type: !147, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !256)
// CHECK:STDOUT: !248 = !DILocalVariable(arg: 1, scope: !247, type: !35)
// CHECK:STDOUT: !249 = !DILocalVariable(arg: 2, scope: !247, type: !35)
// CHECK:STDOUT: !255 = !DILocation(line: 19, column: 5, scope: !247)
// CHECK:STDOUT: !256 = !{!248, !249}
// CHECK:STDOUT: !257 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.c3359c2c217ad593", scope: null, file: !32, line: 18, type: !147, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !266)
// CHECK:STDOUT: !258 = !DILocalVariable(arg: 1, scope: !257, type: !35)
// CHECK:STDOUT: !259 = !DILocalVariable(arg: 2, scope: !257, type: !35)
// CHECK:STDOUT: !265 = !DILocation(line: 19, column: 5, scope: !257)
// CHECK:STDOUT: !266 = !{!258, !259}
// CHECK:STDOUT:
