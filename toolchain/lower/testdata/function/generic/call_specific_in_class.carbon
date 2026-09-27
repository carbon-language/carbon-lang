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
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !55
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !61
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !65
// CHECK:STDOUT:   %var_i32.var = alloca i32, align 4, !dbg !69
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !75
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !79
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !55
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !55
// CHECK:STDOUT:   %.loc34_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !57
// CHECK:STDOUT:   %F.call.loc34 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc34_5), !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !61
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !61
// CHECK:STDOUT:   %.loc36_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !63
// CHECK:STDOUT:   %F.call.loc36 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc36_5), !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !65
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !65
// CHECK:STDOUT:   %.loc38_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !67
// CHECK:STDOUT:   %F.call.loc38 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc38_5), !dbg !68
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_i32.var), !dbg !69
// CHECK:STDOUT:   store i32 0, ptr %var_i32.var, align 4, !dbg !69
// CHECK:STDOUT:   %.loc40_5 = load i32, ptr %var_i32.var, align 4, !dbg !71
// CHECK:STDOUT:   %F.call.loc40 = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %.loc40_5), !dbg !72
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !75
// CHECK:STDOUT:   store double 0.000000e+00, ptr %var_f64.var, align 8, !dbg !75
// CHECK:STDOUT:   %.loc42_5 = load double, ptr %var_f64.var, align 8, !dbg !77
// CHECK:STDOUT:   %F.call.loc42 = call double @_CF.Main.c5e4ba542a56b71f(double %.loc42_5), !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !79
// CHECK:STDOUT:   %F.call.loc44 = call %type.26 @_CF.Main.c3359c2c217ad593(%type.26 zeroinitializer), !dbg !81
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %_.var), !dbg !79
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %var_f64.var), !dbg !75
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %var_i32.var), !dbg !69
// CHECK:STDOUT:   call void @"_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core"(ptr %ptr_i8.var), !dbg !65
// CHECK:STDOUT:   call void @"_CSelfDestruct.749b51755634ad17:core.Destroy.Core"(ptr %ptr_f64.var), !dbg !61
// CHECK:STDOUT:   call void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %ptr_i32.var), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.749b51755634ad17:core.Destroy.Core"(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %self) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a76b78461416b4a8(ptr %x) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x), !dbg !109
// CHECK:STDOUT:   ret ptr %G.call, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x) #0 !dbg !130 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x), !dbg !136
// CHECK:STDOUT:   ret i32 %G.call, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.c5e4ba542a56b71f(double %x) #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x), !dbg !145
// CHECK:STDOUT:   ret double %G.call, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.26 @_CF.Main.c3359c2c217ad593(%type.26 %x) #0 !dbg !148 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call %type.26 @_CG.Main.c3359c2c217ad593(%type.26 %x), !dbg !154
// CHECK:STDOUT:   ret %type.26 %G.call, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.a76b78461416b4a8(ptr %x) #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !162
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !162
// CHECK:STDOUT:   %C.Cfn.call = call ptr @_CCfn.C.Main.a76b78461416b4a8(ptr %c.var, ptr %x), !dbg !164
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !162
// CHECK:STDOUT:   ret ptr %C.Cfn.call, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x) #0 !dbg !192 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !197
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !197
// CHECK:STDOUT:   %C.Cfn.call = call i32 @_CCfn.C.Main.4d6ceefeb7f772f4(ptr %c.var, i32 %x), !dbg !199
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !197
// CHECK:STDOUT:   ret i32 %C.Cfn.call, !dbg !203
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.c5e4ba542a56b71f(double %x) #0 !dbg !205 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !210
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !210
// CHECK:STDOUT:   %C.Cfn.call = call double @_CCfn.C.Main.c5e4ba542a56b71f(ptr %c.var, double %x), !dbg !212
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !210
// CHECK:STDOUT:   ret double %C.Cfn.call, !dbg !214
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.26 @_CG.Main.c3359c2c217ad593(%type.26 %x) #0 !dbg !216 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !221
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !221
// CHECK:STDOUT:   %C.Cfn.call = call %type.26 @_CCfn.C.Main.c3359c2c217ad593(ptr %c.var, %type.26 %x), !dbg !223
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !221
// CHECK:STDOUT:   ret %type.26 %C.Cfn.call, !dbg !225
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CCfn.C.Main.a76b78461416b4a8(ptr %self, ptr %x) #0 !dbg !227 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !235
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CCfn.C.Main.4d6ceefeb7f772f4(ptr %self, i32 %x) #0 !dbg !257 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !265
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CCfn.C.Main.c5e4ba542a56b71f(ptr %self, double %x) #0 !dbg !267 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !275
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.26 @_CCfn.C.Main.c3359c2c217ad593(ptr %self, %type.26 %x) #0 !dbg !277 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.26 %x, !dbg !285
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
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !32, line: 24, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !49)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !35)
// CHECK:STDOUT: !48 = !DILocation(line: 24, column: 3, scope: !46)
// CHECK:STDOUT: !49 = !{!47}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !32, line: 24, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !53)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !35)
// CHECK:STDOUT: !52 = !DILocation(line: 24, column: 3, scope: !50)
// CHECK:STDOUT: !53 = !{!51}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !32, line: 32, type: !39, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !55 = !DILocation(line: 33, column: 3, scope: !54)
// CHECK:STDOUT: !57 = !DILocation(line: 34, column: 5, scope: !54)
// CHECK:STDOUT: !58 = !DILocation(line: 34, column: 3, scope: !54)
// CHECK:STDOUT: !59 = !{!35, !35}
// CHECK:STDOUT: !60 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !61 = !DILocation(line: 35, column: 3, scope: !54)
// CHECK:STDOUT: !63 = !DILocation(line: 36, column: 5, scope: !54)
// CHECK:STDOUT: !64 = !DILocation(line: 36, column: 3, scope: !54)
// CHECK:STDOUT: !65 = !DILocation(line: 37, column: 3, scope: !54)
// CHECK:STDOUT: !67 = !DILocation(line: 38, column: 5, scope: !54)
// CHECK:STDOUT: !68 = !DILocation(line: 38, column: 3, scope: !54)
// CHECK:STDOUT: !69 = !DILocation(line: 39, column: 3, scope: !54)
// CHECK:STDOUT: !71 = !DILocation(line: 40, column: 5, scope: !54)
// CHECK:STDOUT: !72 = !DILocation(line: 40, column: 3, scope: !54)
// CHECK:STDOUT: !73 = !{!34, !34}
// CHECK:STDOUT: !74 = !DISubroutineType(types: !73)
// CHECK:STDOUT: !75 = !DILocation(line: 41, column: 3, scope: !54)
// CHECK:STDOUT: !77 = !DILocation(line: 42, column: 5, scope: !54)
// CHECK:STDOUT: !78 = !DILocation(line: 42, column: 3, scope: !54)
// CHECK:STDOUT: !79 = !DILocation(line: 43, column: 3, scope: !54)
// CHECK:STDOUT: !81 = !DILocation(line: 44, column: 3, scope: !54)
// CHECK:STDOUT: !82 = !DILocation(line: 32, column: 1, scope: !54)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !32, line: 41, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !86)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !35)
// CHECK:STDOUT: !85 = !DILocation(line: 41, column: 3, scope: !83)
// CHECK:STDOUT: !86 = !{!84}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !32, line: 39, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !34)
// CHECK:STDOUT: !89 = !DILocation(line: 39, column: 3, scope: !87)
// CHECK:STDOUT: !90 = !{!88}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core", scope: null, file: !32, line: 37, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !94)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !35)
// CHECK:STDOUT: !93 = !DILocation(line: 37, column: 3, scope: !91)
// CHECK:STDOUT: !94 = !{!92}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.749b51755634ad17:core.Destroy.Core", scope: null, file: !32, line: 35, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !35)
// CHECK:STDOUT: !97 = !DILocation(line: 35, column: 3, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d088c7eb746488be:core.Destroy.Core", scope: null, file: !32, line: 33, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !102)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !35)
// CHECK:STDOUT: !101 = !DILocation(line: 33, column: 3, scope: !99)
// CHECK:STDOUT: !102 = !{!100}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a76b78461416b4a8", scope: null, file: !32, line: 28, type: !60, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !111)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !35)
// CHECK:STDOUT: !109 = !DILocation(line: 29, column: 10, scope: !103)
// CHECK:STDOUT: !110 = !DILocation(line: 29, column: 3, scope: !103)
// CHECK:STDOUT: !111 = !{!104}
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 28, type: !74, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !138)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !34)
// CHECK:STDOUT: !136 = !DILocation(line: 29, column: 10, scope: !130)
// CHECK:STDOUT: !137 = !DILocation(line: 29, column: 3, scope: !130)
// CHECK:STDOUT: !138 = !{!131}
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c5e4ba542a56b71f", scope: null, file: !32, line: 28, type: !60, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !147)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !35)
// CHECK:STDOUT: !145 = !DILocation(line: 29, column: 10, scope: !139)
// CHECK:STDOUT: !146 = !DILocation(line: 29, column: 3, scope: !139)
// CHECK:STDOUT: !147 = !{!140}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c3359c2c217ad593", scope: null, file: !32, line: 28, type: !60, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !156)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !35)
// CHECK:STDOUT: !154 = !DILocation(line: 29, column: 10, scope: !148)
// CHECK:STDOUT: !155 = !DILocation(line: 29, column: 3, scope: !148)
// CHECK:STDOUT: !156 = !{!149}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.a76b78461416b4a8", scope: null, file: !32, line: 23, type: !60, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !169)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !157, type: !35)
// CHECK:STDOUT: !162 = !DILocation(line: 24, column: 3, scope: !157)
// CHECK:STDOUT: !164 = !DILocation(line: 25, column: 10, scope: !157)
// CHECK:STDOUT: !166 = !{!35, !35, !35}
// CHECK:STDOUT: !167 = !DISubroutineType(types: !166)
// CHECK:STDOUT: !168 = !DILocation(line: 25, column: 3, scope: !157)
// CHECK:STDOUT: !169 = !{!158}
// CHECK:STDOUT: !192 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 23, type: !74, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !204)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 1, scope: !192, type: !34)
// CHECK:STDOUT: !197 = !DILocation(line: 24, column: 3, scope: !192)
// CHECK:STDOUT: !199 = !DILocation(line: 25, column: 10, scope: !192)
// CHECK:STDOUT: !201 = !{!34, !35, !34}
// CHECK:STDOUT: !202 = !DISubroutineType(types: !201)
// CHECK:STDOUT: !203 = !DILocation(line: 25, column: 3, scope: !192)
// CHECK:STDOUT: !204 = !{!193}
// CHECK:STDOUT: !205 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c5e4ba542a56b71f", scope: null, file: !32, line: 23, type: !60, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !215)
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 1, scope: !205, type: !35)
// CHECK:STDOUT: !210 = !DILocation(line: 24, column: 3, scope: !205)
// CHECK:STDOUT: !212 = !DILocation(line: 25, column: 10, scope: !205)
// CHECK:STDOUT: !214 = !DILocation(line: 25, column: 3, scope: !205)
// CHECK:STDOUT: !215 = !{!206}
// CHECK:STDOUT: !216 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c3359c2c217ad593", scope: null, file: !32, line: 23, type: !60, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !226)
// CHECK:STDOUT: !217 = !DILocalVariable(arg: 1, scope: !216, type: !35)
// CHECK:STDOUT: !221 = !DILocation(line: 24, column: 3, scope: !216)
// CHECK:STDOUT: !223 = !DILocation(line: 25, column: 10, scope: !216)
// CHECK:STDOUT: !225 = !DILocation(line: 25, column: 3, scope: !216)
// CHECK:STDOUT: !226 = !{!217}
// CHECK:STDOUT: !227 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.a76b78461416b4a8", scope: null, file: !32, line: 18, type: !167, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !236)
// CHECK:STDOUT: !228 = !DILocalVariable(arg: 1, scope: !227, type: !35)
// CHECK:STDOUT: !229 = !DILocalVariable(arg: 2, scope: !227, type: !35)
// CHECK:STDOUT: !235 = !DILocation(line: 19, column: 5, scope: !227)
// CHECK:STDOUT: !236 = !{!228, !229}
// CHECK:STDOUT: !257 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 18, type: !202, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !266)
// CHECK:STDOUT: !258 = !DILocalVariable(arg: 1, scope: !257, type: !35)
// CHECK:STDOUT: !259 = !DILocalVariable(arg: 2, scope: !257, type: !34)
// CHECK:STDOUT: !265 = !DILocation(line: 19, column: 5, scope: !257)
// CHECK:STDOUT: !266 = !{!258, !259}
// CHECK:STDOUT: !267 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.c5e4ba542a56b71f", scope: null, file: !32, line: 18, type: !167, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !276)
// CHECK:STDOUT: !268 = !DILocalVariable(arg: 1, scope: !267, type: !35)
// CHECK:STDOUT: !269 = !DILocalVariable(arg: 2, scope: !267, type: !35)
// CHECK:STDOUT: !275 = !DILocation(line: 19, column: 5, scope: !267)
// CHECK:STDOUT: !276 = !{!268, !269}
// CHECK:STDOUT: !277 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.c3359c2c217ad593", scope: null, file: !32, line: 18, type: !167, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !286)
// CHECK:STDOUT: !278 = !DILocalVariable(arg: 1, scope: !277, type: !35)
// CHECK:STDOUT: !279 = !DILocalVariable(arg: 2, scope: !277, type: !35)
// CHECK:STDOUT: !285 = !DILocation(line: 19, column: 5, scope: !277)
// CHECK:STDOUT: !286 = !{!278, !279}
// CHECK:STDOUT:
