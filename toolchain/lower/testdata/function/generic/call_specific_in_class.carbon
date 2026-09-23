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
// CHECK:STDOUT:   %F.call.loc44 = call %type.26 @_CF.Main.71d6c598da8ca542(%type.26 zeroinitializer), !dbg !81
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
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.7945ecba5c505fc0:core.Destroy.Core"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self), !dbg !93
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core"(ptr %self) #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.749b51755634ad17:core.Destroy.Core"(ptr %self) #0 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %self) #0 !dbg !115 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a76b78461416b4a8(ptr %x) #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x), !dbg !125
// CHECK:STDOUT:   ret ptr %G.call, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x) #0 !dbg !146 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x), !dbg !152
// CHECK:STDOUT:   ret i32 %G.call, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.c5e4ba542a56b71f(double %x) #0 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x), !dbg !161
// CHECK:STDOUT:   ret double %G.call, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.26 @_CF.Main.71d6c598da8ca542(%type.26 %x) #0 !dbg !164 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call %type.26 @_CG.Main.71d6c598da8ca542(%type.26 %x), !dbg !170
// CHECK:STDOUT:   ret %type.26 %G.call, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.a76b78461416b4a8(ptr %x) #0 !dbg !173 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !178
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !178
// CHECK:STDOUT:   %C.Cfn.call = call ptr @_CCfn.C.Main.a76b78461416b4a8(ptr %c.var, ptr %x), !dbg !180
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !178
// CHECK:STDOUT:   ret ptr %C.Cfn.call, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x) #0 !dbg !208 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !213
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !213
// CHECK:STDOUT:   %C.Cfn.call = call i32 @_CCfn.C.Main.4d6ceefeb7f772f4(ptr %c.var, i32 %x), !dbg !215
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !213
// CHECK:STDOUT:   ret i32 %C.Cfn.call, !dbg !219
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.c5e4ba542a56b71f(double %x) #0 !dbg !221 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !226
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !226
// CHECK:STDOUT:   %C.Cfn.call = call double @_CCfn.C.Main.c5e4ba542a56b71f(ptr %c.var, double %x), !dbg !228
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !226
// CHECK:STDOUT:   ret double %C.Cfn.call, !dbg !230
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.26 @_CG.Main.71d6c598da8ca542(%type.26 %x) #0 !dbg !232 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !237
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !237
// CHECK:STDOUT:   %C.Cfn.call = call %type.26 @_CCfn.C.Main.71d6c598da8ca542(ptr %c.var, %type.26 %x), !dbg !239
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !237
// CHECK:STDOUT:   ret %type.26 %C.Cfn.call, !dbg !241
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CCfn.C.Main.a76b78461416b4a8(ptr %self, ptr %x) #0 !dbg !243 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !251
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CCfn.C.Main.4d6ceefeb7f772f4(ptr %self, i32 %x) #0 !dbg !273 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !281
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CCfn.C.Main.c5e4ba542a56b71f(ptr %self, double %x) #0 !dbg !283 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !291
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.26 @_CCfn.C.Main.71d6c598da8ca542(ptr %self, %type.26 %x) #0 !dbg !293 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.26 %x, !dbg !301
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
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.7945ecba5c505fc0:core.Destroy.Core", scope: null, file: !32, line: 41, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !86)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !35)
// CHECK:STDOUT: !85 = !DILocation(line: 41, column: 3, scope: !83)
// CHECK:STDOUT: !86 = !{!84}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !32, line: 41, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !35)
// CHECK:STDOUT: !89 = !DILocation(line: 41, column: 3, scope: !87)
// CHECK:STDOUT: !90 = !{!88}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !32, line: 41, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !94)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !35)
// CHECK:STDOUT: !93 = !DILocation(line: 41, column: 3, scope: !91)
// CHECK:STDOUT: !94 = !{!92}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !32, line: 39, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !34)
// CHECK:STDOUT: !97 = !DILocation(line: 39, column: 3, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !32, line: 39, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !102)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !34)
// CHECK:STDOUT: !101 = !DILocation(line: 39, column: 3, scope: !99)
// CHECK:STDOUT: !102 = !{!100}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !32, line: 39, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !106)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !34)
// CHECK:STDOUT: !105 = !DILocation(line: 39, column: 3, scope: !103)
// CHECK:STDOUT: !106 = !{!104}
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core", scope: null, file: !32, line: 37, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !110)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !35)
// CHECK:STDOUT: !109 = !DILocation(line: 37, column: 3, scope: !107)
// CHECK:STDOUT: !110 = !{!108}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.749b51755634ad17:core.Destroy.Core", scope: null, file: !32, line: 35, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !114)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !35)
// CHECK:STDOUT: !113 = !DILocation(line: 35, column: 3, scope: !111)
// CHECK:STDOUT: !114 = !{!112}
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d088c7eb746488be:core.Destroy.Core", scope: null, file: !32, line: 33, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !118)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !35)
// CHECK:STDOUT: !117 = !DILocation(line: 33, column: 3, scope: !115)
// CHECK:STDOUT: !118 = !{!116}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a76b78461416b4a8", scope: null, file: !32, line: 28, type: !60, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !127)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !35)
// CHECK:STDOUT: !125 = !DILocation(line: 29, column: 10, scope: !119)
// CHECK:STDOUT: !126 = !DILocation(line: 29, column: 3, scope: !119)
// CHECK:STDOUT: !127 = !{!120}
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 28, type: !74, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !154)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !34)
// CHECK:STDOUT: !152 = !DILocation(line: 29, column: 10, scope: !146)
// CHECK:STDOUT: !153 = !DILocation(line: 29, column: 3, scope: !146)
// CHECK:STDOUT: !154 = !{!147}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c5e4ba542a56b71f", scope: null, file: !32, line: 28, type: !60, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !163)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !35)
// CHECK:STDOUT: !161 = !DILocation(line: 29, column: 10, scope: !155)
// CHECK:STDOUT: !162 = !DILocation(line: 29, column: 3, scope: !155)
// CHECK:STDOUT: !163 = !{!156}
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.71d6c598da8ca542", scope: null, file: !32, line: 28, type: !60, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !172)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !35)
// CHECK:STDOUT: !170 = !DILocation(line: 29, column: 10, scope: !164)
// CHECK:STDOUT: !171 = !DILocation(line: 29, column: 3, scope: !164)
// CHECK:STDOUT: !172 = !{!165}
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.a76b78461416b4a8", scope: null, file: !32, line: 23, type: !60, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !185)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 1, scope: !173, type: !35)
// CHECK:STDOUT: !178 = !DILocation(line: 24, column: 3, scope: !173)
// CHECK:STDOUT: !180 = !DILocation(line: 25, column: 10, scope: !173)
// CHECK:STDOUT: !182 = !{!35, !35, !35}
// CHECK:STDOUT: !183 = !DISubroutineType(types: !182)
// CHECK:STDOUT: !184 = !DILocation(line: 25, column: 3, scope: !173)
// CHECK:STDOUT: !185 = !{!174}
// CHECK:STDOUT: !208 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 23, type: !74, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !220)
// CHECK:STDOUT: !209 = !DILocalVariable(arg: 1, scope: !208, type: !34)
// CHECK:STDOUT: !213 = !DILocation(line: 24, column: 3, scope: !208)
// CHECK:STDOUT: !215 = !DILocation(line: 25, column: 10, scope: !208)
// CHECK:STDOUT: !217 = !{!34, !35, !34}
// CHECK:STDOUT: !218 = !DISubroutineType(types: !217)
// CHECK:STDOUT: !219 = !DILocation(line: 25, column: 3, scope: !208)
// CHECK:STDOUT: !220 = !{!209}
// CHECK:STDOUT: !221 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c5e4ba542a56b71f", scope: null, file: !32, line: 23, type: !60, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !231)
// CHECK:STDOUT: !222 = !DILocalVariable(arg: 1, scope: !221, type: !35)
// CHECK:STDOUT: !226 = !DILocation(line: 24, column: 3, scope: !221)
// CHECK:STDOUT: !228 = !DILocation(line: 25, column: 10, scope: !221)
// CHECK:STDOUT: !230 = !DILocation(line: 25, column: 3, scope: !221)
// CHECK:STDOUT: !231 = !{!222}
// CHECK:STDOUT: !232 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.71d6c598da8ca542", scope: null, file: !32, line: 23, type: !60, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !242)
// CHECK:STDOUT: !233 = !DILocalVariable(arg: 1, scope: !232, type: !35)
// CHECK:STDOUT: !237 = !DILocation(line: 24, column: 3, scope: !232)
// CHECK:STDOUT: !239 = !DILocation(line: 25, column: 10, scope: !232)
// CHECK:STDOUT: !241 = !DILocation(line: 25, column: 3, scope: !232)
// CHECK:STDOUT: !242 = !{!233}
// CHECK:STDOUT: !243 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.a76b78461416b4a8", scope: null, file: !32, line: 18, type: !183, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !252)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 1, scope: !243, type: !35)
// CHECK:STDOUT: !245 = !DILocalVariable(arg: 2, scope: !243, type: !35)
// CHECK:STDOUT: !251 = !DILocation(line: 19, column: 5, scope: !243)
// CHECK:STDOUT: !252 = !{!244, !245}
// CHECK:STDOUT: !273 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 18, type: !218, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !282)
// CHECK:STDOUT: !274 = !DILocalVariable(arg: 1, scope: !273, type: !35)
// CHECK:STDOUT: !275 = !DILocalVariable(arg: 2, scope: !273, type: !34)
// CHECK:STDOUT: !281 = !DILocation(line: 19, column: 5, scope: !273)
// CHECK:STDOUT: !282 = !{!274, !275}
// CHECK:STDOUT: !283 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.c5e4ba542a56b71f", scope: null, file: !32, line: 18, type: !183, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !292)
// CHECK:STDOUT: !284 = !DILocalVariable(arg: 1, scope: !283, type: !35)
// CHECK:STDOUT: !285 = !DILocalVariable(arg: 2, scope: !283, type: !35)
// CHECK:STDOUT: !291 = !DILocation(line: 19, column: 5, scope: !283)
// CHECK:STDOUT: !292 = !{!284, !285}
// CHECK:STDOUT: !293 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.71d6c598da8ca542", scope: null, file: !32, line: 18, type: !183, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !302)
// CHECK:STDOUT: !294 = !DILocalVariable(arg: 1, scope: !293, type: !35)
// CHECK:STDOUT: !295 = !DILocalVariable(arg: 2, scope: !293, type: !35)
// CHECK:STDOUT: !301 = !DILocation(line: 19, column: 5, scope: !293)
// CHECK:STDOUT: !302 = !{!294, !295}
// CHECK:STDOUT:
