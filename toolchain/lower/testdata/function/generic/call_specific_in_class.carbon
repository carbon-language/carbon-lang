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
  fn Cfn[self: Self, T:! Core.Copy](x: T) -> T {
    return x;
  }
}

fn G[T:! Core.Copy](x: T) -> T {
  var c: C;
  return c.Cfn(x);
}

fn F[T:! Core.Copy](x: T) -> T {
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
  var c: C;
  F(C);
}

// CHECK:STDOUT: ; ModuleID = 'call_specific_in_class.carbon'
// CHECK:STDOUT: source_filename = "call_specific_in_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !8
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !9
// CHECK:STDOUT:   %var_i32.var = alloca i32, align 4, !dbg !10
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !11
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !7
// CHECK:STDOUT:   %.loc34_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !13
// CHECK:STDOUT:   %F.call.loc34 = call ptr @_CF.Main.0a952f8bcc623ce6(ptr %.loc34_5), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !8
// CHECK:STDOUT:   %.loc36_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !15
// CHECK:STDOUT:   %F.call.loc36 = call ptr @_CF.Main.0a952f8bcc623ce6(ptr %.loc36_5), !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !9
// CHECK:STDOUT:   %.loc38_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !17
// CHECK:STDOUT:   %F.call.loc38 = call ptr @_CF.Main.0a952f8bcc623ce6(ptr %.loc38_5), !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_i32.var), !dbg !10
// CHECK:STDOUT:   store i32 0, ptr %var_i32.var, align 4, !dbg !10
// CHECK:STDOUT:   %.loc40_5 = load i32, ptr %var_i32.var, align 4, !dbg !19
// CHECK:STDOUT:   %F.call.loc40 = call i32 @_CF.Main.5450dc8e8b8e0899(i32 %.loc40_5), !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !11
// CHECK:STDOUT:   store double 0.000000e+00, ptr %var_f64.var, align 8, !dbg !11
// CHECK:STDOUT:   %.loc42_5 = load double, ptr %var_f64.var, align 8, !dbg !21
// CHECK:STDOUT:   %F.call.loc42 = call double @_CF.Main.83aece103b0a8681(double %.loc42_5), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !12
// CHECK:STDOUT:   %F.call.loc44 = call %type @_CF.Main.402deed6b8733082(%type zeroinitializer), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.0a952f8bcc623ce6(ptr %x) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.0a952f8bcc623ce6(ptr %x), !dbg !31
// CHECK:STDOUT:   ret ptr %G.call, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.5450dc8e8b8e0899(i32 %x) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.5450dc8e8b8e0899(i32 %x), !dbg !39
// CHECK:STDOUT:   ret i32 %G.call, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.83aece103b0a8681(double %x) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.83aece103b0a8681(double %x), !dbg !44
// CHECK:STDOUT:   ret double %G.call, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type @_CF.Main.402deed6b8733082(%type %x) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call %type @_CG.Main.402deed6b8733082(%type %x), !dbg !49
// CHECK:STDOUT:   ret %type %G.call, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.0a952f8bcc623ce6(ptr %x) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !54
// CHECK:STDOUT:   %C.Cfn.call = call ptr @_CCfn.C.Main.0a952f8bcc623ce6(ptr %c.var, ptr %x), !dbg !55
// CHECK:STDOUT:   ret ptr %C.Cfn.call, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.5450dc8e8b8e0899(i32 %x) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !60
// CHECK:STDOUT:   %C.Cfn.call = call i32 @_CCfn.C.Main.5450dc8e8b8e0899(ptr %c.var, i32 %x), !dbg !61
// CHECK:STDOUT:   ret i32 %C.Cfn.call, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.83aece103b0a8681(double %x) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !66
// CHECK:STDOUT:   %C.Cfn.call = call double @_CCfn.C.Main.83aece103b0a8681(ptr %c.var, double %x), !dbg !67
// CHECK:STDOUT:   ret double %C.Cfn.call, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type @_CG.Main.402deed6b8733082(%type %x) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !72
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !72
// CHECK:STDOUT:   %C.Cfn.call = call %type @_CCfn.C.Main.402deed6b8733082(ptr %c.var, %type %x), !dbg !73
// CHECK:STDOUT:   ret %type %C.Cfn.call, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CCfn.C.Main.0a952f8bcc623ce6(ptr %self, ptr %x) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CCfn.C.Main.5450dc8e8b8e0899(ptr %self, i32 %x) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CCfn.C.Main.83aece103b0a8681(ptr %self, double %x) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type @_CCfn.C.Main.402deed6b8733082(ptr %self, %type %x) #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type %x, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 9, 8, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.0a952f8bcc623ce6, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_specific_in_class.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 33, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 35, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 37, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 39, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 41, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 43, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 34, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 34, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 36, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 36, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 38, column: 5, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 38, column: 3, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 40, column: 5, scope: !4)
// CHECK:STDOUT: !20 = !DILocation(line: 40, column: 3, scope: !4)
// CHECK:STDOUT: !21 = !DILocation(line: 42, column: 5, scope: !4)
// CHECK:STDOUT: !22 = !DILocation(line: 42, column: 3, scope: !4)
// CHECK:STDOUT: !23 = !DILocation(line: 44, column: 3, scope: !4)
// CHECK:STDOUT: !24 = !DILocation(line: 32, column: 1, scope: !4)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.0a952f8bcc623ce6", scope: null, file: !3, line: 28, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !29)
// CHECK:STDOUT: !26 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !27 = !{!28, !28}
// CHECK:STDOUT: !28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !25, type: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 29, column: 10, scope: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 29, column: 3, scope: !25)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.5450dc8e8b8e0899", scope: null, file: !3, line: 28, type: !34, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !37)
// CHECK:STDOUT: !34 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !35 = !{!36, !36}
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{!38}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !33, type: !36)
// CHECK:STDOUT: !39 = !DILocation(line: 29, column: 10, scope: !33)
// CHECK:STDOUT: !40 = !DILocation(line: 29, column: 3, scope: !33)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.83aece103b0a8681", scope: null, file: !3, line: 28, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !28)
// CHECK:STDOUT: !44 = !DILocation(line: 29, column: 10, scope: !41)
// CHECK:STDOUT: !45 = !DILocation(line: 29, column: 3, scope: !41)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.402deed6b8733082", scope: null, file: !3, line: 28, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !47)
// CHECK:STDOUT: !47 = !{!48}
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !46, type: !28)
// CHECK:STDOUT: !49 = !DILocation(line: 29, column: 10, scope: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 29, column: 3, scope: !46)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.0a952f8bcc623ce6", scope: null, file: !3, line: 23, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !52)
// CHECK:STDOUT: !52 = !{!53}
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !51, type: !28)
// CHECK:STDOUT: !54 = !DILocation(line: 24, column: 3, scope: !51)
// CHECK:STDOUT: !55 = !DILocation(line: 25, column: 10, scope: !51)
// CHECK:STDOUT: !56 = !DILocation(line: 25, column: 3, scope: !51)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.5450dc8e8b8e0899", scope: null, file: !3, line: 23, type: !34, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !58)
// CHECK:STDOUT: !58 = !{!59}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !57, type: !36)
// CHECK:STDOUT: !60 = !DILocation(line: 24, column: 3, scope: !57)
// CHECK:STDOUT: !61 = !DILocation(line: 25, column: 10, scope: !57)
// CHECK:STDOUT: !62 = !DILocation(line: 25, column: 3, scope: !57)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.83aece103b0a8681", scope: null, file: !3, line: 23, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !64)
// CHECK:STDOUT: !64 = !{!65}
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !63, type: !28)
// CHECK:STDOUT: !66 = !DILocation(line: 24, column: 3, scope: !63)
// CHECK:STDOUT: !67 = !DILocation(line: 25, column: 10, scope: !63)
// CHECK:STDOUT: !68 = !DILocation(line: 25, column: 3, scope: !63)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.402deed6b8733082", scope: null, file: !3, line: 23, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !70)
// CHECK:STDOUT: !70 = !{!71}
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !69, type: !28)
// CHECK:STDOUT: !72 = !DILocation(line: 24, column: 3, scope: !69)
// CHECK:STDOUT: !73 = !DILocation(line: 25, column: 10, scope: !69)
// CHECK:STDOUT: !74 = !DILocation(line: 25, column: 3, scope: !69)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.0a952f8bcc623ce6", scope: null, file: !3, line: 18, type: !76, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !78)
// CHECK:STDOUT: !76 = !DISubroutineType(types: !77)
// CHECK:STDOUT: !77 = !{!28, !28, !28}
// CHECK:STDOUT: !78 = !{!79, !80}
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !75, type: !28)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 2, scope: !75, type: !28)
// CHECK:STDOUT: !81 = !DILocation(line: 19, column: 5, scope: !75)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.5450dc8e8b8e0899", scope: null, file: !3, line: 18, type: !83, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !85)
// CHECK:STDOUT: !83 = !DISubroutineType(types: !84)
// CHECK:STDOUT: !84 = !{!36, !28, !36}
// CHECK:STDOUT: !85 = !{!86, !87}
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !82, type: !28)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 2, scope: !82, type: !36)
// CHECK:STDOUT: !88 = !DILocation(line: 19, column: 5, scope: !82)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.83aece103b0a8681", scope: null, file: !3, line: 18, type: !76, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !90)
// CHECK:STDOUT: !90 = !{!91, !92}
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !89, type: !28)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 2, scope: !89, type: !28)
// CHECK:STDOUT: !93 = !DILocation(line: 19, column: 5, scope: !89)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.402deed6b8733082", scope: null, file: !3, line: 18, type: !76, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !95)
// CHECK:STDOUT: !95 = !{!96, !97}
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !94, type: !28)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 2, scope: !94, type: !28)
// CHECK:STDOUT: !98 = !DILocation(line: 19, column: 5, scope: !94)
