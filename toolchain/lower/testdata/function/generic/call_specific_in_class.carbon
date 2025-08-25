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
  fn Cfn[self: Self, T:! type](x: T) -> T {
    return x;
  }
}

fn G[T:! type](x: T) -> T {
  var c: C;
  return c.Cfn(x);
}

fn F[T:! type](x: T) -> T {
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
// CHECK:STDOUT: define void @_CM.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !8
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !9
// CHECK:STDOUT:   %var_i32.var = alloca i32, align 4, !dbg !10
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !11
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !7
// CHECK:STDOUT:   %.loc34 = load ptr, ptr %ptr_i32.var, align 8, !dbg !13
// CHECK:STDOUT:   %F.call.loc34 = call ptr @_CF.Main.e8193710fd35b608(ptr %.loc34), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !8
// CHECK:STDOUT:   %.loc36 = load ptr, ptr %ptr_f64.var, align 8, !dbg !15
// CHECK:STDOUT:   %F.call.loc36 = call ptr @_CF.Main.e8193710fd35b608(ptr %.loc36), !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !9
// CHECK:STDOUT:   %.loc38 = load ptr, ptr %ptr_i8.var, align 8, !dbg !17
// CHECK:STDOUT:   %F.call.loc38 = call ptr @_CF.Main.e8193710fd35b608(ptr %.loc38), !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_i32.var), !dbg !10
// CHECK:STDOUT:   store i32 0, ptr %var_i32.var, align 4, !dbg !10
// CHECK:STDOUT:   %.loc40 = load i32, ptr %var_i32.var, align 4, !dbg !19
// CHECK:STDOUT:   %F.call.loc40 = call i32 @_CF.Main.b88d1103f417c6d4(i32 %.loc40), !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !11
// CHECK:STDOUT:   store double 0.000000e+00, ptr %var_f64.var, align 8, !dbg !11
// CHECK:STDOUT:   %.loc42 = load double, ptr %var_f64.var, align 8, !dbg !21
// CHECK:STDOUT:   %F.call.loc42 = call double @_CF.Main.d4b5665541d5d7a8(double %.loc42), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !12
// CHECK:STDOUT:   %F.call.loc44 = call %type @_CF.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.e8193710fd35b608(ptr %x) !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.e8193710fd35b608(ptr %x), !dbg !26
// CHECK:STDOUT:   ret ptr %G.call, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.b88d1103f417c6d4(i32 %x) !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.b88d1103f417c6d4(i32 %x), !dbg !29
// CHECK:STDOUT:   ret i32 %G.call, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.d4b5665541d5d7a8(double %x) !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.d4b5665541d5d7a8(double %x), !dbg !32
// CHECK:STDOUT:   ret double %G.call, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr %type @_CF.Main.5754c7a55c7cbe4a(%type %x) !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call %type @_CG.Main.5754c7a55c7cbe4a(%type %x), !dbg !35
// CHECK:STDOUT:   ret %type %G.call, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.e8193710fd35b608(ptr %x) !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !38
// CHECK:STDOUT:   %C.Cfn.call = call ptr @_CCfn.C.Main.e8193710fd35b608(ptr %c.var, ptr %x), !dbg !39
// CHECK:STDOUT:   ret ptr %C.Cfn.call, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.b88d1103f417c6d4(i32 %x) !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !42
// CHECK:STDOUT:   %C.Cfn.call = call i32 @_CCfn.C.Main.b88d1103f417c6d4(ptr %c.var, i32 %x), !dbg !43
// CHECK:STDOUT:   ret i32 %C.Cfn.call, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.d4b5665541d5d7a8(double %x) !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !46
// CHECK:STDOUT:   %C.Cfn.call = call double @_CCfn.C.Main.d4b5665541d5d7a8(ptr %c.var, double %x), !dbg !47
// CHECK:STDOUT:   ret double %C.Cfn.call, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr %type @_CG.Main.5754c7a55c7cbe4a(%type %x) !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !50
// CHECK:STDOUT:   %C.Cfn.call = call %type @_CCfn.C.Main.5754c7a55c7cbe4a(ptr %c.var, %type %x), !dbg !51
// CHECK:STDOUT:   ret %type %C.Cfn.call, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CCfn.C.Main.e8193710fd35b608(ptr %self, ptr %x) !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CCfn.C.Main.b88d1103f417c6d4(ptr %self, i32 %x) !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CCfn.C.Main.d4b5665541d5d7a8(ptr %self, double %x) !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr %type @_CCfn.C.Main.5754c7a55c7cbe4a(ptr %self, %type %x) !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type %x, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 9, 8, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.e8193710fd35b608, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_specific_in_class.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
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
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.e8193710fd35b608", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !26 = !DILocation(line: 29, column: 10, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 29, column: 3, scope: !25)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b88d1103f417c6d4", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !29 = !DILocation(line: 29, column: 10, scope: !28)
// CHECK:STDOUT: !30 = !DILocation(line: 29, column: 3, scope: !28)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.d4b5665541d5d7a8", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !32 = !DILocation(line: 29, column: 10, scope: !31)
// CHECK:STDOUT: !33 = !DILocation(line: 29, column: 3, scope: !31)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.5754c7a55c7cbe4a", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !35 = !DILocation(line: 29, column: 10, scope: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 29, column: 3, scope: !34)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.e8193710fd35b608", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !38 = !DILocation(line: 24, column: 3, scope: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 25, column: 10, scope: !37)
// CHECK:STDOUT: !40 = !DILocation(line: 25, column: 3, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.b88d1103f417c6d4", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !42 = !DILocation(line: 24, column: 3, scope: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 25, column: 10, scope: !41)
// CHECK:STDOUT: !44 = !DILocation(line: 25, column: 3, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.d4b5665541d5d7a8", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !46 = !DILocation(line: 24, column: 3, scope: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 25, column: 10, scope: !45)
// CHECK:STDOUT: !48 = !DILocation(line: 25, column: 3, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.5754c7a55c7cbe4a", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !50 = !DILocation(line: 24, column: 3, scope: !49)
// CHECK:STDOUT: !51 = !DILocation(line: 25, column: 10, scope: !49)
// CHECK:STDOUT: !52 = !DILocation(line: 25, column: 3, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.e8193710fd35b608", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !54 = !DILocation(line: 19, column: 5, scope: !53)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.b88d1103f417c6d4", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !56 = !DILocation(line: 19, column: 5, scope: !55)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.d4b5665541d5d7a8", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !58 = !DILocation(line: 19, column: 5, scope: !57)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.5754c7a55c7cbe4a", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !60 = !DILocation(line: 19, column: 5, scope: !59)
