// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_basic.carbon

class C {
  impl as Core.Copy {
    fn Op(unused self) -> Self { return {}; }
  }
}

// The two specifics for recursive function F, when T is a pointer, could be
// deduplicated.
fn F[T:! Core.Copy](x: T, count: i32) -> T {
  if (count > 0) {
    return x;
  }
  return F(x, count + 1);
}

fn M() {
  var n: i32 = 0;
  var m: f64 = 1.0;
  var ptr_i32 : i32* = &n;
  var ptr_f64 : f64* = &m;

  F(n, 0);
  F(m, 0);
  F(ptr_i32, 0);
  F(ptr_f64, 0);

  var c: C = {};
  F(c, 0);
}

// CHECK:STDOUT: ; ModuleID = 'call_recursive_basic.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc15_43 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.C.Main:Copy.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc15_43, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !14
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !15
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !16
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !17
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !18
// CHECK:STDOUT:   %.loc40_9.3.temp = alloca {}, align 1, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !14
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !15
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !16
// CHECK:STDOUT:   store ptr %n.var, ptr %ptr_i32.var, align 8, !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !17
// CHECK:STDOUT:   store ptr %m.var, ptr %ptr_f64.var, align 8, !dbg !17
// CHECK:STDOUT:   %.loc34_5 = load i32, ptr %n.var, align 4, !dbg !20
// CHECK:STDOUT:   %F.call.loc34 = call i32 @_CF.Main.5bd8d5767580997a(i32 %.loc34_5, i32 0), !dbg !21
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %m.var, align 8, !dbg !22
// CHECK:STDOUT:   %F.call.loc35 = call double @_CF.Main.c27b765f2159c9a0(double %.loc35_5, i32 0), !dbg !23
// CHECK:STDOUT:   %.loc36_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !24
// CHECK:STDOUT:   %F.call.loc36 = call ptr @_CF.Main.2621742a01881e29(ptr %.loc36_5, i32 0), !dbg !25
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !26
// CHECK:STDOUT:   %F.call.loc37 = call ptr @_CF.Main.2621742a01881e29(ptr %.loc37_5, i32 0), !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !18
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc15_43, i64 0, i1 false), !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc40_9.3.temp), !dbg !19
// CHECK:STDOUT:   call void @_CF.Main.4d597ec621e4ba00(ptr %.loc40_9.3.temp, ptr %c.var, i32 0), !dbg !19
// CHECK:STDOUT:   call void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr %.loc40_9.3.temp), !dbg !19
// CHECK:STDOUT:   call void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr %c.var), !dbg !18
// CHECK:STDOUT:   call void @"_COp.30d8007fdcc57f83:core.Destroy.Core"(ptr %m.var), !dbg !15
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %n.var), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.30d8007fdcc57f83:core.Destroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.5bd8d5767580997a(i32 %x, i32 %count) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !52
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !53
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !54
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !55
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.5bd8d5767580997a(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !56
// CHECK:STDOUT:   ret i32 %F.call, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.c27b765f2159c9a0(double %x, i32 %count) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !64
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !65
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !66
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !67
// CHECK:STDOUT:   %F.call = call double @_CF.Main.c27b765f2159c9a0(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !68
// CHECK:STDOUT:   ret double %F.call, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.2621742a01881e29(ptr %x, i32 %count) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !74
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !75
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !76
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !77
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.2621742a01881e29(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !78
// CHECK:STDOUT:   ret ptr %F.call, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.4d597ec621e4ba00(ptr sret({}) %return, ptr %x, i32 %count) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !84
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !85
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @"_COp.C.Main:Copy.Core"(ptr %return, ptr %x), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !88
// CHECK:STDOUT:   call void @_CF.Main.4d597ec621e4ba00(ptr %return, ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.807eb4f2d3ed7e54:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.2621742a01881e29, { 1, 2, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "call_recursive_basic.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.C.Main:Copy.Core", scope: null, file: !1, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 15, column: 34, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !1, line: 28, type: !12, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 29, column: 3, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 30, column: 3, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 31, column: 3, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 32, column: 3, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 39, column: 3, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 40, column: 3, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 34, column: 5, scope: !11)
// CHECK:STDOUT: !21 = !DILocation(line: 34, column: 3, scope: !11)
// CHECK:STDOUT: !22 = !DILocation(line: 35, column: 5, scope: !11)
// CHECK:STDOUT: !23 = !DILocation(line: 35, column: 3, scope: !11)
// CHECK:STDOUT: !24 = !DILocation(line: 36, column: 5, scope: !11)
// CHECK:STDOUT: !25 = !DILocation(line: 36, column: 3, scope: !11)
// CHECK:STDOUT: !26 = !DILocation(line: 37, column: 5, scope: !11)
// CHECK:STDOUT: !27 = !DILocation(line: 37, column: 3, scope: !11)
// CHECK:STDOUT: !28 = !DILocation(line: 28, column: 1, scope: !11)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp.807eb4f2d3ed7e54:core.Destroy.Core", scope: null, file: !1, line: 40, type: !30, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !32)
// CHECK:STDOUT: !30 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !31 = !{null, !7}
// CHECK:STDOUT: !32 = !{!33}
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !29, type: !7)
// CHECK:STDOUT: !34 = !DILocation(line: 40, column: 3, scope: !29)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Op", linkageName: "_COp.30d8007fdcc57f83:core.Destroy.Core", scope: null, file: !1, line: 30, type: !30, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !36)
// CHECK:STDOUT: !36 = !{!37}
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !35, type: !7)
// CHECK:STDOUT: !38 = !DILocation(line: 30, column: 3, scope: !35)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !1, line: 29, type: !40, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !43)
// CHECK:STDOUT: !40 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !41 = !{null, !42}
// CHECK:STDOUT: !42 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !43 = !{!44}
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !39, type: !42)
// CHECK:STDOUT: !45 = !DILocation(line: 29, column: 3, scope: !39)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.5bd8d5767580997a", scope: null, file: !1, line: 21, type: !47, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !49)
// CHECK:STDOUT: !47 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !48 = !{!42, !42, !42}
// CHECK:STDOUT: !49 = !{!50, !51}
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !46, type: !42)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 2, scope: !46, type: !42)
// CHECK:STDOUT: !52 = !DILocation(line: 22, column: 7, scope: !46)
// CHECK:STDOUT: !53 = !DILocation(line: 22, column: 6, scope: !46)
// CHECK:STDOUT: !54 = !DILocation(line: 23, column: 5, scope: !46)
// CHECK:STDOUT: !55 = !DILocation(line: 25, column: 15, scope: !46)
// CHECK:STDOUT: !56 = !DILocation(line: 25, column: 10, scope: !46)
// CHECK:STDOUT: !57 = !DILocation(line: 25, column: 3, scope: !46)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c27b765f2159c9a0", scope: null, file: !1, line: 21, type: !59, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !61)
// CHECK:STDOUT: !59 = !DISubroutineType(types: !60)
// CHECK:STDOUT: !60 = !{!7, !7, !42}
// CHECK:STDOUT: !61 = !{!62, !63}
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !58, type: !7)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 2, scope: !58, type: !42)
// CHECK:STDOUT: !64 = !DILocation(line: 22, column: 7, scope: !58)
// CHECK:STDOUT: !65 = !DILocation(line: 22, column: 6, scope: !58)
// CHECK:STDOUT: !66 = !DILocation(line: 23, column: 5, scope: !58)
// CHECK:STDOUT: !67 = !DILocation(line: 25, column: 15, scope: !58)
// CHECK:STDOUT: !68 = !DILocation(line: 25, column: 10, scope: !58)
// CHECK:STDOUT: !69 = !DILocation(line: 25, column: 3, scope: !58)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2621742a01881e29", scope: null, file: !1, line: 21, type: !59, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !71)
// CHECK:STDOUT: !71 = !{!72, !73}
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !70, type: !7)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 2, scope: !70, type: !42)
// CHECK:STDOUT: !74 = !DILocation(line: 22, column: 7, scope: !70)
// CHECK:STDOUT: !75 = !DILocation(line: 22, column: 6, scope: !70)
// CHECK:STDOUT: !76 = !DILocation(line: 23, column: 5, scope: !70)
// CHECK:STDOUT: !77 = !DILocation(line: 25, column: 15, scope: !70)
// CHECK:STDOUT: !78 = !DILocation(line: 25, column: 10, scope: !70)
// CHECK:STDOUT: !79 = !DILocation(line: 25, column: 3, scope: !70)
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d597ec621e4ba00", scope: null, file: !1, line: 21, type: !59, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !81)
// CHECK:STDOUT: !81 = !{!82, !83}
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !80, type: !7)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 2, scope: !80, type: !42)
// CHECK:STDOUT: !84 = !DILocation(line: 22, column: 7, scope: !80)
// CHECK:STDOUT: !85 = !DILocation(line: 22, column: 6, scope: !80)
// CHECK:STDOUT: !86 = !DILocation(line: 23, column: 12, scope: !80)
// CHECK:STDOUT: !87 = !DILocation(line: 23, column: 5, scope: !80)
// CHECK:STDOUT: !88 = !DILocation(line: 25, column: 15, scope: !80)
// CHECK:STDOUT: !89 = !DILocation(line: 25, column: 10, scope: !80)
// CHECK:STDOUT: !90 = !DILocation(line: 25, column: 3, scope: !80)
