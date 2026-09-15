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
fn F[T: Core.Copy](x: T, count: i32) -> T {
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_recursive_basic.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc15_43 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.C.Main:Copy.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc15_43, i64 0, i1 false), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !24
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !26
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !28
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !32
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !48
// CHECK:STDOUT:   %.loc40_9.3.temp = alloca {}, align 1, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !24
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !26
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !28
// CHECK:STDOUT:   store ptr %n.var, ptr %ptr_i32.var, align 8, !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !32
// CHECK:STDOUT:   store ptr %m.var, ptr %ptr_f64.var, align 8, !dbg !32
// CHECK:STDOUT:   %.loc34_5 = load i32, ptr %n.var, align 4, !dbg !36
// CHECK:STDOUT:   %F.call.loc34 = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %.loc34_5, i32 0), !dbg !37
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %m.var, align 8, !dbg !40
// CHECK:STDOUT:   %F.call.loc35 = call double @_CF.Main.c5e4ba542a56b71f(double %.loc35_5, i32 0), !dbg !41
// CHECK:STDOUT:   %.loc36_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !44
// CHECK:STDOUT:   %F.call.loc36 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc36_5, i32 0), !dbg !45
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !46
// CHECK:STDOUT:   %F.call.loc37 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc37_5, i32 0), !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !48
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc15_43, i64 0, i1 false), !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc40_9.3.temp), !dbg !51
// CHECK:STDOUT:   call void @_CF.Main.42513763e01cba2c(ptr %.loc40_9.3.temp, ptr %c.var, i32 0), !dbg !51
// CHECK:STDOUT:   call void @"_COp.08d9e38b928ba575:core.Destroy.Core"(ptr %.loc40_9.3.temp), !dbg !51
// CHECK:STDOUT:   call void @"_COp.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !48
// CHECK:STDOUT:   call void @"_COp.dc14d4dd21801791:core.Destroy.Core"(ptr %m.var), !dbg !26
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %n.var), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !72
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !73
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !75
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !77
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !78
// CHECK:STDOUT:   ret i32 %F.call, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !88
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !89
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !91
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !93
// CHECK:STDOUT:   %F.call = call double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !94
// CHECK:STDOUT:   ret double %F.call, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !104
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !105
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !107
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !109
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !110
// CHECK:STDOUT:   ret ptr %F.call, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.42513763e01cba2c(ptr sret({}) %return, ptr %x, i32 %count) #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !136
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !137
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @"_COp.C.Main:Copy.Core"(ptr %return, ptr %x), !dbg !138
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !141
// CHECK:STDOUT:   call void @_CF.Main.42513763e01cba2c(ptr %return, ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !142
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.08d9e38b928ba575:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.a76b78461416b4a8, { 1, 2, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "call_recursive_basic.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{null}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{null, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{null, !6}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "Op", linkageName: "_COp.C.Main:Copy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !22)
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !16, type: !7)
// CHECK:STDOUT: !21 = !DILocation(line: 15, column: 34, scope: !16)
// CHECK:STDOUT: !22 = !{!17}
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 28, type: !11, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !24 = !DILocation(line: 29, column: 3, scope: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 30, column: 3, scope: !23)
// CHECK:STDOUT: !28 = !DILocation(line: 31, column: 3, scope: !23)
// CHECK:STDOUT: !32 = !DILocation(line: 32, column: 3, scope: !23)
// CHECK:STDOUT: !36 = !DILocation(line: 34, column: 5, scope: !23)
// CHECK:STDOUT: !37 = !DILocation(line: 34, column: 3, scope: !23)
// CHECK:STDOUT: !38 = !{!6, !6, !6}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 35, column: 5, scope: !23)
// CHECK:STDOUT: !41 = !DILocation(line: 35, column: 3, scope: !23)
// CHECK:STDOUT: !42 = !{!7, !7, !6}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 36, column: 5, scope: !23)
// CHECK:STDOUT: !45 = !DILocation(line: 36, column: 3, scope: !23)
// CHECK:STDOUT: !46 = !DILocation(line: 37, column: 5, scope: !23)
// CHECK:STDOUT: !47 = !DILocation(line: 37, column: 3, scope: !23)
// CHECK:STDOUT: !48 = !DILocation(line: 39, column: 3, scope: !23)
// CHECK:STDOUT: !51 = !DILocation(line: 40, column: 3, scope: !23)
// CHECK:STDOUT: !52 = !DILocation(line: 28, column: 1, scope: !23)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !4, line: 40, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !7)
// CHECK:STDOUT: !55 = !DILocation(line: 40, column: 3, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Op", linkageName: "_COp.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 30, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !7)
// CHECK:STDOUT: !59 = !DILocation(line: 30, column: 3, scope: !57)
// CHECK:STDOUT: !60 = !{!58}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 29, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !64)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !6)
// CHECK:STDOUT: !63 = !DILocation(line: 29, column: 3, scope: !61)
// CHECK:STDOUT: !64 = !{!62}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 21, type: !39, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !6)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 2, scope: !65, type: !6)
// CHECK:STDOUT: !72 = !DILocation(line: 22, column: 7, scope: !65)
// CHECK:STDOUT: !73 = !DILocation(line: 22, column: 6, scope: !65)
// CHECK:STDOUT: !75 = !DILocation(line: 23, column: 5, scope: !65)
// CHECK:STDOUT: !77 = !DILocation(line: 25, column: 15, scope: !65)
// CHECK:STDOUT: !78 = !DILocation(line: 25, column: 10, scope: !65)
// CHECK:STDOUT: !79 = !DILocation(line: 25, column: 3, scope: !65)
// CHECK:STDOUT: !80 = !{!66, !67}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 21, type: !43, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !96)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !7)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 2, scope: !81, type: !6)
// CHECK:STDOUT: !88 = !DILocation(line: 22, column: 7, scope: !81)
// CHECK:STDOUT: !89 = !DILocation(line: 22, column: 6, scope: !81)
// CHECK:STDOUT: !91 = !DILocation(line: 23, column: 5, scope: !81)
// CHECK:STDOUT: !93 = !DILocation(line: 25, column: 15, scope: !81)
// CHECK:STDOUT: !94 = !DILocation(line: 25, column: 10, scope: !81)
// CHECK:STDOUT: !95 = !DILocation(line: 25, column: 3, scope: !81)
// CHECK:STDOUT: !96 = !{!82, !83}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a76b78461416b4a8", scope: null, file: !4, line: 21, type: !43, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !112)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !7)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 2, scope: !97, type: !6)
// CHECK:STDOUT: !104 = !DILocation(line: 22, column: 7, scope: !97)
// CHECK:STDOUT: !105 = !DILocation(line: 22, column: 6, scope: !97)
// CHECK:STDOUT: !107 = !DILocation(line: 23, column: 5, scope: !97)
// CHECK:STDOUT: !109 = !DILocation(line: 25, column: 15, scope: !97)
// CHECK:STDOUT: !110 = !DILocation(line: 25, column: 10, scope: !97)
// CHECK:STDOUT: !111 = !DILocation(line: 25, column: 3, scope: !97)
// CHECK:STDOUT: !112 = !{!98, !99}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.42513763e01cba2c", scope: null, file: !4, line: 21, type: !43, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !144)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !7)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 2, scope: !129, type: !6)
// CHECK:STDOUT: !136 = !DILocation(line: 22, column: 7, scope: !129)
// CHECK:STDOUT: !137 = !DILocation(line: 22, column: 6, scope: !129)
// CHECK:STDOUT: !138 = !DILocation(line: 23, column: 12, scope: !129)
// CHECK:STDOUT: !139 = !DILocation(line: 23, column: 5, scope: !129)
// CHECK:STDOUT: !141 = !DILocation(line: 25, column: 15, scope: !129)
// CHECK:STDOUT: !142 = !DILocation(line: 25, column: 10, scope: !129)
// CHECK:STDOUT: !143 = !DILocation(line: 25, column: 3, scope: !129)
// CHECK:STDOUT: !144 = !{!130, !131}
// CHECK:STDOUT:
