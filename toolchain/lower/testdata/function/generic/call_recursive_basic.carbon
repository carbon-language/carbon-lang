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
// CHECK:STDOUT: define void @"_COp.C.Main:Copy.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc15_43, i64 0, i1 false), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !20
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !22
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !24
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !28
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !44
// CHECK:STDOUT:   %.loc40_9.3.temp = alloca {}, align 1, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !20
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !22
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !24
// CHECK:STDOUT:   store ptr %n.var, ptr %ptr_i32.var, align 8, !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !28
// CHECK:STDOUT:   store ptr %m.var, ptr %ptr_f64.var, align 8, !dbg !28
// CHECK:STDOUT:   %.loc34_5 = load i32, ptr %n.var, align 4, !dbg !32
// CHECK:STDOUT:   %F.call.loc34 = call i32 @_CF.Main.2fad9574e050607e(i32 %.loc34_5, i32 0), !dbg !33
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %m.var, align 8, !dbg !36
// CHECK:STDOUT:   %F.call.loc35 = call double @_CF.Main.21aefb7dfdebb8e4(double %.loc35_5, i32 0), !dbg !37
// CHECK:STDOUT:   %.loc36_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !40
// CHECK:STDOUT:   %F.call.loc36 = call ptr @_CF.Main.417c9c4a651ab31b(ptr %.loc36_5, i32 0), !dbg !41
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !42
// CHECK:STDOUT:   %F.call.loc37 = call ptr @_CF.Main.417c9c4a651ab31b(ptr %.loc37_5, i32 0), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !44
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc15_43, i64 0, i1 false), !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc40_9.3.temp), !dbg !47
// CHECK:STDOUT:   call void @_CF.Main.42513763e01cba2c(ptr %.loc40_9.3.temp, ptr %c.var, i32 0), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !56
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !57
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !59
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !61
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.2fad9574e050607e(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !62
// CHECK:STDOUT:   ret i32 %F.call, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !72
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !73
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !75
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !77
// CHECK:STDOUT:   %F.call = call double @_CF.Main.21aefb7dfdebb8e4(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !78
// CHECK:STDOUT:   ret double %F.call, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !88
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !89
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !91
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !93
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.417c9c4a651ab31b(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !94
// CHECK:STDOUT:   ret ptr %F.call, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.42513763e01cba2c(ptr sret({}) %return, ptr %x, i32 %count) #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !120
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !121
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @"_COp.C.Main:Copy.Core"(ptr %return, ptr %x), !dbg !122
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !125
// CHECK:STDOUT:   call void @_CF.Main.42513763e01cba2c(ptr %return, ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.417c9c4a651ab31b, { 1, 2, 0 }
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
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.C.Main:Copy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !18)
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 15, column: 34, scope: !12)
// CHECK:STDOUT: !18 = !{!13}
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 28, type: !11, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !20 = !DILocation(line: 29, column: 3, scope: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 30, column: 3, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 31, column: 3, scope: !19)
// CHECK:STDOUT: !28 = !DILocation(line: 32, column: 3, scope: !19)
// CHECK:STDOUT: !32 = !DILocation(line: 34, column: 5, scope: !19)
// CHECK:STDOUT: !33 = !DILocation(line: 34, column: 3, scope: !19)
// CHECK:STDOUT: !34 = !{!6, !6, !6}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 35, column: 5, scope: !19)
// CHECK:STDOUT: !37 = !DILocation(line: 35, column: 3, scope: !19)
// CHECK:STDOUT: !38 = !{!7, !7, !6}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 36, column: 5, scope: !19)
// CHECK:STDOUT: !41 = !DILocation(line: 36, column: 3, scope: !19)
// CHECK:STDOUT: !42 = !DILocation(line: 37, column: 5, scope: !19)
// CHECK:STDOUT: !43 = !DILocation(line: 37, column: 3, scope: !19)
// CHECK:STDOUT: !44 = !DILocation(line: 39, column: 3, scope: !19)
// CHECK:STDOUT: !47 = !DILocation(line: 40, column: 3, scope: !19)
// CHECK:STDOUT: !48 = !DILocation(line: 28, column: 1, scope: !19)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2fad9574e050607e", scope: null, file: !4, line: 21, type: !35, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !64)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !6)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 2, scope: !49, type: !6)
// CHECK:STDOUT: !56 = !DILocation(line: 22, column: 7, scope: !49)
// CHECK:STDOUT: !57 = !DILocation(line: 22, column: 6, scope: !49)
// CHECK:STDOUT: !59 = !DILocation(line: 23, column: 5, scope: !49)
// CHECK:STDOUT: !61 = !DILocation(line: 25, column: 15, scope: !49)
// CHECK:STDOUT: !62 = !DILocation(line: 25, column: 10, scope: !49)
// CHECK:STDOUT: !63 = !DILocation(line: 25, column: 3, scope: !49)
// CHECK:STDOUT: !64 = !{!50, !51}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 21, type: !39, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !7)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 2, scope: !65, type: !6)
// CHECK:STDOUT: !72 = !DILocation(line: 22, column: 7, scope: !65)
// CHECK:STDOUT: !73 = !DILocation(line: 22, column: 6, scope: !65)
// CHECK:STDOUT: !75 = !DILocation(line: 23, column: 5, scope: !65)
// CHECK:STDOUT: !77 = !DILocation(line: 25, column: 15, scope: !65)
// CHECK:STDOUT: !78 = !DILocation(line: 25, column: 10, scope: !65)
// CHECK:STDOUT: !79 = !DILocation(line: 25, column: 3, scope: !65)
// CHECK:STDOUT: !80 = !{!66, !67}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.417c9c4a651ab31b", scope: null, file: !4, line: 21, type: !39, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !96)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !7)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 2, scope: !81, type: !6)
// CHECK:STDOUT: !88 = !DILocation(line: 22, column: 7, scope: !81)
// CHECK:STDOUT: !89 = !DILocation(line: 22, column: 6, scope: !81)
// CHECK:STDOUT: !91 = !DILocation(line: 23, column: 5, scope: !81)
// CHECK:STDOUT: !93 = !DILocation(line: 25, column: 15, scope: !81)
// CHECK:STDOUT: !94 = !DILocation(line: 25, column: 10, scope: !81)
// CHECK:STDOUT: !95 = !DILocation(line: 25, column: 3, scope: !81)
// CHECK:STDOUT: !96 = !{!82, !83}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.42513763e01cba2c", scope: null, file: !4, line: 21, type: !39, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !128)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !7)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 2, scope: !113, type: !6)
// CHECK:STDOUT: !120 = !DILocation(line: 22, column: 7, scope: !113)
// CHECK:STDOUT: !121 = !DILocation(line: 22, column: 6, scope: !113)
// CHECK:STDOUT: !122 = !DILocation(line: 23, column: 12, scope: !113)
// CHECK:STDOUT: !123 = !DILocation(line: 23, column: 5, scope: !113)
// CHECK:STDOUT: !125 = !DILocation(line: 25, column: 15, scope: !113)
// CHECK:STDOUT: !126 = !DILocation(line: 25, column: 10, scope: !113)
// CHECK:STDOUT: !127 = !DILocation(line: 25, column: 3, scope: !113)
// CHECK:STDOUT: !128 = !{!114, !115}
// CHECK:STDOUT:
