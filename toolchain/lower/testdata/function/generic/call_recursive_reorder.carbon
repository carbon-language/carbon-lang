// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_reorder.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_reorder.carbon

// Builds on `call_recursive_basic.carbon`.

fn F[T: type, U: type](x: T, y: U, count: i32) -> i32 {
  if (count > 2) {
    return count;
  }
  return F(y, x, count + 1);
}

fn M() {
  var n: i32 = 0;
  var m: f64 = 1.0;
  var b: bool = false;
  var ptr_i32 : i32* = &n;
  var ptr_f64 : f64* = &m;
  var ptr_bool : bool* = &b;

  F(ptr_i32, ptr_bool, 0);
  F(ptr_f64, ptr_bool, 0);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_recursive_reorder.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_reorder.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !15
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !17
// CHECK:STDOUT:   %b.var = alloca i1, align 1, !dbg !19
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !21
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !25
// CHECK:STDOUT:   %ptr_bool.var = alloca ptr, align 8, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !15
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !17
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !19
// CHECK:STDOUT:   store i8 0, ptr %b.var, align 1, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !21
// CHECK:STDOUT:   store ptr %n.var, ptr %ptr_i32.var, align 8, !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !25
// CHECK:STDOUT:   store ptr %m.var, ptr %ptr_f64.var, align 8, !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_bool.var), !dbg !29
// CHECK:STDOUT:   store ptr %b.var, ptr %ptr_bool.var, align 8, !dbg !29
// CHECK:STDOUT:   %.loc30_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !33
// CHECK:STDOUT:   %.loc30_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !34
// CHECK:STDOUT:   %F.call.loc30 = call i32 @_CF.Main.369f53a8d4c38871(ptr %.loc30_5, ptr %.loc30_14, i32 0), !dbg !35
// CHECK:STDOUT:   %.loc31_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !38
// CHECK:STDOUT:   %.loc31_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !39
// CHECK:STDOUT:   %F.call.loc31 = call i32 @_CF.Main.369f53a8d4c38871(ptr %.loc31_5, ptr %.loc31_14, i32 0), !dbg !40
// CHECK:STDOUT:   call void @"_COp.dc14d4dd21801791:core.Destroy.Core"(ptr %m.var), !dbg !17
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %n.var), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.369f53a8d4c38871(ptr %x, ptr %y, i32 %count) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !59
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !60
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !62
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !65
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.369f53a8d4c38871(ptr %y, ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !66
// CHECK:STDOUT:   ret i32 %F.call, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.369f53a8d4c38871, { 0, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "call_recursive_reorder.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{null, !9}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{null, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 22, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DILocation(line: 23, column: 3, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 24, column: 3, scope: !14)
// CHECK:STDOUT: !19 = !DILocation(line: 25, column: 3, scope: !14)
// CHECK:STDOUT: !21 = !DILocation(line: 26, column: 3, scope: !14)
// CHECK:STDOUT: !25 = !DILocation(line: 27, column: 3, scope: !14)
// CHECK:STDOUT: !29 = !DILocation(line: 28, column: 3, scope: !14)
// CHECK:STDOUT: !33 = !DILocation(line: 30, column: 5, scope: !14)
// CHECK:STDOUT: !34 = !DILocation(line: 30, column: 14, scope: !14)
// CHECK:STDOUT: !35 = !DILocation(line: 30, column: 3, scope: !14)
// CHECK:STDOUT: !36 = !{!6, !9, !9, !6}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 31, column: 5, scope: !14)
// CHECK:STDOUT: !39 = !DILocation(line: 31, column: 14, scope: !14)
// CHECK:STDOUT: !40 = !DILocation(line: 31, column: 3, scope: !14)
// CHECK:STDOUT: !41 = !DILocation(line: 22, column: 1, scope: !14)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Op", linkageName: "_COp.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 24, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !9)
// CHECK:STDOUT: !44 = !DILocation(line: 24, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 23, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !49)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !6)
// CHECK:STDOUT: !48 = !DILocation(line: 23, column: 3, scope: !46)
// CHECK:STDOUT: !49 = !{!47}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.369f53a8d4c38871", scope: null, file: !4, line: 15, type: !37, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !68)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !9)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 2, scope: !50, type: !9)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 3, scope: !50, type: !6)
// CHECK:STDOUT: !59 = !DILocation(line: 16, column: 7, scope: !50)
// CHECK:STDOUT: !60 = !DILocation(line: 16, column: 6, scope: !50)
// CHECK:STDOUT: !62 = !DILocation(line: 17, column: 5, scope: !50)
// CHECK:STDOUT: !65 = !DILocation(line: 19, column: 18, scope: !50)
// CHECK:STDOUT: !66 = !DILocation(line: 19, column: 10, scope: !50)
// CHECK:STDOUT: !67 = !DILocation(line: 19, column: 3, scope: !50)
// CHECK:STDOUT: !68 = !{!51, !52, !53}
// CHECK:STDOUT:
