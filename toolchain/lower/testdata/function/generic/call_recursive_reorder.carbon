// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_reorder.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_reorder.carbon

// Builds on `call_recursive_basic.carbon`.

fn F[T:! type, U:! type](x: T, y: U, count: i32) -> i32 {
  if (count > 2) {
    return count;
  }
  return F(y, x, count + 1);
}

fn M() {
  var ptr_i32 : i32*;
  var ptr_f64 : f64*;
  var ptr_bool : bool*;

  F(ptr_i32, ptr_bool, 0);
  F(ptr_f64, ptr_bool, 0);
}

// CHECK:STDOUT: ; ModuleID = 'call_recursive_reorder.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_reorder.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CM.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !8
// CHECK:STDOUT:   %ptr_bool.var = alloca ptr, align 8, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_i32.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_f64.var), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_bool.var), !dbg !9
// CHECK:STDOUT:   %.loc25_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !10
// CHECK:STDOUT:   %.loc25_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !11
// CHECK:STDOUT:   %F.call.loc25 = call i32 @_CF.Main.2fe87eb83f5a4614(ptr %.loc25_5, ptr %.loc25_14, i32 0), !dbg !12
// CHECK:STDOUT:   %.loc26_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !13
// CHECK:STDOUT:   %.loc26_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !14
// CHECK:STDOUT:   %F.call.loc26 = call i32 @_CF.Main.2fe87eb83f5a4614(ptr %.loc26_5, ptr %.loc26_14, i32 0), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.2fe87eb83f5a4614(ptr %x, ptr %y, i32 %count) !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 2, !dbg !18
// CHECK:STDOUT:   br i1 %int.greater, label %if.then, label %if.else, !dbg !19
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !21
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.2fe87eb83f5a4614(ptr %y, ptr %x, i32 %int.sadd), !dbg !22
// CHECK:STDOUT:   ret i32 %F.call, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.2fe87eb83f5a4614, { 0, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_recursive_reorder.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 21, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 22, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 23, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 25, column: 5, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 25, column: 14, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 25, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 26, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 26, column: 14, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 26, column: 3, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 20, column: 1, scope: !4)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2fe87eb83f5a4614", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !18 = !DILocation(line: 14, column: 7, scope: !17)
// CHECK:STDOUT: !19 = !DILocation(line: 14, column: 6, scope: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 15, column: 5, scope: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 17, column: 18, scope: !17)
// CHECK:STDOUT: !22 = !DILocation(line: 17, column: 10, scope: !17)
// CHECK:STDOUT: !23 = !DILocation(line: 17, column: 3, scope: !17)
