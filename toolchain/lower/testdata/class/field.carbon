// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/field.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

class C {
  var a: i32;
  var b: C*;
}

fn F(c: C) -> i32 {
  return (*c.b).a;
}

fn Run() -> i32 {
  var c: C;
  c.a = 1;
  c.b = &c;
  return F(c);
}

// --- inplace_init_with_nonconst.carbon

library "[[@TEST_NAME]]";

class Other { }

class Use {
  var a: Other*;
  var b: Other;
}

fn Run() {
  var o: Other;
  // .a is initialized with a non-constant expression
  // .b is initialized with in-place class_init
  var _: Use = {.a = &o, .b = {}};
}

// --- implicit_init_with_nonempty_nonconst.carbon

library "[[@TEST_NAME]]";

class Other {
  var v: i32;
}

class Use {
  var a: Other*;
  var b: Other;
}

fn Run() {
  var o: Other;
  // .a is initialized with a non-constant expression
  // .b is initialized with in-place class_init
  var _: Use = {.a = &o, .b = {.v = 42}};
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main(ptr %c) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_13.1.b = getelementptr inbounds nuw { i32, ptr }, ptr %c, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   %.loc10_13.2 = load ptr, ptr %.loc10_13.1.b, align 8, !dbg !33
// CHECK:STDOUT:   %.loc10_16.1.a = getelementptr inbounds nuw { i32, ptr }, ptr %.loc10_13.2, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   %.loc10_16.2 = load i32, ptr %.loc10_16.1.a, align 4, !dbg !35
// CHECK:STDOUT:   ret i32 %.loc10_16.2, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i32, ptr }, align 8, !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !41
// CHECK:STDOUT:   %.loc15_4.a = getelementptr inbounds nuw { i32, ptr }, ptr %c.var, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   store i32 1, ptr %.loc15_4.a, align 4, !dbg !43
// CHECK:STDOUT:   %.loc16.b = getelementptr inbounds nuw { i32, ptr }, ptr %c.var, i32 0, i32 1, !dbg !44
// CHECK:STDOUT:   store ptr %c.var, ptr %.loc16.b, align 8, !dbg !44
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(ptr %c.var), !dbg !48
// CHECK:STDOUT:   ret i32 %F.call, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!22, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!22}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !20, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !37)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !23)
// CHECK:STDOUT: !33 = !DILocation(line: 10, column: 12, scope: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 10, column: 10, scope: !28)
// CHECK:STDOUT: !36 = !DILocation(line: 10, column: 3, scope: !28)
// CHECK:STDOUT: !37 = !{!29}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !20, line: 13, type: !27, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !41 = !DILocation(line: 14, column: 3, scope: !38)
// CHECK:STDOUT: !43 = !DILocation(line: 15, column: 3, scope: !38)
// CHECK:STDOUT: !44 = !DILocation(line: 16, column: 3, scope: !38)
// CHECK:STDOUT: !48 = !DILocation(line: 17, column: 10, scope: !38)
// CHECK:STDOUT: !49 = !DILocation(line: 17, column: 3, scope: !38)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'inplace_init_with_nonconst.carbon'
// CHECK:STDOUT: source_filename = "inplace_init_with_nonconst.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Other.val.loc15_33.5 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %o.var = alloca {}, align 1, !dbg !53
// CHECK:STDOUT:   %_.var = alloca { ptr, {} }, align 8, !dbg !55
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %o.var), !dbg !53
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !55
// CHECK:STDOUT:   %.loc15_33.2.a = getelementptr inbounds nuw { ptr, {} }, ptr %_.var, i32 0, i32 0, !dbg !58
// CHECK:STDOUT:   store ptr %o.var, ptr %.loc15_33.2.a, align 8, !dbg !58
// CHECK:STDOUT:   %.loc15_33.4.b = getelementptr inbounds nuw { ptr, {} }, ptr %_.var, i32 0, i32 1, !dbg !58
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %.loc15_33.4.b, ptr align 1 @Other.val.loc15_33.5, i64 0, i1 false), !dbg !58
// CHECK:STDOUT:   ret i32 0, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!51}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !26 = !{!22}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !50 = !DIFile(filename: "inplace_init_with_nonconst.carbon", directory: "")
// CHECK:STDOUT: !51 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !50, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !50, line: 11, type: !27, spFlags: DISPFlagDefinition, unit: !51)
// CHECK:STDOUT: !53 = !DILocation(line: 12, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !DILocation(line: 15, column: 3, scope: !52)
// CHECK:STDOUT: !58 = !DILocation(line: 15, column: 16, scope: !52)
// CHECK:STDOUT: !60 = !DILocation(line: 11, column: 1, scope: !52)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'implicit_init_with_nonempty_nonconst.carbon'
// CHECK:STDOUT: source_filename = "implicit_init_with_nonempty_nonconst.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Other.val.loc17_40.5 = internal constant { i32 } { i32 42 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %o.var = alloca { i32 }, align 4, !dbg !64
// CHECK:STDOUT:   %_.var = alloca <{ ptr, { i32 } }>, align 8, !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %o.var), !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !66
// CHECK:STDOUT:   %.loc17_40.2.a = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %_.var, i32 0, i32 0, !dbg !69
// CHECK:STDOUT:   store ptr %o.var, ptr %.loc17_40.2.a, align 8, !dbg !69
// CHECK:STDOUT:   %.loc17_40.4.b = getelementptr inbounds nuw <{ ptr, { i32 } }>, ptr %_.var, i32 0, i32 1, !dbg !69
// CHECK:STDOUT:   %.loc17_39.3.v = getelementptr inbounds nuw { i32 }, ptr %.loc17_40.4.b, i32 0, i32 0, !dbg !70
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %.loc17_40.4.b, ptr align 4 @Other.val.loc17_40.5, i64 4, i1 false), !dbg !69
// CHECK:STDOUT:   ret i32 0, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!62}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !26 = !{!22}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !61 = !DIFile(filename: "implicit_init_with_nonempty_nonconst.carbon", directory: "")
// CHECK:STDOUT: !62 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !61, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !61, line: 13, type: !27, spFlags: DISPFlagDefinition, unit: !62)
// CHECK:STDOUT: !64 = !DILocation(line: 14, column: 3, scope: !63)
// CHECK:STDOUT: !66 = !DILocation(line: 17, column: 3, scope: !63)
// CHECK:STDOUT: !69 = !DILocation(line: 17, column: 16, scope: !63)
// CHECK:STDOUT: !70 = !DILocation(line: 17, column: 31, scope: !63)
// CHECK:STDOUT: !72 = !DILocation(line: 13, column: 1, scope: !63)
// CHECK:STDOUT:
