// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/var/destroy_control_flow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/var/destroy_control_flow.carbon

// --- destructible.carbon

library "[[@TEST_NAME]]";

class Destructible {
  impl as Core.DestroyImpl {
    fn Op(self);
  }
};

fn Use(ref d: Destructible);

// --- if_then.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn IfThen(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  if (cond) {
    var b: Destructible = {};
    Use(ref b);
  }
  Use(ref a);
}

// --- if_else.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn IfElse(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  if (cond) {
    var b: Destructible = {};
    Use(ref b);
  } else {
    var c: Destructible = {};
    Use(ref c);
  }
  Use(ref a);
}

// --- if_return.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn IfReturn(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  if (cond) {
    var b: Destructible = {};
    Use(ref b);
    return;
  }
}

// --- while_break.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn WhileBreak(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  while (cond) {
    Use(ref a);
    var b: Destructible = {};
    Use(ref b);
    if (cond) {
      var c: Destructible = {};
      Use(ref c);
      break;
    }
  }
  Use(ref a);
}

// --- while_continue.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn WhileContinue(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  while (cond) {
    Use(ref a);
    var b: Destructible = {};
    Use(ref b);
    if (cond) {
      var c: Destructible = {};
      Use(ref c);
      continue;
    }
  }
  Use(ref a);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'destructible.carbon'
// CHECK:STDOUT: source_filename = "destructible.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Destructible.Main:DestroyImpl.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.1421ed8cb8ddfaad.Core:Destructible.Main:DestroyImpl.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Destructible.Main:DestroyImpl.Core"(ptr %self), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!36}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !35 = !DIFile(filename: "destructible.carbon", directory: "")
// CHECK:STDOUT: !36 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !35, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.1421ed8cb8ddfaad.Core:Destructible.Main:DestroyImpl.Core", scope: null, file: !35, line: 6, type: !39, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !44)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !37)
// CHECK:STDOUT: !43 = !DILocation(line: 6, column: 5, scope: !40)
// CHECK:STDOUT: !44 = !{!41}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_then.carbon'
// CHECK:STDOUT: source_filename = "if_then.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfThen.Main(i1 %cond) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !50
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !57
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !50
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !50
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !54
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !56
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !57
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !57
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !61
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %b.var), !dbg !57
// CHECK:STDOUT:   br label %if.else, !dbg !62
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !65
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %a.var), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %t) #0 !dbg !68 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr %t), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c59c2663f7aa0268, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!46}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !45 = !DIFile(filename: "if_then.carbon", directory: "")
// CHECK:STDOUT: !46 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !45, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "IfThen", linkageName: "_CIfThen.Main", scope: null, file: !45, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !46, retainedNodes: !67)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !37)
// CHECK:STDOUT: !50 = !DILocation(line: 6, column: 3, scope: !47)
// CHECK:STDOUT: !54 = !DILocation(line: 7, column: 3, scope: !47)
// CHECK:STDOUT: !56 = !DILocation(line: 8, column: 6, scope: !47)
// CHECK:STDOUT: !57 = !DILocation(line: 9, column: 5, scope: !47)
// CHECK:STDOUT: !61 = !DILocation(line: 10, column: 5, scope: !47)
// CHECK:STDOUT: !62 = !DILocation(line: 8, column: 3, scope: !47)
// CHECK:STDOUT: !65 = !DILocation(line: 12, column: 3, scope: !47)
// CHECK:STDOUT: !66 = !DILocation(line: 5, column: 1, scope: !47)
// CHECK:STDOUT: !67 = !{!48}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c59c2663f7aa0268", scope: null, file: !25, line: 32, type: !39, spFlags: DISPFlagDefinition, unit: !46, retainedNodes: !74)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !37)
// CHECK:STDOUT: !71 = !DILocation(line: 33, column: 5, scope: !68)
// CHECK:STDOUT: !73 = !DILocation(line: 32, column: 1, scope: !68)
// CHECK:STDOUT: !74 = !{!69}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_else.carbon'
// CHECK:STDOUT: source_filename = "if_else.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfElse.Main(i1 %cond) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !80
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !87
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !93
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !80
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !80
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !84
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !86
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !87
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !87
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !91
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %b.var), !dbg !87
// CHECK:STDOUT:   br label %if.done, !dbg !92
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !93
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !93
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !97
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %c.var), !dbg !93
// CHECK:STDOUT:   br label %if.done, !dbg !92
// CHECK:STDOUT:
// CHECK:STDOUT: if.done:                                          ; preds = %if.else, %if.then
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !100
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %a.var), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %t) #0 !dbg !103 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr %t), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c59c2663f7aa0268, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!76}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !75 = !DIFile(filename: "if_else.carbon", directory: "")
// CHECK:STDOUT: !76 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !75, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "IfElse", linkageName: "_CIfElse.Main", scope: null, file: !75, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !102)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !37)
// CHECK:STDOUT: !80 = !DILocation(line: 6, column: 3, scope: !77)
// CHECK:STDOUT: !84 = !DILocation(line: 7, column: 3, scope: !77)
// CHECK:STDOUT: !86 = !DILocation(line: 8, column: 6, scope: !77)
// CHECK:STDOUT: !87 = !DILocation(line: 9, column: 5, scope: !77)
// CHECK:STDOUT: !91 = !DILocation(line: 10, column: 5, scope: !77)
// CHECK:STDOUT: !92 = !DILocation(line: 8, column: 3, scope: !77)
// CHECK:STDOUT: !93 = !DILocation(line: 12, column: 5, scope: !77)
// CHECK:STDOUT: !97 = !DILocation(line: 13, column: 5, scope: !77)
// CHECK:STDOUT: !100 = !DILocation(line: 15, column: 3, scope: !77)
// CHECK:STDOUT: !101 = !DILocation(line: 5, column: 1, scope: !77)
// CHECK:STDOUT: !102 = !{!78}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c59c2663f7aa0268", scope: null, file: !25, line: 32, type: !39, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !109)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !37)
// CHECK:STDOUT: !106 = !DILocation(line: 33, column: 5, scope: !103)
// CHECK:STDOUT: !108 = !DILocation(line: 32, column: 1, scope: !103)
// CHECK:STDOUT: !109 = !{!104}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_return.carbon'
// CHECK:STDOUT: source_filename = "if_return.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfReturn.Main(i1 %cond) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !115
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !122
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !115
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !115
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !119
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !121
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !122
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !122
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !126
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %b.var), !dbg !122
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %a.var), !dbg !115
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %a.var), !dbg !115
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %t) #0 !dbg !130 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr %t), !dbg !133
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c59c2663f7aa0268, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!111}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !110 = !DIFile(filename: "if_return.carbon", directory: "")
// CHECK:STDOUT: !111 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !110, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "IfReturn", linkageName: "_CIfReturn.Main", scope: null, file: !110, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !111, retainedNodes: !129)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !37)
// CHECK:STDOUT: !115 = !DILocation(line: 6, column: 3, scope: !112)
// CHECK:STDOUT: !119 = !DILocation(line: 7, column: 3, scope: !112)
// CHECK:STDOUT: !121 = !DILocation(line: 8, column: 6, scope: !112)
// CHECK:STDOUT: !122 = !DILocation(line: 9, column: 5, scope: !112)
// CHECK:STDOUT: !126 = !DILocation(line: 10, column: 5, scope: !112)
// CHECK:STDOUT: !127 = !DILocation(line: 11, column: 5, scope: !112)
// CHECK:STDOUT: !128 = !DILocation(line: 5, column: 1, scope: !112)
// CHECK:STDOUT: !129 = !{!113}
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c59c2663f7aa0268", scope: null, file: !25, line: 32, type: !39, spFlags: DISPFlagDefinition, unit: !111, retainedNodes: !136)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !37)
// CHECK:STDOUT: !133 = !DILocation(line: 33, column: 5, scope: !130)
// CHECK:STDOUT: !135 = !DILocation(line: 32, column: 1, scope: !130)
// CHECK:STDOUT: !136 = !{!131}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'while_break.carbon'
// CHECK:STDOUT: source_filename = "while_break.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhileBreak.Main(i1 %cond) #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !142
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !152
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !159
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !142
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !142
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !146
// CHECK:STDOUT:   br label %while.cond, !dbg !147
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %entry
// CHECK:STDOUT:   br i1 %cond, label %while.body, label %while.done, !dbg !147
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !151
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !152
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !152
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !156
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !158
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !159
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !159
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !163
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %c.var), !dbg !159
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %b.var), !dbg !152
// CHECK:STDOUT:   br label %while.done, !dbg !164
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %b.var), !dbg !152
// CHECK:STDOUT:   br label %while.cond, !dbg !165
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %if.then, %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !168
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %a.var), !dbg !142
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %t) #0 !dbg !171 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr %t), !dbg !174
// CHECK:STDOUT:   ret void, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c59c2663f7aa0268, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!138}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !137 = !DIFile(filename: "while_break.carbon", directory: "")
// CHECK:STDOUT: !138 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !137, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "WhileBreak", linkageName: "_CWhileBreak.Main", scope: null, file: !137, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !170)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !37)
// CHECK:STDOUT: !142 = !DILocation(line: 6, column: 3, scope: !139)
// CHECK:STDOUT: !146 = !DILocation(line: 7, column: 3, scope: !139)
// CHECK:STDOUT: !147 = !DILocation(line: 8, column: 9, scope: !139)
// CHECK:STDOUT: !151 = !DILocation(line: 9, column: 5, scope: !139)
// CHECK:STDOUT: !152 = !DILocation(line: 10, column: 5, scope: !139)
// CHECK:STDOUT: !156 = !DILocation(line: 11, column: 5, scope: !139)
// CHECK:STDOUT: !158 = !DILocation(line: 12, column: 8, scope: !139)
// CHECK:STDOUT: !159 = !DILocation(line: 13, column: 7, scope: !139)
// CHECK:STDOUT: !163 = !DILocation(line: 14, column: 7, scope: !139)
// CHECK:STDOUT: !164 = !DILocation(line: 15, column: 7, scope: !139)
// CHECK:STDOUT: !165 = !DILocation(line: 8, column: 3, scope: !139)
// CHECK:STDOUT: !168 = !DILocation(line: 18, column: 3, scope: !139)
// CHECK:STDOUT: !169 = !DILocation(line: 5, column: 1, scope: !139)
// CHECK:STDOUT: !170 = !{!140}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c59c2663f7aa0268", scope: null, file: !25, line: 32, type: !39, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !177)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !37)
// CHECK:STDOUT: !174 = !DILocation(line: 33, column: 5, scope: !171)
// CHECK:STDOUT: !176 = !DILocation(line: 32, column: 1, scope: !171)
// CHECK:STDOUT: !177 = !{!172}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'while_continue.carbon'
// CHECK:STDOUT: source_filename = "while_continue.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhileContinue.Main(i1 %cond) #0 !dbg !180 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !183
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !193
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !200
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !183
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !183
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !187
// CHECK:STDOUT:   br label %while.cond, !dbg !188
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %if.then, %entry
// CHECK:STDOUT:   br i1 %cond, label %while.body, label %while.done, !dbg !188
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !192
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !193
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !193
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !197
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !199
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !200
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3.1, i64 0, i1 false), !dbg !200
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !204
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %c.var), !dbg !200
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %b.var), !dbg !193
// CHECK:STDOUT:   br label %while.cond, !dbg !205
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %b.var), !dbg !193
// CHECK:STDOUT:   br label %while.cond, !dbg !206
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !209
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %a.var), !dbg !183
// CHECK:STDOUT:   ret void, !dbg !210
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c59c2663f7aa0268(ptr %t) #0 !dbg !212 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.4a8993846ac47976.Core:Destructible.Main:DestroyImpl.Core"(ptr %t), !dbg !215
// CHECK:STDOUT:   ret void, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c59c2663f7aa0268, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!179}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !178 = !DIFile(filename: "while_continue.carbon", directory: "")
// CHECK:STDOUT: !179 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !178, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !180 = distinct !DISubprogram(name: "WhileContinue", linkageName: "_CWhileContinue.Main", scope: null, file: !178, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !179, retainedNodes: !211)
// CHECK:STDOUT: !181 = !DILocalVariable(arg: 1, scope: !180, type: !37)
// CHECK:STDOUT: !183 = !DILocation(line: 6, column: 3, scope: !180)
// CHECK:STDOUT: !187 = !DILocation(line: 7, column: 3, scope: !180)
// CHECK:STDOUT: !188 = !DILocation(line: 8, column: 9, scope: !180)
// CHECK:STDOUT: !192 = !DILocation(line: 9, column: 5, scope: !180)
// CHECK:STDOUT: !193 = !DILocation(line: 10, column: 5, scope: !180)
// CHECK:STDOUT: !197 = !DILocation(line: 11, column: 5, scope: !180)
// CHECK:STDOUT: !199 = !DILocation(line: 12, column: 8, scope: !180)
// CHECK:STDOUT: !200 = !DILocation(line: 13, column: 7, scope: !180)
// CHECK:STDOUT: !204 = !DILocation(line: 14, column: 7, scope: !180)
// CHECK:STDOUT: !205 = !DILocation(line: 15, column: 7, scope: !180)
// CHECK:STDOUT: !206 = !DILocation(line: 8, column: 3, scope: !180)
// CHECK:STDOUT: !209 = !DILocation(line: 18, column: 3, scope: !180)
// CHECK:STDOUT: !210 = !DILocation(line: 5, column: 1, scope: !180)
// CHECK:STDOUT: !211 = !{!181}
// CHECK:STDOUT: !212 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c59c2663f7aa0268", scope: null, file: !25, line: 32, type: !39, spFlags: DISPFlagDefinition, unit: !179, retainedNodes: !218)
// CHECK:STDOUT: !213 = !DILocalVariable(arg: 1, scope: !212, type: !37)
// CHECK:STDOUT: !215 = !DILocation(line: 33, column: 5, scope: !212)
// CHECK:STDOUT: !217 = !DILocation(line: 32, column: 1, scope: !212)
// CHECK:STDOUT: !218 = !{!213}
// CHECK:STDOUT:
