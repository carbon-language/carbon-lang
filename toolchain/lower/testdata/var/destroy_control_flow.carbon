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
  // TODO: Implement Core.Destructor instead.
  impl as Core.Destroy {
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
// CHECK:STDOUT: declare void @"_COp.Destructible.Main:Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Destroy.3fda586d04e243ea.Core:Destructible.Main:Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Destructible.Main:Destroy.Core"(ptr %self), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "destructible.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Destroy.3fda586d04e243ea.Core:Destructible.Main:Destroy.Core", scope: null, file: !32, line: 7, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !41)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !34)
// CHECK:STDOUT: !40 = !DILocation(line: 7, column: 5, scope: !37)
// CHECK:STDOUT: !41 = !{!38}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_then.carbon'
// CHECK:STDOUT: source_filename = "if_then.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfThen.Main(i1 %cond) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !47
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !47
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !47
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !51
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !53
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !54
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !54
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !58
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %b.var), !dbg !54
// CHECK:STDOUT:   br label %if.else, !dbg !59
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !62
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %a.var), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.474c886d55da95e6:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!43}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !42 = !DIFile(filename: "if_then.carbon", directory: "")
// CHECK:STDOUT: !43 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !42, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "IfThen", linkageName: "_CIfThen.Main", scope: null, file: !42, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !64)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !34)
// CHECK:STDOUT: !47 = !DILocation(line: 6, column: 3, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 7, column: 3, scope: !44)
// CHECK:STDOUT: !53 = !DILocation(line: 8, column: 6, scope: !44)
// CHECK:STDOUT: !54 = !DILocation(line: 9, column: 5, scope: !44)
// CHECK:STDOUT: !58 = !DILocation(line: 10, column: 5, scope: !44)
// CHECK:STDOUT: !59 = !DILocation(line: 8, column: 3, scope: !44)
// CHECK:STDOUT: !62 = !DILocation(line: 12, column: 3, scope: !44)
// CHECK:STDOUT: !63 = !DILocation(line: 5, column: 1, scope: !44)
// CHECK:STDOUT: !64 = !{!45}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Op", linkageName: "_COp.474c886d55da95e6:core.Destroy.Core", scope: null, file: !42, line: 9, type: !36, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !34)
// CHECK:STDOUT: !67 = !DILocation(line: 9, column: 5, scope: !65)
// CHECK:STDOUT: !68 = !{!66}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_else.carbon'
// CHECK:STDOUT: source_filename = "if_else.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfElse.Main(i1 %cond) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !74
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !81
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !74
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !74
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !78
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !80
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !81
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !81
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !85
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %b.var), !dbg !81
// CHECK:STDOUT:   br label %if.done, !dbg !86
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !87
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !87
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !91
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %c.var), !dbg !87
// CHECK:STDOUT:   br label %if.done, !dbg !86
// CHECK:STDOUT:
// CHECK:STDOUT: if.done:                                          ; preds = %if.else, %if.then
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !94
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %a.var), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.474c886d55da95e6:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!70}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !69 = !DIFile(filename: "if_else.carbon", directory: "")
// CHECK:STDOUT: !70 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !69, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "IfElse", linkageName: "_CIfElse.Main", scope: null, file: !69, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !96)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !34)
// CHECK:STDOUT: !74 = !DILocation(line: 6, column: 3, scope: !71)
// CHECK:STDOUT: !78 = !DILocation(line: 7, column: 3, scope: !71)
// CHECK:STDOUT: !80 = !DILocation(line: 8, column: 6, scope: !71)
// CHECK:STDOUT: !81 = !DILocation(line: 9, column: 5, scope: !71)
// CHECK:STDOUT: !85 = !DILocation(line: 10, column: 5, scope: !71)
// CHECK:STDOUT: !86 = !DILocation(line: 8, column: 3, scope: !71)
// CHECK:STDOUT: !87 = !DILocation(line: 12, column: 5, scope: !71)
// CHECK:STDOUT: !91 = !DILocation(line: 13, column: 5, scope: !71)
// CHECK:STDOUT: !94 = !DILocation(line: 15, column: 3, scope: !71)
// CHECK:STDOUT: !95 = !DILocation(line: 5, column: 1, scope: !71)
// CHECK:STDOUT: !96 = !{!72}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Op", linkageName: "_COp.474c886d55da95e6:core.Destroy.Core", scope: null, file: !69, line: 9, type: !36, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !34)
// CHECK:STDOUT: !99 = !DILocation(line: 9, column: 5, scope: !97)
// CHECK:STDOUT: !100 = !{!98}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_return.carbon'
// CHECK:STDOUT: source_filename = "if_return.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfReturn.Main(i1 %cond) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !106
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !113
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !106
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !106
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !110
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !112
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !113
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !113
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !117
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %b.var), !dbg !113
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %a.var), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %a.var), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %self) #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.474c886d55da95e6:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!102}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !101 = !DIFile(filename: "if_return.carbon", directory: "")
// CHECK:STDOUT: !102 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !101, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "IfReturn", linkageName: "_CIfReturn.Main", scope: null, file: !101, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !102, retainedNodes: !120)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !34)
// CHECK:STDOUT: !106 = !DILocation(line: 6, column: 3, scope: !103)
// CHECK:STDOUT: !110 = !DILocation(line: 7, column: 3, scope: !103)
// CHECK:STDOUT: !112 = !DILocation(line: 8, column: 6, scope: !103)
// CHECK:STDOUT: !113 = !DILocation(line: 9, column: 5, scope: !103)
// CHECK:STDOUT: !117 = !DILocation(line: 10, column: 5, scope: !103)
// CHECK:STDOUT: !118 = !DILocation(line: 11, column: 5, scope: !103)
// CHECK:STDOUT: !119 = !DILocation(line: 5, column: 1, scope: !103)
// CHECK:STDOUT: !120 = !{!104}
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "Op", linkageName: "_COp.474c886d55da95e6:core.Destroy.Core", scope: null, file: !101, line: 9, type: !36, spFlags: DISPFlagDefinition, unit: !102, retainedNodes: !124)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !121, type: !34)
// CHECK:STDOUT: !123 = !DILocation(line: 9, column: 5, scope: !121)
// CHECK:STDOUT: !124 = !{!122}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'while_break.carbon'
// CHECK:STDOUT: source_filename = "while_break.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhileBreak.Main(i1 %cond) #0 !dbg !127 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !130
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !140
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !147
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !130
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !130
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !134
// CHECK:STDOUT:   br label %while.cond, !dbg !135
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %entry
// CHECK:STDOUT:   br i1 %cond, label %while.body, label %while.done, !dbg !135
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !139
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !140
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !140
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !144
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !146
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !147
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !147
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !151
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %c.var), !dbg !147
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %b.var), !dbg !140
// CHECK:STDOUT:   br label %while.done, !dbg !152
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %b.var), !dbg !140
// CHECK:STDOUT:   br label %while.cond, !dbg !153
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %if.then, %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !156
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %a.var), !dbg !130
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %self) #0 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.474c886d55da95e6:core.Destroy.Core", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!126}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !125 = !DIFile(filename: "while_break.carbon", directory: "")
// CHECK:STDOUT: !126 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !125, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "WhileBreak", linkageName: "_CWhileBreak.Main", scope: null, file: !125, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !126, retainedNodes: !158)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !34)
// CHECK:STDOUT: !130 = !DILocation(line: 6, column: 3, scope: !127)
// CHECK:STDOUT: !134 = !DILocation(line: 7, column: 3, scope: !127)
// CHECK:STDOUT: !135 = !DILocation(line: 8, column: 9, scope: !127)
// CHECK:STDOUT: !139 = !DILocation(line: 9, column: 5, scope: !127)
// CHECK:STDOUT: !140 = !DILocation(line: 10, column: 5, scope: !127)
// CHECK:STDOUT: !144 = !DILocation(line: 11, column: 5, scope: !127)
// CHECK:STDOUT: !146 = !DILocation(line: 12, column: 8, scope: !127)
// CHECK:STDOUT: !147 = !DILocation(line: 13, column: 7, scope: !127)
// CHECK:STDOUT: !151 = !DILocation(line: 14, column: 7, scope: !127)
// CHECK:STDOUT: !152 = !DILocation(line: 15, column: 7, scope: !127)
// CHECK:STDOUT: !153 = !DILocation(line: 8, column: 3, scope: !127)
// CHECK:STDOUT: !156 = !DILocation(line: 18, column: 3, scope: !127)
// CHECK:STDOUT: !157 = !DILocation(line: 5, column: 1, scope: !127)
// CHECK:STDOUT: !158 = !{!128}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "Op", linkageName: "_COp.474c886d55da95e6:core.Destroy.Core", scope: null, file: !125, line: 13, type: !36, spFlags: DISPFlagDefinition, unit: !126, retainedNodes: !162)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !34)
// CHECK:STDOUT: !161 = !DILocation(line: 13, column: 7, scope: !159)
// CHECK:STDOUT: !162 = !{!160}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'while_continue.carbon'
// CHECK:STDOUT: source_filename = "while_continue.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhileContinue.Main(i1 %cond) #0 !dbg !165 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !168
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !178
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !185
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !168
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !168
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !172
// CHECK:STDOUT:   br label %while.cond, !dbg !173
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %if.then, %entry
// CHECK:STDOUT:   br i1 %cond, label %while.body, label %while.done, !dbg !173
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !177
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !178
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !178
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !182
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !184
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !185
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !185
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !189
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %c.var), !dbg !185
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %b.var), !dbg !178
// CHECK:STDOUT:   br label %while.cond, !dbg !190
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %b.var), !dbg !178
// CHECK:STDOUT:   br label %while.cond, !dbg !191
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !194
// CHECK:STDOUT:   call void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %a.var), !dbg !168
// CHECK:STDOUT:   ret void, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.474c886d55da95e6:core.Destroy.Core"(ptr %self) #0 !dbg !197 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CUse.Main, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.474c886d55da95e6:core.Destroy.Core", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!164}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !163 = !DIFile(filename: "while_continue.carbon", directory: "")
// CHECK:STDOUT: !164 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !163, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "WhileContinue", linkageName: "_CWhileContinue.Main", scope: null, file: !163, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !164, retainedNodes: !196)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 1, scope: !165, type: !34)
// CHECK:STDOUT: !168 = !DILocation(line: 6, column: 3, scope: !165)
// CHECK:STDOUT: !172 = !DILocation(line: 7, column: 3, scope: !165)
// CHECK:STDOUT: !173 = !DILocation(line: 8, column: 9, scope: !165)
// CHECK:STDOUT: !177 = !DILocation(line: 9, column: 5, scope: !165)
// CHECK:STDOUT: !178 = !DILocation(line: 10, column: 5, scope: !165)
// CHECK:STDOUT: !182 = !DILocation(line: 11, column: 5, scope: !165)
// CHECK:STDOUT: !184 = !DILocation(line: 12, column: 8, scope: !165)
// CHECK:STDOUT: !185 = !DILocation(line: 13, column: 7, scope: !165)
// CHECK:STDOUT: !189 = !DILocation(line: 14, column: 7, scope: !165)
// CHECK:STDOUT: !190 = !DILocation(line: 15, column: 7, scope: !165)
// CHECK:STDOUT: !191 = !DILocation(line: 8, column: 3, scope: !165)
// CHECK:STDOUT: !194 = !DILocation(line: 18, column: 3, scope: !165)
// CHECK:STDOUT: !195 = !DILocation(line: 5, column: 1, scope: !165)
// CHECK:STDOUT: !196 = !{!166}
// CHECK:STDOUT: !197 = distinct !DISubprogram(name: "Op", linkageName: "_COp.474c886d55da95e6:core.Destroy.Core", scope: null, file: !163, line: 13, type: !36, spFlags: DISPFlagDefinition, unit: !164, retainedNodes: !200)
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 1, scope: !197, type: !34)
// CHECK:STDOUT: !199 = !DILocation(line: 13, column: 7, scope: !197)
// CHECK:STDOUT: !200 = !{!198}
// CHECK:STDOUT:
