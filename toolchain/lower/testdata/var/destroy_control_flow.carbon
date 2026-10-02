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
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %b.var), !dbg !54
// CHECK:STDOUT:   br label %if.else, !dbg !59
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !62
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %a.var), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !71
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
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", { 1, 0 }
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
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !42, line: 9, type: !36, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !34)
// CHECK:STDOUT: !67 = !DILocation(line: 9, column: 5, scope: !65)
// CHECK:STDOUT: !68 = !{!66}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !42, line: 9, type: !36, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !72)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !34)
// CHECK:STDOUT: !71 = !DILocation(line: 9, column: 5, scope: !69)
// CHECK:STDOUT: !72 = !{!70}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_else.carbon'
// CHECK:STDOUT: source_filename = "if_else.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfElse.Main(i1 %cond) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !78
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !85
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !91
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !78
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !78
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !82
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !84
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !85
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !85
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !89
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %b.var), !dbg !85
// CHECK:STDOUT:   br label %if.done, !dbg !90
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !91
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !91
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !95
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %c.var), !dbg !91
// CHECK:STDOUT:   br label %if.done, !dbg !90
// CHECK:STDOUT:
// CHECK:STDOUT: if.done:                                          ; preds = %if.else, %if.then
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !98
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %a.var), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !107
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
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!74}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !73 = !DIFile(filename: "if_else.carbon", directory: "")
// CHECK:STDOUT: !74 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !73, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "IfElse", linkageName: "_CIfElse.Main", scope: null, file: !73, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !74, retainedNodes: !100)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !34)
// CHECK:STDOUT: !78 = !DILocation(line: 6, column: 3, scope: !75)
// CHECK:STDOUT: !82 = !DILocation(line: 7, column: 3, scope: !75)
// CHECK:STDOUT: !84 = !DILocation(line: 8, column: 6, scope: !75)
// CHECK:STDOUT: !85 = !DILocation(line: 9, column: 5, scope: !75)
// CHECK:STDOUT: !89 = !DILocation(line: 10, column: 5, scope: !75)
// CHECK:STDOUT: !90 = !DILocation(line: 8, column: 3, scope: !75)
// CHECK:STDOUT: !91 = !DILocation(line: 12, column: 5, scope: !75)
// CHECK:STDOUT: !95 = !DILocation(line: 13, column: 5, scope: !75)
// CHECK:STDOUT: !98 = !DILocation(line: 15, column: 3, scope: !75)
// CHECK:STDOUT: !99 = !DILocation(line: 5, column: 1, scope: !75)
// CHECK:STDOUT: !100 = !{!76}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !73, line: 9, type: !36, spFlags: DISPFlagDefinition, unit: !74, retainedNodes: !104)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !34)
// CHECK:STDOUT: !103 = !DILocation(line: 9, column: 5, scope: !101)
// CHECK:STDOUT: !104 = !{!102}
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !73, line: 9, type: !36, spFlags: DISPFlagDefinition, unit: !74, retainedNodes: !108)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !34)
// CHECK:STDOUT: !107 = !DILocation(line: 9, column: 5, scope: !105)
// CHECK:STDOUT: !108 = !{!106}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_return.carbon'
// CHECK:STDOUT: source_filename = "if_return.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfReturn.Main(i1 %cond) #0 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !114
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !121
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !114
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !114
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !118
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !120
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !121
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !121
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !125
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %b.var), !dbg !121
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %a.var), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !126
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %a.var), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !135
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
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!110}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !109 = !DIFile(filename: "if_return.carbon", directory: "")
// CHECK:STDOUT: !110 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !109, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "IfReturn", linkageName: "_CIfReturn.Main", scope: null, file: !109, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !110, retainedNodes: !128)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !34)
// CHECK:STDOUT: !114 = !DILocation(line: 6, column: 3, scope: !111)
// CHECK:STDOUT: !118 = !DILocation(line: 7, column: 3, scope: !111)
// CHECK:STDOUT: !120 = !DILocation(line: 8, column: 6, scope: !111)
// CHECK:STDOUT: !121 = !DILocation(line: 9, column: 5, scope: !111)
// CHECK:STDOUT: !125 = !DILocation(line: 10, column: 5, scope: !111)
// CHECK:STDOUT: !126 = !DILocation(line: 11, column: 5, scope: !111)
// CHECK:STDOUT: !127 = !DILocation(line: 5, column: 1, scope: !111)
// CHECK:STDOUT: !128 = !{!112}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !109, line: 9, type: !36, spFlags: DISPFlagDefinition, unit: !110, retainedNodes: !132)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !34)
// CHECK:STDOUT: !131 = !DILocation(line: 9, column: 5, scope: !129)
// CHECK:STDOUT: !132 = !{!130}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !109, line: 9, type: !36, spFlags: DISPFlagDefinition, unit: !110, retainedNodes: !136)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !34)
// CHECK:STDOUT: !135 = !DILocation(line: 9, column: 5, scope: !133)
// CHECK:STDOUT: !136 = !{!134}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'while_break.carbon'
// CHECK:STDOUT: source_filename = "while_break.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhileBreak.Main(i1 %cond) #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !142
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !152
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !159
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !142
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !142
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !146
// CHECK:STDOUT:   br label %while.cond, !dbg !147
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %entry
// CHECK:STDOUT:   br i1 %cond, label %while.body, label %while.done, !dbg !147
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !151
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !152
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !152
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !156
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !158
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !159
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !159
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !163
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %c.var), !dbg !159
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %b.var), !dbg !152
// CHECK:STDOUT:   br label %while.done, !dbg !164
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %b.var), !dbg !152
// CHECK:STDOUT:   br label %while.cond, !dbg !165
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %if.then, %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !168
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %a.var), !dbg !142
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !177
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
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!138}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !137 = !DIFile(filename: "while_break.carbon", directory: "")
// CHECK:STDOUT: !138 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !137, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "WhileBreak", linkageName: "_CWhileBreak.Main", scope: null, file: !137, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !170)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !34)
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
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !137, line: 13, type: !36, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !174)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !34)
// CHECK:STDOUT: !173 = !DILocation(line: 13, column: 7, scope: !171)
// CHECK:STDOUT: !174 = !{!172}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !137, line: 13, type: !36, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !178)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !34)
// CHECK:STDOUT: !177 = !DILocation(line: 13, column: 7, scope: !175)
// CHECK:STDOUT: !178 = !{!176}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'while_continue.carbon'
// CHECK:STDOUT: source_filename = "while_continue.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhileContinue.Main(i1 %cond) #0 !dbg !181 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !184
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !194
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !201
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !184
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !184
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !188
// CHECK:STDOUT:   br label %while.cond, !dbg !189
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %if.then, %entry
// CHECK:STDOUT:   br i1 %cond, label %while.body, label %while.done, !dbg !189
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !193
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !194
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !194
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !198
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !200
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !201
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !201
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !205
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %c.var), !dbg !201
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %b.var), !dbg !194
// CHECK:STDOUT:   br label %while.cond, !dbg !206
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %b.var), !dbg !194
// CHECK:STDOUT:   br label %while.cond, !dbg !207
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !210
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %a.var), !dbg !184
// CHECK:STDOUT:   ret void, !dbg !211
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !213 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !215
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core"(ptr %self) #0 !dbg !217 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !219
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
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!180}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !179 = !DIFile(filename: "while_continue.carbon", directory: "")
// CHECK:STDOUT: !180 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !179, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !181 = distinct !DISubprogram(name: "WhileContinue", linkageName: "_CWhileContinue.Main", scope: null, file: !179, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !180, retainedNodes: !212)
// CHECK:STDOUT: !182 = !DILocalVariable(arg: 1, scope: !181, type: !34)
// CHECK:STDOUT: !184 = !DILocation(line: 6, column: 3, scope: !181)
// CHECK:STDOUT: !188 = !DILocation(line: 7, column: 3, scope: !181)
// CHECK:STDOUT: !189 = !DILocation(line: 8, column: 9, scope: !181)
// CHECK:STDOUT: !193 = !DILocation(line: 9, column: 5, scope: !181)
// CHECK:STDOUT: !194 = !DILocation(line: 10, column: 5, scope: !181)
// CHECK:STDOUT: !198 = !DILocation(line: 11, column: 5, scope: !181)
// CHECK:STDOUT: !200 = !DILocation(line: 12, column: 8, scope: !181)
// CHECK:STDOUT: !201 = !DILocation(line: 13, column: 7, scope: !181)
// CHECK:STDOUT: !205 = !DILocation(line: 14, column: 7, scope: !181)
// CHECK:STDOUT: !206 = !DILocation(line: 15, column: 7, scope: !181)
// CHECK:STDOUT: !207 = !DILocation(line: 8, column: 3, scope: !181)
// CHECK:STDOUT: !210 = !DILocation(line: 18, column: 3, scope: !181)
// CHECK:STDOUT: !211 = !DILocation(line: 5, column: 1, scope: !181)
// CHECK:STDOUT: !212 = !{!182}
// CHECK:STDOUT: !213 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !179, line: 13, type: !36, spFlags: DISPFlagDefinition, unit: !180, retainedNodes: !216)
// CHECK:STDOUT: !214 = !DILocalVariable(arg: 1, scope: !213, type: !34)
// CHECK:STDOUT: !215 = !DILocation(line: 13, column: 7, scope: !213)
// CHECK:STDOUT: !216 = !{!214}
// CHECK:STDOUT: !217 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c3fcb3d93f31df0e:core.Destroy.Core", scope: null, file: !179, line: 13, type: !36, spFlags: DISPFlagDefinition, unit: !180, retainedNodes: !220)
// CHECK:STDOUT: !218 = !DILocalVariable(arg: 1, scope: !217, type: !34)
// CHECK:STDOUT: !219 = !DILocation(line: 13, column: 7, scope: !217)
// CHECK:STDOUT: !220 = !{!218}
// CHECK:STDOUT:
