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
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Destroy.3fda586d04e243ea.Core:Destructible.Main:Destroy.Core"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Destructible.Main:Destroy.Core"(ptr %self), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "destructible.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Destroy.3fda586d04e243ea.Core:Destructible.Main:Destroy.Core", scope: null, file: !1, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 5, scope: !4)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_then.carbon'
// CHECK:STDOUT: source_filename = "if_then.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfThen.Main(i1 %cond) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !10
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !10
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !12
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !13
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !11
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !11
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !14
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %b.var), !dbg !11
// CHECK:STDOUT:   br label %if.else, !dbg !15
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !16
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %a.var), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %self) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
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
// CHECK:STDOUT: uselistorder ptr @"_COp.eef1c8b64ca90051:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "if_then.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "IfThen", linkageName: "_CIfThen.Main", scope: null, file: !1, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 9, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 8, column: 6, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 10, column: 5, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 8, column: 3, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 12, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 5, column: 1, scope: !4)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Op", linkageName: "_COp.eef1c8b64ca90051:core.Destroy.Core", scope: null, file: !1, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !18, type: !7)
// CHECK:STDOUT: !21 = !DILocation(line: 9, column: 5, scope: !18)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_else.carbon'
// CHECK:STDOUT: source_filename = "if_else.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfElse.Main(i1 %cond) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !10
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !11
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !10
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !13
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !11
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !11
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !15
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %b.var), !dbg !11
// CHECK:STDOUT:   br label %if.done, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !12
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !12
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !17
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %c.var), !dbg !12
// CHECK:STDOUT:   br label %if.done, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: if.done:                                          ; preds = %if.else, %if.then
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !18
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %a.var), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
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
// CHECK:STDOUT: uselistorder ptr @"_COp.eef1c8b64ca90051:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "if_else.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "IfElse", linkageName: "_CIfElse.Main", scope: null, file: !1, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 9, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 12, column: 5, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 8, column: 6, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 10, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 8, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 13, column: 5, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 5, column: 1, scope: !4)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Op", linkageName: "_COp.eef1c8b64ca90051:core.Destroy.Core", scope: null, file: !1, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !21)
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !20, type: !7)
// CHECK:STDOUT: !23 = !DILocation(line: 9, column: 5, scope: !20)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'if_return.carbon'
// CHECK:STDOUT: source_filename = "if_return.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIfReturn.Main(i1 %cond) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !10
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !10
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !12
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !13
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !11
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !11
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !14
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %b.var), !dbg !11
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %a.var), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %a.var), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !20
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
// CHECK:STDOUT: uselistorder ptr @"_COp.eef1c8b64ca90051:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "if_return.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "IfReturn", linkageName: "_CIfReturn.Main", scope: null, file: !1, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 9, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 8, column: 6, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 10, column: 5, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 11, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 5, column: 1, scope: !4)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Op", linkageName: "_COp.eef1c8b64ca90051:core.Destroy.Core", scope: null, file: !1, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !18)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !17, type: !7)
// CHECK:STDOUT: !20 = !DILocation(line: 9, column: 5, scope: !17)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'while_break.carbon'
// CHECK:STDOUT: source_filename = "while_break.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhileBreak.Main(i1 %cond) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !10
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !11
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !10
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !13
// CHECK:STDOUT:   br label %while.cond, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %entry
// CHECK:STDOUT:   br i1 %cond, label %while.body, label %while.done, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !11
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !11
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !16
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !17
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !12
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !12
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !18
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %c.var), !dbg !12
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %b.var), !dbg !11
// CHECK:STDOUT:   br label %while.done, !dbg !19
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %b.var), !dbg !11
// CHECK:STDOUT:   br label %while.cond, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %if.then, %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !21
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %a.var), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
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
// CHECK:STDOUT: uselistorder ptr @"_COp.eef1c8b64ca90051:core.Destroy.Core", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "while_break.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "WhileBreak", linkageName: "_CWhileBreak.Main", scope: null, file: !1, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 10, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 13, column: 7, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 8, column: 9, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 9, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 11, column: 5, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 12, column: 8, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 14, column: 7, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 15, column: 7, scope: !4)
// CHECK:STDOUT: !20 = !DILocation(line: 8, column: 3, scope: !4)
// CHECK:STDOUT: !21 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !22 = !DILocation(line: 5, column: 1, scope: !4)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Op", linkageName: "_COp.eef1c8b64ca90051:core.Destroy.Core", scope: null, file: !1, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !23, type: !7)
// CHECK:STDOUT: !26 = !DILocation(line: 13, column: 7, scope: !23)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'while_continue.carbon'
// CHECK:STDOUT: source_filename = "while_continue.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Destructible.val.loc6_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhileContinue.Main(i1 %cond) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !10
// CHECK:STDOUT:   %b.var = alloca {}, align 1, !dbg !11
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !10
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !13
// CHECK:STDOUT:   br label %while.cond, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %if.then, %entry
// CHECK:STDOUT:   br i1 %cond, label %while.body, label %while.done, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !11
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !11
// CHECK:STDOUT:   call void @_CUse.Main(ptr %b.var), !dbg !16
// CHECK:STDOUT:   br i1 %cond, label %if.then, label %if.else, !dbg !17
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !12
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @Destructible.val.loc6_3, i64 0, i1 false), !dbg !12
// CHECK:STDOUT:   call void @_CUse.Main(ptr %c.var), !dbg !18
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %c.var), !dbg !12
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %b.var), !dbg !11
// CHECK:STDOUT:   br label %while.cond, !dbg !19
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.body
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %b.var), !dbg !11
// CHECK:STDOUT:   br label %while.cond, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CUse.Main(ptr %a.var), !dbg !21
// CHECK:STDOUT:   call void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %a.var), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.eef1c8b64ca90051:core.Destroy.Core"(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
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
// CHECK:STDOUT: uselistorder ptr @"_COp.eef1c8b64ca90051:core.Destroy.Core", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "while_continue.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "WhileContinue", linkageName: "_CWhileContinue.Main", scope: null, file: !1, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 10, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 13, column: 7, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 8, column: 9, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 9, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 11, column: 5, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 12, column: 8, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 14, column: 7, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 15, column: 7, scope: !4)
// CHECK:STDOUT: !20 = !DILocation(line: 8, column: 3, scope: !4)
// CHECK:STDOUT: !21 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !22 = !DILocation(line: 5, column: 1, scope: !4)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Op", linkageName: "_COp.eef1c8b64ca90051:core.Destroy.Core", scope: null, file: !1, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !23, type: !7)
// CHECK:STDOUT: !26 = !DILocation(line: 13, column: 7, scope: !23)
// CHECK:STDOUT:
