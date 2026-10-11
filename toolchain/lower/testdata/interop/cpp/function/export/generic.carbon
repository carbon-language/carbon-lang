// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/export/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/export/generic.carbon

// --- generic_type_impls_interface.carbon
library "[[@TEST_NAME]]";
import Cpp;

interface I {
  fn Doit(self);
}

class A {
  impl as I {
    fn Doit(unused self) {}
  }
}
class B {
  impl as I {
    fn Doit(unused self) {}
  }
}

fn F[T: I](t: T) {
  t.Doit();
}

inline Cpp '''
void G() {
  Carbon::A a;
  Carbon::B b;
  Carbon::F(a);
  Carbon::F(b);
}
''';

fn Run() {
  Cpp.G();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_type_impls_interface.carbon'
// CHECK:STDOUT: source_filename = "generic_type_impls_interface.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::A" = type {}
// CHECK:STDOUT: %"class.Carbon::B" = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon1AD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon1BD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z1Gv() #0 personality ptr @__gxx_personality_v0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a = alloca %"class.Carbon::A", align 1
// CHECK:STDOUT:   %b = alloca %"class.Carbon::B", align 1
// CHECK:STDOUT:   %agg.tmp = alloca %"class.Carbon::A", align 1
// CHECK:STDOUT:   %exn.slot = alloca ptr, align 8
// CHECK:STDOUT:   %ehselector.slot = alloca i32, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %"class.Carbon::B", align 1
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a) #4
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b) #4
// CHECK:STDOUT:   invoke void @_ZN6CarbonL1FENS_1AE()
// CHECK:STDOUT:           to label %invoke.cont unwind label %lpad
// CHECK:STDOUT:
// CHECK:STDOUT: invoke.cont:                                      ; preds = %entry
// CHECK:STDOUT:   call void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 1 %agg.tmp) #4
// CHECK:STDOUT:   invoke void @_ZN6CarbonL1FENS_1BE()
// CHECK:STDOUT:           to label %invoke.cont3 unwind label %lpad2
// CHECK:STDOUT:
// CHECK:STDOUT: invoke.cont3:                                     ; preds = %invoke.cont
// CHECK:STDOUT:   call void @_ZN6Carbon1BD2Ev(ptr noundef nonnull align 1 %agg.tmp1) #4
// CHECK:STDOUT:   call void @_ZN6Carbon1BD2Ev(ptr noundef nonnull align 1 %b) #4
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %b) #4
// CHECK:STDOUT:   call void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 1 %a) #4
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %a) #4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT:
// CHECK:STDOUT: lpad:                                             ; preds = %entry
// CHECK:STDOUT:   %0 = landingpad { ptr, i32 }
// CHECK:STDOUT:           cleanup
// CHECK:STDOUT:   %1 = extractvalue { ptr, i32 } %0, 0
// CHECK:STDOUT:   store ptr %1, ptr %exn.slot, align 8
// CHECK:STDOUT:   %2 = extractvalue { ptr, i32 } %0, 1
// CHECK:STDOUT:   store i32 %2, ptr %ehselector.slot, align 4
// CHECK:STDOUT:   call void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 1 %agg.tmp) #4
// CHECK:STDOUT:   br label %ehcleanup
// CHECK:STDOUT:
// CHECK:STDOUT: lpad2:                                            ; preds = %invoke.cont
// CHECK:STDOUT:   %3 = landingpad { ptr, i32 }
// CHECK:STDOUT:           cleanup
// CHECK:STDOUT:   %4 = extractvalue { ptr, i32 } %3, 0
// CHECK:STDOUT:   store ptr %4, ptr %exn.slot, align 8
// CHECK:STDOUT:   %5 = extractvalue { ptr, i32 } %3, 1
// CHECK:STDOUT:   store i32 %5, ptr %ehselector.slot, align 4
// CHECK:STDOUT:   call void @_ZN6Carbon1BD2Ev(ptr noundef nonnull align 1 %agg.tmp1) #4
// CHECK:STDOUT:   br label %ehcleanup
// CHECK:STDOUT:
// CHECK:STDOUT: ehcleanup:                                        ; preds = %lpad2, %lpad
// CHECK:STDOUT:   call void @_ZN6Carbon1BD2Ev(ptr noundef nonnull align 1 %b) #4
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %b) #4
// CHECK:STDOUT:   call void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 1 %a) #4
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %a) #4
// CHECK:STDOUT:   br label %eh.resume
// CHECK:STDOUT:
// CHECK:STDOUT: eh.resume:                                        ; preds = %ehcleanup
// CHECK:STDOUT:   %exn = load ptr, ptr %exn.slot, align 8
// CHECK:STDOUT:   %sel = load i32, ptr %ehselector.slot, align 4
// CHECK:STDOUT:   %lpad.val = insertvalue { ptr, i32 } poison, ptr %exn, 0
// CHECK:STDOUT:   %lpad.val7 = insertvalue { ptr, i32 } %lpad.val, i32 %sel, 1
// CHECK:STDOUT:   resume { ptr, i32 } %lpad.val7
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder ptr %a, { 0, 2, 1, 3, 4 }
// CHECK:STDOUT:   uselistorder ptr %b, { 0, 2, 1, 3, 4 }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL1FENS_1AE() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %0 = alloca %"class.Carbon::A", align 1
// CHECK:STDOUT:   call void @_CF__carbon_thunk.26f2099a79512a5d.Main(ptr noundef nonnull align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @__gxx_personality_v0(...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 1 %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !83
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.A.Main"(ptr noundef nonnull align 1 %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL1FENS_1BE() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %0 = alloca %"class.Carbon::B", align 1
// CHECK:STDOUT:   call void @_CF__carbon_thunk.7a3b9f04ddf4269e.Main(ptr noundef nonnull align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon1BD2Ev(ptr noundef nonnull align 1 %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !85
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.B.Main"(ptr noundef nonnull align 1 %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CDoit.A.Main:I.Main"(ptr %self) #4 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CDoit.B.Main:I.Main"(ptr %self) #4 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.A.Main"(ptr %self) #5 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1ff1de8b251403c3(ptr %self), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.B.Main"(ptr %self) #5 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %self), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.26f2099a79512a5d.Main(ptr %_) #4 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.6662a1ffb773360a(ptr %_), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.7a3b9f04ddf4269e.Main(ptr %_) #4 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.d53b17eed3ce903c(ptr %_), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #4 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z1Gv(), !dbg !51
// CHECK:STDOUT:   ret i32 0, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.1ff1de8b251403c3(ptr %t) #4 !dbg !53 {
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %t) #4 !dbg !60 {
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.6662a1ffb773360a(ptr %t) #4 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CDoit.A.Main:I.Main"(ptr %t), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.d53b17eed3ce903c(ptr %t) #4 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CDoit.B.Main:I.Main"(ptr %t), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon1AD2Ev, { 0, 2, 1, 3 }
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon1BD2Ev, { 0, 2, 1, 3 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.end.p0, { 3, 1, 2, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nounwind }
// CHECK:STDOUT: attributes #5 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!86, !87, !88, !4, !5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!91}
// CHECK:STDOUT:
// CHECK:STDOUT: !4 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !5 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIFile(filename: "generic_type_impls_interface.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !20 = !{!19}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Doit", linkageName: "_CDoit.A.Main:I.Main", scope: null, file: !14, line: 10, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !28)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !16)
// CHECK:STDOUT: !27 = !DILocation(line: 10, column: 5, scope: !24)
// CHECK:STDOUT: !28 = !{!25}
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Doit", linkageName: "_CDoit.B.Main:I.Main", scope: null, file: !14, line: 15, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !33)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !29, type: !16)
// CHECK:STDOUT: !32 = !DILocation(line: 15, column: 5, scope: !29)
// CHECK:STDOUT: !33 = !{!30}
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.A.Main", scope: null, file: !14, line: 8, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !37)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !16)
// CHECK:STDOUT: !36 = !DILocation(line: 8, column: 1, scope: !34)
// CHECK:STDOUT: !37 = !{!35}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.B.Main", scope: null, file: !14, line: 13, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !41)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !16)
// CHECK:STDOUT: !40 = !DILocation(line: 13, column: 1, scope: !38)
// CHECK:STDOUT: !41 = !{!39}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "F__carbon_thunk.26f2099a79512a5d", linkageName: "_CF__carbon_thunk.26f2099a79512a5d.Main", scope: null, file: !14, line: 19, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !16)
// CHECK:STDOUT: !44 = !DILocation(line: 19, column: 1, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "F__carbon_thunk.7a3b9f04ddf4269e", linkageName: "_CF__carbon_thunk.7a3b9f04ddf4269e.Main", scope: null, file: !14, line: 19, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !49)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !16)
// CHECK:STDOUT: !48 = !DILocation(line: 19, column: 1, scope: !46)
// CHECK:STDOUT: !49 = !{!47}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !14, line: 32, type: !21, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !51 = !DILocation(line: 33, column: 3, scope: !50)
// CHECK:STDOUT: !52 = !DILocation(line: 32, column: 1, scope: !50)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.1ff1de8b251403c3", scope: null, file: !6, line: 26, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !59)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !16)
// CHECK:STDOUT: !58 = !DILocation(line: 26, column: 1, scope: !53)
// CHECK:STDOUT: !59 = !{!54}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c8fa36f2e394941a", scope: null, file: !6, line: 26, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !66)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !16)
// CHECK:STDOUT: !65 = !DILocation(line: 26, column: 1, scope: !60)
// CHECK:STDOUT: !66 = !{!61}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.6662a1ffb773360a", scope: null, file: !14, line: 19, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !72)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !16)
// CHECK:STDOUT: !70 = !DILocation(line: 20, column: 3, scope: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 19, column: 1, scope: !67)
// CHECK:STDOUT: !72 = !{!68}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.d53b17eed3ce903c", scope: null, file: !14, line: 19, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !78)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !16)
// CHECK:STDOUT: !76 = !DILocation(line: 20, column: 3, scope: !73)
// CHECK:STDOUT: !77 = !DILocation(line: 19, column: 1, scope: !73)
// CHECK:STDOUT: !78 = !{!74}
// CHECK:STDOUT: !79 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !80 = !{!"omnipotent char", !79, i64 0}
// CHECK:STDOUT: !81 = !{!"any pointer", !80, i64 0}
// CHECK:STDOUT: !82 = !{!"p1 _ZTSN6Carbon1AE", !81, i64 0}
// CHECK:STDOUT: !83 = !{!82, !82, i64 0}
// CHECK:STDOUT: !84 = !{!"p1 _ZTSN6Carbon1BE", !81, i64 0}
// CHECK:STDOUT: !85 = !{!84, !84, i64 0}
// CHECK:STDOUT: !86 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !87 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !88 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !89 = !{!"int", !80, i64 0}
// CHECK:STDOUT: !90 = !{!"__libc_errno", !89, i64 0}
// CHECK:STDOUT: !91 = !{!90, !89, i64 0}
// CHECK:STDOUT:
