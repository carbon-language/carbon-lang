; RUN: llc -filetype=obj -o %t.o %s
; RUN: llvm-dwarfdump -debug-dump=info %t.o | FileCheck %s

; Testcase generated using 'clang -O2 -S -emit-llvm' from the following:
;; int *g;
;;
;; static __attribute__((always_inline)) int f(int a) {
;;   int l;
;;   g = &l;
;;   return a;
;; }
;;
;; int main(void) {
;;   f(0);
;;   f(0);
;;   return 0;
;; }

; Check that we the first call to f(0) has no inlined subroutine (since the
; function is optimized out), and the second call correctly describes the
; formal parameter 'a'.

; CHECK:       DW_TAG_inlined_subroutine
; CHECK-NOT:   DW_TAG_inlined_subroutine
; CHECK:         DW_AT_low_pc
; CHECK-NEXT:    DW_AT_high_pc
; CHECK-NOT:   DW_TAG_inlined_subroutine
; CHECK:         DW_TAG_formal_parameter
; CHECK-NEXT:      DW_AT_const_value
; CHECK-NEXT:      DW_AT_abstract_origin {{.*}} "a"
; CHECK-NOT:   DW_TAG_inlined_subroutine
; CHECK:         DW_TAG_variable
; CHECK-NEXT:      DW_AT_location
; CHECK-NEXT:      DW_AT_abstract_origin {{.*}} "l"

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

@g = common global i32* null, align 8

; Function Attrs: nounwind ssp uwtable
define i32 @main() #0 {
entry:
  %l.i2 = alloca i32, align 4
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !12, metadata !21), !dbg !22
  %0 = bitcast i32* %l.i2 to i8*, !dbg !24
  call void @llvm.lifetime.start(i64 4, i8* %0), !dbg !24
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !12, metadata !21), !dbg !24
  tail call void @llvm.dbg.value(metadata i32* %l.i2, i64 0, metadata !13, metadata !21), !dbg !26
  store i32* %l.i2, i32** @g, align 8, !dbg !27, !tbaa !28
  call void @llvm.lifetime.end(i64 4, i8* %0), !dbg !32
  ret i32 0, !dbg !33
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #2

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #2

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18, !19}
!llvm.ident = !{!20}

!0 = !MDCompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (trunk 235110) (llvm/trunk 235108)", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !14, imports: !2)
!1 = !MDFile(filename: "t.c", directory: "/path/to/dir")
!2 = !{}
!3 = !{!4, !8}
!4 = !MDSubprogram(name: "main", scope: !1, file: !1, line: 9, type: !5, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: true, function: i32 ()* @main, variables: !2)
!5 = !MDSubroutineType(types: !6)
!6 = !{!7}
!7 = !MDBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !MDSubprogram(name: "f", scope: !1, file: !1, line: 3, type: !9, isLocal: true, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, variables: !11)
!9 = !MDSubroutineType(types: !10)
!10 = !{!7, !7}
!11 = !{!12, !13}
!12 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "a", arg: 1, scope: !8, file: !1, line: 3, type: !7)
!13 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "l", scope: !8, file: !1, line: 4, type: !7)
!14 = !{!15}
!15 = !MDGlobalVariable(name: "g", scope: !0, file: !1, line: 1, type: !16, isLocal: false, isDefinition: true, variable: i32** @g)
!16 = !MDDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64)
!17 = !{i32 2, !"Dwarf Version", i32 2}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"PIC Level", i32 2}
!20 = !{!"clang version 3.7.0 (trunk 235110) (llvm/trunk 235108)"}
!21 = !MDExpression()
!22 = !MDLocation(line: 3, column: 49, scope: !8, inlinedAt: !23)
!23 = distinct !MDLocation(line: 10, column: 3, scope: !4)
!24 = !MDLocation(line: 3, column: 49, scope: !8, inlinedAt: !25)
!25 = distinct !MDLocation(line: 11, column: 3, scope: !4)
!26 = !MDLocation(line: 4, column: 7, scope: !8, inlinedAt: !25)
!27 = !MDLocation(line: 5, column: 5, scope: !8, inlinedAt: !25)
!28 = !{!29, !29, i64 0}
!29 = !{!"any pointer", !30, i64 0}
!30 = !{!"omnipotent char", !31, i64 0}
!31 = !{!"Simple C/C++ TBAA"}
!32 = !MDLocation(line: 11, column: 3, scope: !4)
!33 = !MDLocation(line: 12, column: 3, scope: !4)
