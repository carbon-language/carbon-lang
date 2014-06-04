; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; The formal parameter 'b' for Function 'x' when inlined within 'a' is lost on
; mips and powerpc64 (and on x86_64 at at least -O2). Presumably this is a
; SelectionDAG issue (do mips/powerpc64 use FastISel?).
; XFAIL: mips, powerpc64

; Build from the following source with clang -O2.

; The important details are that 'x's abstract definition is first built during
; the definition of 'b', where the parameter to 'x' is constant and so 'x's 's'
; variable is optimized away. No abstract definition DIE for 's' is constructed.
; Then, during 'a' emission, the abstract DbgVariable for 's' is created, but
; the abstract DIE isn't (since the abstract definition for 'b' is already
; built). This results in 's' inlined in 'a' being emitted with its name, line,
; file there, rather than referencing an abstract definition.

; extern int t;
;
; void f(int);
;
; inline void x(bool b) {
;   if (b) {
;     int s = t;
;     f(s);
;   }
;   f(0);
; }
;
; void b() {
;   x(false);
; }
;
; void a(bool u) {
;   x(u);
; }

; CHECK: [[ABS_X:.*]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "x"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[ABS_B:.*]]:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "b"
; FIXME: Missing 'x's local 's' variable.

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "b"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_inlined_subroutine
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} {[[ABS_X]]}
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_abstract_origin {{.*}} {[[ABS_B]]}
; Notice 'x's local variable 's' is missing. Not necessarily a bug here,
; since it's been optimized entirely away and it should be described in
; abstract subprogram.
; CHECK-NOT: DW_TAG
; CHECK: NULL
; CHECK-NOT: DW_TAG
; CHECK: NULL

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "a"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_inlined_subroutine
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} {[[ABS_X]]}
; CHECK-NOT: {{DW_TAG|NULL}}
; FIXME: This formal parameter goes missing at least at -O2, maybe before that.
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_abstract_origin {{.*}} {[[ABS_B]]}

; The two lexical blocks here are caused by the scope of the if that includes
; the condition variable, and the scope within the if's composite statement. I'm
; not sure we really need both of them.

; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_lexical_block
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:       DW_TAG_lexical_block
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:         DW_TAG_variable
; CHECK-NOT: DW_TAG

; FIXME: This shouldn't have a name here, it should use DW_AT_abstract_origin
; to reference an abstract variable definition instead

; CHECK:           DW_AT_name {{.*}} "s"


@t = external global i32

; Function Attrs: uwtable
define void @_Z1bv() #0 {
entry:
  tail call void @llvm.dbg.value(metadata !24, i64 0, metadata !25), !dbg !27
  tail call void @_Z1fi(i32 0), !dbg !28
  ret void, !dbg !29
}

; Function Attrs: uwtable
define void @_Z1ab(i1 zeroext %u) #0 {
entry:
  tail call void @llvm.dbg.value(metadata !{i1 %u}, i64 0, metadata !13), !dbg !30
  tail call void @llvm.dbg.value(metadata !{i1 %u}, i64 0, metadata !31), !dbg !33
  br i1 %u, label %if.then.i, label %_Z1xb.exit, !dbg !34

if.then.i:                                        ; preds = %entry
  %0 = load i32* @t, align 4, !dbg !35, !tbaa !36
  tail call void @llvm.dbg.value(metadata !{i32 %0}, i64 0, metadata !40), !dbg !35
  tail call void @_Z1fi(i32 %0), !dbg !41
  br label %_Z1xb.exit, !dbg !42

_Z1xb.exit:                                       ; preds = %entry, %if.then.i
  tail call void @_Z1fi(i32 0), !dbg !43
  ret void, !dbg !44
}

declare void @_Z1fi(i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/missing-abstract-variables.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"missing-abstract-variables.cc", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !8, metadata !14}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"b", metadata !"b", metadata !"_Z1bv", i32 13, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void ()* @_Z1bv, null, null, metadata !2, i32 13} ; [ DW_TAG_subprogram ] [line 13] [def] [b]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/missing-abstract-variables.cc]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"a", metadata !"a", metadata !"_Z1ab", i32 17, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (i1)* @_Z1ab, null, null, metadata !12, i32 17} ; [ DW_TAG_subprogram ] [line 17] [def] [a]
!9 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{null, metadata !11}
!11 = metadata !{i32 786468, null, null, metadata !"bool", i32 0, i64 8, i64 8, i64 0, i32 0, i32 2} ; [ DW_TAG_base_type ] [bool] [line 0, size 8, align 8, offset 0, enc DW_ATE_boolean]
!12 = metadata !{metadata !13}
!13 = metadata !{i32 786689, metadata !8, metadata !"u", metadata !5, i32 16777233, metadata !11, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [u] [line 17]
!14 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"x", metadata !"x", metadata !"_Z1xb", i32 5, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, null, null, metadata !15, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [x]
!15 = metadata !{metadata !16, metadata !17}
!16 = metadata !{i32 786689, metadata !14, metadata !"b", metadata !5, i32 16777221, metadata !11, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [b] [line 5]
!17 = metadata !{i32 786688, metadata !18, metadata !"s", metadata !5, i32 7, metadata !20, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [s] [line 7]
!18 = metadata !{i32 786443, metadata !1, metadata !19, i32 6, i32 0, i32 0, i32 1} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/missing-abstract-variables.cc]
!19 = metadata !{i32 786443, metadata !1, metadata !14, i32 6, i32 0, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/missing-abstract-variables.cc]
!20 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!21 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!22 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!23 = metadata !{metadata !"clang version 3.5.0 "}
!24 = metadata !{i1 false}
!25 = metadata !{i32 786689, metadata !14, metadata !"b", metadata !5, i32 16777221, metadata !11, i32 0, metadata !26} ; [ DW_TAG_arg_variable ] [b] [line 5]
!26 = metadata !{i32 14, i32 0, metadata !4, null}
!27 = metadata !{i32 5, i32 0, metadata !14, metadata !26}
!28 = metadata !{i32 10, i32 0, metadata !14, metadata !26}
!29 = metadata !{i32 15, i32 0, metadata !4, null}
!30 = metadata !{i32 17, i32 0, metadata !8, null}
!31 = metadata !{i32 786689, metadata !14, metadata !"b", metadata !5, i32 16777221, metadata !11, i32 0, metadata !32} ; [ DW_TAG_arg_variable ] [b] [line 5]
!32 = metadata !{i32 18, i32 0, metadata !8, null}
!33 = metadata !{i32 5, i32 0, metadata !14, metadata !32}
!34 = metadata !{i32 6, i32 0, metadata !19, metadata !32}
!35 = metadata !{i32 7, i32 0, metadata !18, metadata !32}
!36 = metadata !{metadata !37, metadata !37, i64 0}
!37 = metadata !{metadata !"int", metadata !38, i64 0}
!38 = metadata !{metadata !"omnipotent char", metadata !39, i64 0}
!39 = metadata !{metadata !"Simple C/C++ TBAA"}
!40 = metadata !{i32 786688, metadata !18, metadata !"s", metadata !5, i32 7, metadata !20, i32 0, metadata !32} ; [ DW_TAG_auto_variable ] [s] [line 7]
!41 = metadata !{i32 8, i32 0, metadata !18, metadata !32} ; [ DW_TAG_imported_declaration ]
!42 = metadata !{i32 9, i32 0, metadata !18, metadata !32}
!43 = metadata !{i32 10, i32 0, metadata !14, metadata !32}
!44 = metadata !{i32 19, i32 0, metadata !8, null}
