; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; IR generated from clang -O0 with:
; struct C {
;   ~C();
; };
; extern bool b;
; void fun4() { b && (C(), 1); }
; __attribute__((always_inline)) C::~C() { }

; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "C"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[C_DTOR_DECL:.*]]:  DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "~C"

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "fun4"

; FIXME: The dtor is inlined into fun4 and should have an inlined_subroutine
; entry. (it may be necessary to put some non-trivial instruction, such as an
; assignment to a global, in the dtor just to ensure its emission/inlining)

; CHECK-NOT: DW_TAG_inlined_subroutine

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name {{.*}} "_ZN1CD1Ev"

; FIXME: But I think where the real issue is for PR20038 is that the D1 ctor, ;
; calling and inlining D2, doesn't end up with an inlined_subroutine. Though this
; might be more the result of a lack of any actual work in D2 (again, could use
; an assignment to global, etc)

; CHECK-NOT: DW_TAG_inlined_subroutine

%struct.C = type { i8 }

@b = external global i8

; Function Attrs: nounwind
define void @_Z4fun4v() #0 {
entry:
  %this.addr.i.i = alloca %struct.C*, align 8, !dbg !21
  %this.addr.i = alloca %struct.C*, align 8
  %agg.tmp.ensured = alloca %struct.C, align 1
  %cleanup.cond = alloca i1
  %0 = load i8* @b, align 1, !dbg !22
  %tobool = trunc i8 %0 to i1, !dbg !22
  store i1 false, i1* %cleanup.cond
  br i1 %tobool, label %land.rhs, label %land.end, !dbg !22

land.rhs:                                         ; preds = %entry
  store i1 true, i1* %cleanup.cond, !dbg !23
  br label %land.end

land.end:                                         ; preds = %land.rhs, %entry
  %1 = phi i1 [ false, %entry ], [ true, %land.rhs ]
  %cleanup.is_active = load i1* %cleanup.cond
  br i1 %cleanup.is_active, label %cleanup.action, label %cleanup.done

cleanup.action:                                   ; preds = %land.end
  store %struct.C* %agg.tmp.ensured, %struct.C** %this.addr.i, align 8
  call void @llvm.dbg.declare(metadata !{%struct.C** %this.addr.i}, metadata !25), !dbg !27
  %this1.i = load %struct.C** %this.addr.i
  store %struct.C* %this1.i, %struct.C** %this.addr.i.i, align 8, !dbg !21
  call void @llvm.dbg.declare(metadata !{%struct.C** %this.addr.i.i}, metadata !28), !dbg !29
  %this1.i.i = load %struct.C** %this.addr.i.i, !dbg !21
  br label %cleanup.done

cleanup.done:                                     ; preds = %cleanup.action, %land.end
  ret void, !dbg !22
}

; Function Attrs: alwaysinline nounwind
define void @_ZN1CD1Ev(%struct.C* %this) unnamed_addr #1 align 2 {
entry:
  %this.addr.i = alloca %struct.C*, align 8, !dbg !21
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.C** %this.addr}, metadata !25), !dbg !27
  %this1 = load %struct.C** %this.addr
  store %struct.C* %this1, %struct.C** %this.addr.i, align 8, !dbg !21
  call void @llvm.dbg.declare(metadata !{%struct.C** %this.addr.i}, metadata !28), !dbg !29
  %this1.i = load %struct.C** %this.addr.i, !dbg !21
  ret void, !dbg !21
}

; Function Attrs: alwaysinline nounwind
define void @_ZN1CD2Ev(%struct.C* %this) unnamed_addr #1 align 2 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.C** %this.addr}, metadata !28), !dbg !30
  %this1 = load %struct.C** %this.addr
  ret void, !dbg !31
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !19}
!llvm.ident = !{!20}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !11, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/<stdin>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"<stdin>", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !5, null, metadata !"C", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !6, i32 0, null, null, metadata !"_ZTS1C"} ; [ DW_TAG_structure_type ] [C] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !"PR20038.cpp", metadata !"/tmp/dbginfo"}
!6 = metadata !{metadata !7}
!7 = metadata !{i32 786478, metadata !5, metadata !"_ZTS1C", metadata !"~C", metadata !"~C", metadata !"", i32 2, metadata !8, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, null, i32 2} ; [ DW_TAG_subprogram ] [line 2] [~C]
!8 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !9, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!9 = metadata !{null, metadata !10}
!10 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1C]
!11 = metadata !{metadata !12, metadata !16, metadata !17}
!12 = metadata !{i32 786478, metadata !5, metadata !13, metadata !"fun4", metadata !"fun4", metadata !"_Z4fun4v", i32 5, metadata !14, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z4fun4v, null, null, metadata !2, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [fun4]
!13 = metadata !{i32 786473, metadata !5}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/PR20038.cpp]
!14 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !15, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = metadata !{null}
!16 = metadata !{i32 786478, metadata !5, metadata !"_ZTS1C", metadata !"~C", metadata !"~C", metadata !"_ZN1CD2Ev", i32 6, metadata !8, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%struct.C*)* @_ZN1CD2Ev, null, metadata !7, metadata !2, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [~C]
!17 = metadata !{i32 786478, metadata !5, metadata !"_ZTS1C", metadata !"~C", metadata !"~C", metadata !"_ZN1CD1Ev", i32 6, metadata !8, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%struct.C*)* @_ZN1CD1Ev, null, metadata !7, metadata !2, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [~C]
!18 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!19 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!20 = metadata !{metadata !"clang version 3.5.0 "}
!21 = metadata !{i32 6, i32 0, metadata !17, null}
!22 = metadata !{i32 5, i32 0, metadata !12, null}
!23 = metadata !{i32 5, i32 0, metadata !24, null}
!24 = metadata !{i32 786443, metadata !5, metadata !12, i32 5, i32 0, i32 1, i32 1} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/PR20038.cpp]
!25 = metadata !{i32 786689, metadata !17, metadata !"this", null, i32 16777216, metadata !26, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!26 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1C]
!27 = metadata !{i32 0, i32 0, metadata !17, null}
!28 = metadata !{i32 786689, metadata !16, metadata !"this", null, i32 16777216, metadata !26, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!29 = metadata !{i32 0, i32 0, metadata !16, metadata !21}
!30 = metadata !{i32 0, i32 0, metadata !16, null}
!31 = metadata !{i32 6, i32 0, metadata !16, null}
