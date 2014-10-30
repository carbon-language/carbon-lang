; REQUIRES: object-emission

; RUN: %llc_dwarf < %s -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Built with clang from the following source:
; void f1(int);
; __attribute__((always_inline)) inline void f2(int i) { f1(i); }
;
; #define MULTICALL \
;   f2(x);          \
;   f2(y);
;
; void f3(int x, int y) { MULTICALL; }

; FIXME: This produces only one inlined_subroutine, with two formal_parameters
; (both named "this"), one for each of the actual inlined subroutines.  ;
; Inlined scopes are differentiated by the combination of 'inlined at' (call)
; location and the location within the function. If two calls to the same
; function occur at the same location the scopes end up conflated and there
; appears to be only one inlined function.
; To fix this, we'd need to add some kind of unique metadata per call site, possibly something like:
;
; !42 = metadata !{i32 1, i32 0, metadata !43, metadata !44}
; !44 = metadata !{i32 2, i32 0, metadata !45, null}
;
; ->
;
; !42 = metadata !{i32 1, i32 0, metadata !43, metadata !44}
; !44 = metadata !{metadata !45, metadata !44}
; !45 = metadata !{i32 2, i32 0, metadata !45, null}
;
; since cycles in metadata are not uniqued, the !44 node would not be shared
; between calls to the same function from the same location, ensuring separate
; inlined subroutines would be generated.
;
; Once this is done, the (insufficient) hack in clang that adds column
; information to call sites to differentiate inlined callers can be removed as it
; will no longer be necessary.
;
; While it might be nice to omit the duplicate parameter in this case (while
; we wait/work on the real fix), it's actually better to leave it in because it
; allows us to hold the invariant that every DbgVariable has a DIE, every time.
; This has proved valuable in finding other bugs, so I want to avoid removing the
; invariant/assertion. Besides, we don't know which one's the right one anyway...

; CHECK: DW_TAG_subprogram
; CHECK:   DW_TAG_inlined_subroutine
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     NULL
; CHECK-NOT: DW_TAG
; CHECK:   NULL

; Function Attrs: uwtable
define void @_Z2f3ii(i32 %x, i32 %y) #0 {
entry:
  %i.addr.i1 = alloca i32, align 4
  %i.addr.i = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %x.addr}, metadata !15, metadata !16), !dbg !17
  store i32 %y, i32* %y.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %y.addr}, metadata !18, metadata !16), !dbg !19
  %0 = load i32* %x.addr, align 4, !dbg !20
  store i32 %0, i32* %i.addr.i, align 4, !dbg !20
  call void @llvm.dbg.declare(metadata !{i32* %i.addr.i}, metadata !21, metadata !16), !dbg !22
  %1 = load i32* %i.addr.i, align 4, !dbg !23
  call void @_Z2f1i(i32 %1), !dbg !23
  %2 = load i32* %y.addr, align 4, !dbg !20
  store i32 %2, i32* %i.addr.i1, align 4, !dbg !20
  call void @llvm.dbg.declare(metadata !{i32* %i.addr.i1}, metadata !21, metadata !16), !dbg !22
  %3 = load i32* %i.addr.i1, align 4, !dbg !23
  call void @_Z2f1i(i32 %3), !dbg !23
  ret void, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @_Z2f1i(i32) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = metadata !{metadata !"0x11\004\00clang version 3.6.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/duplicate_inline.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"duplicate_inline.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !9}
!4 = metadata !{metadata !"0x2e\00f3\00f3\00_Z2f3ii\008\000\001\000\000\00256\000\008", metadata !1, metadata !5, metadata !6, null, void (i32, i32)* @_Z2f3ii, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 8] [def] [f3]
!5 = metadata !{metadata !"0x29", metadata !1}    ; [ DW_TAG_file_type ] [/tmp/dbginfo/duplicate_inline.cpp]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8, metadata !8}
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !"0x2e\00f2\00f2\00_Z2f2i\002\000\001\000\000\00256\000\002", metadata !1, metadata !5, metadata !10, null, null, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 2] [def] [f2]
!10 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !11, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = metadata !{null, metadata !8}
!12 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!13 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!14 = metadata !{metadata !"clang version 3.6.0 "}
!15 = metadata !{metadata !"0x101\00x\0016777224\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [x] [line 8]
!16 = metadata !{metadata !"0x102"}               ; [ DW_TAG_expression ]
!17 = metadata !{i32 8, i32 13, metadata !4, null}
!18 = metadata !{metadata !"0x101\00y\0033554440\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [y] [line 8]
!19 = metadata !{i32 8, i32 20, metadata !4, null}
!20 = metadata !{i32 8, i32 25, metadata !4, null}
!21 = metadata !{metadata !"0x101\00i\0016777218\000", metadata !9, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [i] [line 2]
!22 = metadata !{i32 2, i32 51, metadata !9, metadata !20}
!23 = metadata !{i32 2, i32 56, metadata !9, metadata !20}
!24 = metadata !{i32 8, i32 36, metadata !4, null}
