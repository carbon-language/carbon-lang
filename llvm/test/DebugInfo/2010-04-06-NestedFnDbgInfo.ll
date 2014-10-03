; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj -o - < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; Radar 7833483
; Do not emit a separate out-of-line definition DIE for the function-local 'foo'
; function (member of the function local 'A' type)
; CHECK: DW_TAG_class_type
; CHECK: DW_TAG_class_type
; CHECK-NEXT: DW_AT_name {{.*}} "A"
; Check that the subprogram inside the class definition has low_pc, only
; attached to the definition.
; CHECK: [[FOO_INL:0x........]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_low_pc
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name {{.*}} "_ZZN1B2fnEvEN1A3fooEv"
; And just double check that there's no out of line definition that references
; this subprogram.
; CHECK-NOT: DW_AT_specification {{.*}} {[[FOO_INL]]}

%class.A = type { i8 }
%class.B = type { i8 }

define i32 @main() ssp {
entry:
  %retval = alloca i32, align 4                   ; <i32*> [#uses=3]
  %b = alloca %class.A, align 1                   ; <%class.A*> [#uses=1]
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata !{%class.A* %b}, metadata !0, metadata !{metadata !"0x102"}), !dbg !14
  %call = call i32 @_ZN1B2fnEv(%class.A* %b), !dbg !15 ; <i32> [#uses=1]
  store i32 %call, i32* %retval, !dbg !15
  %0 = load i32* %retval, !dbg !16                ; <i32> [#uses=1]
  ret i32 %0, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr i32 @_ZN1B2fnEv(%class.A* %this) ssp align 2 {
entry:
  %retval = alloca i32, align 4                   ; <i32*> [#uses=2]
  %this.addr = alloca %class.A*, align 8          ; <%class.A**> [#uses=2]
  %a = alloca %class.A, align 1                   ; <%class.A*> [#uses=1]
  %i = alloca i32, align 4                        ; <i32*> [#uses=2]
  store %class.A* %this, %class.A** %this.addr
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !17, metadata !{metadata !"0x102"}), !dbg !18
  %this1 = load %class.A** %this.addr             ; <%class.A*> [#uses=0]
  call void @llvm.dbg.declare(metadata !{%class.A* %a}, metadata !19, metadata !{metadata !"0x102"}), !dbg !27
  call void @llvm.dbg.declare(metadata !{i32* %i}, metadata !28, metadata !{metadata !"0x102"}), !dbg !29
  %call = call i32 @_ZZN1B2fnEvEN1A3fooEv(%class.A* %a), !dbg !30 ; <i32> [#uses=1]
  store i32 %call, i32* %i, !dbg !30
  %tmp = load i32* %i, !dbg !31                   ; <i32> [#uses=1]
  store i32 %tmp, i32* %retval, !dbg !31
  %0 = load i32* %retval, !dbg !32                ; <i32> [#uses=1]
  ret i32 %0, !dbg !32
}

define internal i32 @_ZZN1B2fnEvEN1A3fooEv(%class.A* %this) ssp align 2 {
entry:
  %retval = alloca i32, align 4                   ; <i32*> [#uses=2]
  %this.addr = alloca %class.A*, align 8          ; <%class.A**> [#uses=2]
  store %class.A* %this, %class.A** %this.addr
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !33, metadata !{metadata !"0x102"}), !dbg !34
  %this1 = load %class.A** %this.addr             ; <%class.A*> [#uses=0]
  store i32 42, i32* %retval, !dbg !35
  %0 = load i32* %retval, !dbg !35                ; <i32> [#uses=1]
  ret i32 %0, !dbg !35
}

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!40}
!37 = metadata !{metadata !2, metadata !10, metadata !23}

!0 = metadata !{metadata !"0x100\00b\0016\000", metadata !1, metadata !3, metadata !8} ; [ DW_TAG_auto_variable ]
!1 = metadata !{metadata !"0xb\0015\0012\000", metadata !38, metadata !2} ; [ DW_TAG_lexical_block ]
!2 = metadata !{metadata !"0x2e\00main\00main\00main\0015\000\001\000\006\000\000\0015", metadata !38, metadata !3, metadata !5, null, i32 ()* @main, null, null, null} ; [ DW_TAG_subprogram ]
!3 = metadata !{metadata !"0x29", metadata !38} ; [ DW_TAG_file_type ]
!4 = metadata !{metadata !"0x11\004\00clang 1.5\000\00\000\00\000", metadata !38, metadata !39, metadata !39, metadata !37, null,  null} ; [ DW_TAG_compile_unit ]
!5 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !38, metadata !3, null, metadata !6, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = metadata !{metadata !7}
!7 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !38, metadata !3} ; [ DW_TAG_base_type ]
!8 = metadata !{metadata !"0x2\00B\002\008\008\000\000\000", metadata !38, metadata !3, null, metadata !9, null, null, null} ; [ DW_TAG_class_type ] [B] [line 2, size 8, align 8, offset 0] [def] [from ]
!9 = metadata !{metadata !10}
!10 = metadata !{metadata !"0x2e\00fn\00fn\00_ZN1B2fnEv\004\000\001\000\006\000\000\004", metadata !38, metadata !8, metadata !11, null, i32 (%class.A*)* @_ZN1B2fnEv, null, null, null} ; [ DW_TAG_subprogram ]
!11 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !38, metadata !3, null, metadata !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !7, metadata !13}
!13 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", metadata !38, metadata !3, metadata !8} ; [ DW_TAG_pointer_type ]
!14 = metadata !{i32 16, i32 5, metadata !1, null}
!15 = metadata !{i32 17, i32 3, metadata !1, null}
!16 = metadata !{i32 18, i32 1, metadata !2, null}
!17 = metadata !{metadata !"0x101\00this\004\000", metadata !10, metadata !3, metadata !13} ; [ DW_TAG_arg_variable ]
!18 = metadata !{i32 4, i32 7, metadata !10, null}
!19 = metadata !{metadata !"0x100\00a\009\000", metadata !20, metadata !3, metadata !21} ; [ DW_TAG_auto_variable ]
!20 = metadata !{metadata !"0xb\004\0012\000", metadata !38, metadata !10} ; [ DW_TAG_lexical_block ]
!21 = metadata !{metadata !"0x2\00A\005\008\008\000\000\000", metadata !38, metadata !10, null, metadata !22, null, null, null} ; [ DW_TAG_class_type ] [A] [line 5, size 8, align 8, offset 0] [def] [from ]
!22 = metadata !{metadata !23}
!23 = metadata !{metadata !"0x2e\00foo\00foo\00_ZZN1B2fnEvEN1A3fooEv\007\000\001\000\006\000\000\007", metadata !38, metadata !21, metadata !24, null, i32 (%class.A*)* @_ZZN1B2fnEvEN1A3fooEv, null, null, null} ; [ DW_TAG_subprogram ]
!24 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !38, metadata !3, null, metadata !25, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!25 = metadata !{metadata !7, metadata !26}
!26 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", metadata !38, metadata !3, metadata !21} ; [ DW_TAG_pointer_type ]
!27 = metadata !{i32 9, i32 7, metadata !20, null}
!28 = metadata !{metadata !"0x100\00i\0010\000", metadata !20, metadata !3, metadata !7} ; [ DW_TAG_auto_variable ]
!29 = metadata !{i32 10, i32 9, metadata !20, null}
!30 = metadata !{i32 10, i32 5, metadata !20, null}
!31 = metadata !{i32 11, i32 5, metadata !20, null}
!32 = metadata !{i32 12, i32 3, metadata !10, null}
!33 = metadata !{metadata !"0x101\00this\007\000", metadata !23, metadata !3, metadata !26} ; [ DW_TAG_arg_variable ]
!34 = metadata !{i32 7, i32 11, metadata !23, null}
!35 = metadata !{i32 7, i32 19, metadata !36, null}
!36 = metadata !{metadata !"0xb\007\0017\000", metadata !38, metadata !23} ; [ DW_TAG_lexical_block ]
!38 = metadata !{metadata !"one.cc", metadata !"/tmp" }
!39 = metadata !{i32 0}
!40 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
