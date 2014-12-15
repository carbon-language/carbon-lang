; RUN: llc -mtriple=x86_64-macosx %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: DW_TAG_structure_type
; CHECK:                 DW_AT_declaration
; CHECK:                 DW_AT_APPLE_runtime_class

%0 = type opaque

@a = common global %0* null, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11, !12, !14}

!0 = !{!"0x11\0016\00clang version 3.1 (trunk 152054 trunk 152094)\000\00\002\00\000", !13, !1, !1, !1, !3,  !1} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x34\00a\00a\00\003\000\001", null, !6, !7, %0** @a, null} ; [ DW_TAG_variable ]
!6 = !{!"0x29", !13} ; [ DW_TAG_file_type ]
!7 = !{!"0xf\00\000\0064\0064\000\000", null, null, !8} ; [ DW_TAG_pointer_type ]
!8 = !{!"0x13\00FooBarBaz\001\000\000\000\004\0016", !13, null, null, null, null, null, null} ; [ DW_TAG_structure_type ] [FooBarBaz] [line 1, size 0, align 0, offset 0] [decl] [from ]
!9 = !{i32 1, !"Objective-C Version", i32 2}
!10 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!11 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!12 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!13 = !{!"foo.m", !"/Users/echristo"}
!14 = !{i32 1, !"Debug Info Version", i32 2}
