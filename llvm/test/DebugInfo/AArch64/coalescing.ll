; RUN: llc -filetype=obj %s -o - | llvm-dwarfdump - | FileCheck %s
;
; Generated at -Os from:
; void *foo(void *dst);
; void start() {
;   unsigned size;
;   foo(&size);
;   if (size != 0) {
;   }
; }

; ModuleID = 'test1.cpp'
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

; Function Attrs: nounwind optsize
define void @_Z5startv() #0 {
entry:
  %size = alloca i32, align 4
  %0 = bitcast i32* %size to i8*, !dbg !15
  %call = call i8* @_Z3fooPv(i8* %0) #3, !dbg !15
  call void @llvm.dbg.value(metadata !{i32* %size}, i64 0, metadata !10, metadata !16), !dbg !17
  ; CHECK: .debug_info contents:
  ; CHECK: DW_TAG_variable
  ; CHECK-NEXT: DW_AT_location
  ; CHECK-NEXT: DW_AT_name {{.*}}"size"
  ; CHECK: .debug_loc contents:
  ; CHECK: Location description: 70 00
  ret void, !dbg !18
}

; Function Attrs: optsize
declare i8* @_Z3fooPv(i8*) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind optsize }
attributes #1 = { optsize }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind optsize }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = metadata !{metadata !"0x11\004\00clang version 3.6.0 (trunk 223149) (llvm/trunk 223115)\001\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [<stdin>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"<stdin>", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00start\00start\00_Z5startv\002\000\001\000\000\00256\001\003", metadata !5, metadata !6, metadata !7, null, void ()* @_Z5startv, null, null, metadata !9} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 3] [start]
!5 = metadata !{metadata !"test1.c", metadata !""}
!6 = metadata !{metadata !"0x29", metadata !5}    ; [ DW_TAG_file_type ] [/test1.c]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null}
!9 = metadata !{metadata !10}
!10 = metadata !{metadata !"0x100\00size\004\000", metadata !4, metadata !6, metadata !11} ; [ DW_TAG_auto_variable ] [size] [line 4]
!11 = metadata !{metadata !"0x24\00unsigned int\000\0032\0032\000\000\007", null, null} ; [ DW_TAG_base_type ] [unsigned int] [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!12 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!13 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!14 = metadata !{metadata !"clang version 3.6.0 (trunk 223149) (llvm/trunk 223115)"}
!15 = metadata !{i32 5, i32 3, metadata !4, null}
!16 = metadata !{metadata !"0x102"}               ; [ DW_TAG_expression ]
!17 = metadata !{i32 4, i32 12, metadata !4, null}
!18 = metadata !{i32 8, i32 1, metadata !4, null}
