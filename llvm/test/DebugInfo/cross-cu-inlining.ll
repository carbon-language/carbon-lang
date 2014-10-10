; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck -implicit-check-not=DW_TAG %s

; Build from source:
; $ clang++ a.cpp b.cpp -g -c -emit-llvm
; $ llvm-link a.bc b.bc -o ab.bc
; $ opt -inline ab.bc -o ab-opt.bc
; $ cat a.cpp
; extern int i;
; int func(int);
; int main() {
;   return func(i);
; }
; $ cat b.cpp
; int __attribute__((always_inline)) func(int x) {
;   return x * 2;
; }

; Ensure that func inlined into main is described and references the abstract
; definition in b.cpp's CU.

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_name {{.*}} "a.cpp"
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_type [DW_FORM_ref_addr] (0x00000000[[INT:.*]])
; CHECK:     DW_TAG_inlined_subroutine
; CHECK:       DW_AT_abstract_origin {{.*}}[[ABS_FUNC:........]] "_Z4funci"
; CHECK:       DW_TAG_formal_parameter
; CHECK:         DW_AT_abstract_origin {{.*}}[[ABS_VAR:........]] "x"

; Check the abstract definition is in the 'b.cpp' CU and doesn't contain any
; concrete information (address range or variable location)
; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_name {{.*}} "b.cpp"
; CHECK: 0x[[ABS_FUNC]]: DW_TAG_subprogram
; CHECK-NOT: DW_AT_low_pc
; CHECK: 0x[[ABS_VAR]]: DW_TAG_formal_parameter
; CHECK-NOT: DW_AT_location
; CHECK: DW_AT_type [DW_FORM_ref4] {{.*}} {0x[[INT]]}
; CHECK-NOT: DW_AT_location

; CHECK: 0x[[INT]]: DW_TAG_base_type
; CHECK:   DW_AT_name {{.*}} "int"

; Check the concrete out of line definition references the abstract and
; provides the address range and variable location
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_low_pc
; CHECK:   DW_AT_abstract_origin {{.*}} {0x[[ABS_FUNC]]} "_Z4funci"
; CHECK:   DW_TAG_formal_parameter
; CHECK:     DW_AT_location
; CHECK:     DW_AT_abstract_origin {{.*}} {0x[[ABS_VAR]]} "x"


@i = external global i32

; Function Attrs: uwtable
define i32 @main() #0 {
entry:
  %x.addr.i = alloca i32, align 4
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = load i32* @i, align 4, !dbg !19
  %1 = bitcast i32* %x.addr.i to i8*
  call void @llvm.lifetime.start(i64 4, i8* %1)
  store i32 %0, i32* %x.addr.i, align 4
  call void @llvm.dbg.declare(metadata !{i32* %x.addr.i}, metadata !20, metadata !{metadata !"0x102"}), !dbg !21
  %2 = load i32* %x.addr.i, align 4, !dbg !22
  %mul.i = mul nsw i32 %2, 2, !dbg !22
  %3 = bitcast i32* %x.addr.i to i8*, !dbg !22
  call void @llvm.lifetime.end(i64 4, i8* %3), !dbg !22
  ret i32 %mul.i, !dbg !19
}

; Function Attrs: alwaysinline nounwind uwtable
define i32 @_Z4funci(i32 %x) #1 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %x.addr}, metadata !20, metadata !{metadata !"0x102"}), !dbg !23
  %0 = load i32* %x.addr, align 4, !dbg !24
  %mul = mul nsw i32 %0, 2, !dbg !24
  ret i32 %mul, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #3

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #3

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0, !9}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!18, !18}

!0 = metadata !{metadata !"0x11\004\00clang version 3.5.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/a.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"a.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00main\00main\00\003\000\001\000\006\00256\000\003", metadata !1, metadata !5, metadata !6, null, i32 ()* @main, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 3] [def] [main]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/a.cpp]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8}
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !"0x11\004\00clang version 3.5.0 \000\00\000\00\001", metadata !10, metadata !2, metadata !2, metadata !11, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/b.cpp] [DW_LANG_C_plus_plus]
!10 = metadata !{metadata !"b.cpp", metadata !"/tmp/dbginfo"}
!11 = metadata !{metadata !12}
!12 = metadata !{metadata !"0x2e\00func\00func\00_Z4funci\001\000\001\000\006\00256\000\001", metadata !10, metadata !13, metadata !14, null, i32 (i32)* @_Z4funci, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [func]
!13 = metadata !{metadata !"0x29", metadata !10}        ; [ DW_TAG_file_type ] [/tmp/dbginfo/b.cpp]
!14 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !15, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = metadata !{metadata !8, metadata !8}
!16 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!17 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!18 = metadata !{metadata !"clang version 3.5.0 "}
!19 = metadata !{i32 4, i32 0, metadata !4, null}
!20 = metadata !{metadata !"0x101\00x\0016777217\000", metadata !12, metadata !13, metadata !8} ; [ DW_TAG_arg_variable ] [x] [line 1]
!21 = metadata !{i32 1, i32 0, metadata !12, metadata !19}
!22 = metadata !{i32 2, i32 0, metadata !12, metadata !19}
!23 = metadata !{i32 1, i32 0, metadata !12, null}
!24 = metadata !{i32 2, i32 0, metadata !12, null}

