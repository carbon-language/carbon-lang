; RUN: llc -march=mips64 -relocation-model=static -mattr=-n64,+n32 < %s | FileCheck --check-prefix=ALL --check-prefix=N32 %s
; RUN: llc -march=mips64el -relocation-model=static -mattr=-n64,+n32 < %s | FileCheck --check-prefix=ALL --check-prefix=N32 %s

; RUN: llc -march=mips64 -relocation-model=static -mattr=-n64,+n64 < %s | FileCheck --check-prefix=ALL --check-prefix=N64 %s
; RUN: llc -march=mips64el -relocation-model=static -mattr=-n64,+n64 < %s | FileCheck --check-prefix=ALL --check-prefix=N64 %s

; Test the fp128 returns for N32/N64 and all byte orders as specified by
; section 5 of MD00305 (MIPS ABIs Described).
;
; O32 is not tested because long double is the same as double on O32.
;
@fp128 = global fp128 zeroinitializer

define fp128 @retldouble() nounwind {
entry:
        %0 = load volatile fp128* @fp128
        ret fp128 %0
}

; ALL-LABEL: retldouble:
; N32-DAG:           ld [[R2:\$[0-9]+]], %lo(fp128)([[R1:\$[0-9]+]])
; N32-DAG:           addiu [[R3:\$[0-9]+]], [[R1]], %lo(fp128)
; N32-DAG:           ld [[R4:\$[0-9]+]], 8([[R3]])
; N32-DAG:           dmtc1 [[R2]], $f0
; N32-DAG:           dmtc1 [[R4]], $f2

; N64-DAG:           ld [[R2:\$[0-9]+]], %got_disp(fp128)([[R1:\$[0-9]+]])
; N64-DAG:           ld [[R3:\$[0-9]+]], 0([[R2]])
; N64-DAG:           ld [[R4:\$[0-9]+]], 8([[R2]])
; N64-DAG:           dmtc1 [[R3]], $f0
; N64-DAG:           dmtc1 [[R4]], $f2
