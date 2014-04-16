; RUN: llc -march=mips < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s
; RUN: llc -march=mipsel < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s

; RUN-TODO: llc -march=mips64 -mattr=-n64,+o32 < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s
; RUN-TODO: llc -march=mips64el -mattr=-n64,+o32 < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s

; RUN: llc -march=mips64 -mattr=-n64,+n32 < %s | FileCheck --check-prefix=ALL --check-prefix=N32 %s
; RUN: llc -march=mips64el -mattr=-n64,+n32 < %s | FileCheck --check-prefix=ALL --check-prefix=N32 %s

; RUN: llc -march=mips64 -mattr=-n64,+n64 < %s | FileCheck --check-prefix=ALL --check-prefix=N64 %s
; RUN: llc -march=mips64el -mattr=-n64,+n64 < %s | FileCheck --check-prefix=ALL --check-prefix=N64 %s

; Test the stack alignment for all ABI's and byte orders as specified by
; section 5 of MD00305 (MIPS ABIs Described).

define void @local_bytes_1() nounwind {
entry:
        %0 = alloca i8
        ret void
}

; ALL-LABEL: local_bytes_1:
; O32:           addiu $sp, $sp, -8
; O32:           addiu $sp, $sp, 8
; N32:           addiu $sp, $sp, -16
; N32:           addiu $sp, $sp, 16
; N64:           addiu $sp, $sp, -16
; N64:           addiu $sp, $sp, 16
