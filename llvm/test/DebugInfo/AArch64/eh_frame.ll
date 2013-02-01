; RUN: llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu %s -filetype=obj -o %t
; RUN: llvm-objdump -s %t | FileCheck %s
@var = global i32 0

declare void @bar()

define i64 @check_largest_class(i32 %in)  {
  %res = load i32* @var
  call void @bar()
  %ext = zext i32 %res to i64
  ret i64 %ext
}

; The really key points we're checking here are:
;  * Return register is x30.
;  * Pointer format is 0x1b (GNU doesn't appear to understand others).

; The rest is largely incidental, but not expected to change regularly.

; Output is:

; CHECK: Contents of section .eh_frame:
; CHECK-NEXT: 0000 10000000 00000000 017a5200 017c1e01  .........zR..|..
; CHECK-NEXT: 0010 1b0c1f00 18000000 18000000 00000000  ................


; Won't check the rest, it's rather incidental.
; 0020 24000000 00440c1f 10449e02 93040000  $....D...D......


; The first CIE:
; -------------------
; 10000000: length of first CIE = 0x10
; 00000000: This is a CIE
; 01: version = 0x1
; 7a 52 00: augmentation string "zR" -- pointer format is specified
; 01: code alignment factor 1
; 7c: data alignment factor -4
; 1e: return address register 30 (== x30).
; 01: 1 byte of augmentation
; 1b: pointer format 1b: DW_EH_PE_pcrel | DW_EH_PE_sdata4
; 0c 1f 00: initial instructions: "DW_CFA_def_cfa x31 ofs 0" in this case

; Next the FDE:
; -------------
; 18000000: FDE length 0x18
; 18000000: Uses CIE 0x18 backwards (only coincidentally same as above)
; 00000000: PC begin for this FDE is at 00000000 (relocation is applied here)
; 24000000: FDE applies up to PC begin+0x24
; 00: Augmentation string length 0 for this FDE
; Rest: call frame instructions
