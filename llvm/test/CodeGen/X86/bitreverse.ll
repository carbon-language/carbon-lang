; RUN: llc < %s -mtriple=i686-unknown | FileCheck %s

; These tests just check that the plumbing is in place for @llvm.bitreverse. The
; actual output is massive at the moment as llvm.bitreverse is not yet legal.

declare <2 x i16> @llvm.bitreverse.v2i16(<2 x i16>) readnone

define <2 x i16> @f(<2 x i16> %a) {
; CHECK-LABEL: f:
; CHECK: shll
  %b = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %a)
  ret <2 x i16> %b
}

declare i8 @llvm.bitreverse.i8(i8) readnone

define i8 @g(i8 %a) {
; CHECK-LABEL: g:
; CHECK: shlb
  %b = call i8 @llvm.bitreverse.i8(i8 %a)
  ret i8 %b
}

; These tests check that bitreverse(constant) calls are folded

define <2 x i16> @fold_v2i16() {
; CHECK-LABEL: fold_v2i16:
; CHECK:       # BB#0:
; CHECK-NEXT:    movw $-4096, %ax # imm = 0xFFFFFFFFFFFFF000
; CHECK-NEXT:    movw $240, %dx
; CHECK-NEXT:    retl
  %b = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> <i16 15, i16 3840>)
  ret <2 x i16> %b
}

define i8 @fold_i8() {
; CHECK-LABEL: fold_i8:
; CHECK:       # BB#0:
; CHECK-NEXT:    movb $-16, %al
; CHECK-NEXT:    retl
  %b = call i8 @llvm.bitreverse.i8(i8 15)
  ret i8 %b
}

; These tests check that bitreverse(bitreverse()) calls are removed

define i8 @identity_i8(i8 %a) {
; CHECK-LABEL: identity_i8:
; CHECK:       # BB#0:
; CHECK-NEXT:    movb {{[0-9]+}}(%esp), %al
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shlb $7, %cl
; CHECK-NEXT:    movl %eax, %edx
; CHECK-NEXT:    shlb $5, %dl
; CHECK-NEXT:    andb $64, %dl
; CHECK-NEXT:    movb %al, %ah
; CHECK-NEXT:    shlb $3, %ah
; CHECK-NEXT:    andb $32, %ah
; CHECK-NEXT:    orb %dl, %ah
; CHECK-NEXT:    movl %eax, %edx
; CHECK-NEXT:    addb %dl, %dl
; CHECK-NEXT:    andb $16, %dl
; CHECK-NEXT:    orb %ah, %dl
; CHECK-NEXT:    movb %al, %ah
; CHECK-NEXT:    shrb %ah
; CHECK-NEXT:    andb $8, %ah
; CHECK-NEXT:    orb %dl, %ah
; CHECK-NEXT:    movl %eax, %edx
; CHECK-NEXT:    shrb $3, %dl
; CHECK-NEXT:    andb $4, %dl
; CHECK-NEXT:    orb %ah, %dl
; CHECK-NEXT:    movb %al, %ah
; CHECK-NEXT:    shrb $5, %ah
; CHECK-NEXT:    andb $2, %ah
; CHECK-NEXT:    orb %dl, %ah
; CHECK-NEXT:    shrb $7, %al
; CHECK-NEXT:    orb %ah, %al
; CHECK-NEXT:    orb %cl, %al
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shlb $7, %cl
; CHECK-NEXT:    movl %eax, %edx
; CHECK-NEXT:    shlb $5, %dl
; CHECK-NEXT:    andb $64, %dl
; CHECK-NEXT:    movb %al, %ah
; CHECK-NEXT:    shlb $3, %ah
; CHECK-NEXT:    andb $32, %ah
; CHECK-NEXT:    orb %dl, %ah
; CHECK-NEXT:    movl %eax, %edx
; CHECK-NEXT:    addb %dl, %dl
; CHECK-NEXT:    andb $16, %dl
; CHECK-NEXT:    orb %ah, %dl
; CHECK-NEXT:    movb %al, %ah
; CHECK-NEXT:    shrb %ah
; CHECK-NEXT:    andb $8, %ah
; CHECK-NEXT:    orb %dl, %ah
; CHECK-NEXT:    movl %eax, %edx
; CHECK-NEXT:    shrb $3, %dl
; CHECK-NEXT:    andb $4, %dl
; CHECK-NEXT:    orb %ah, %dl
; CHECK-NEXT:    movb %al, %ah
; CHECK-NEXT:    shrb $5, %ah
; CHECK-NEXT:    andb $2, %ah
; CHECK-NEXT:    orb %dl, %ah
; CHECK-NEXT:    shrb $7, %al
; CHECK-NEXT:    orb %ah, %al
; CHECK-NEXT:    orb %cl, %al
; CHECK-NEXT:    retl
  %b = call i8 @llvm.bitreverse.i8(i8 %a)
  %c = call i8 @llvm.bitreverse.i8(i8 %b)
  ret i8 %c
}

define <2 x i16> @identity_v2i16(<2 x i16> %a) {
; CHECK-LABEL: identity_v2i16:
; CHECK:       # BB#0:
; CHECK-NEXT:    pushl %ebp
; CHECK-NEXT:  .Ltmp4:
; CHECK-NEXT:    .cfi_def_cfa_offset 8
; CHECK-NEXT:    pushl %ebx
; CHECK-NEXT:  .Ltmp5:
; CHECK-NEXT:    .cfi_def_cfa_offset 12
; CHECK-NEXT:    pushl %edi
; CHECK-NEXT:  .Ltmp6:
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    pushl %esi
; CHECK-NEXT:  .Ltmp7:
; CHECK-NEXT:    .cfi_def_cfa_offset 20
; CHECK-NEXT:  .Ltmp8:
; CHECK-NEXT:    .cfi_offset %esi, -20
; CHECK-NEXT:  .Ltmp9:
; CHECK-NEXT:    .cfi_offset %edi, -16
; CHECK-NEXT:  .Ltmp10:
; CHECK-NEXT:    .cfi_offset %ebx, -12
; CHECK-NEXT:  .Ltmp11:
; CHECK-NEXT:    .cfi_offset %ebp, -8
; CHECK-NEXT:    movzwl {{[0-9]+}}(%esp), %esi
; CHECK-NEXT:    movzwl {{[0-9]+}}(%esp), %ecx
; CHECK-NEXT:    movl %ecx, %eax
; CHECK-NEXT:    shll $15, %eax
; CHECK-NEXT:    movl %ecx, %edx
; CHECK-NEXT:    andl $2, %edx
; CHECK-NEXT:    shll $13, %edx
; CHECK-NEXT:    orl %eax, %edx
; CHECK-NEXT:    movl %ecx, %eax
; CHECK-NEXT:    andl $4, %eax
; CHECK-NEXT:    shll $11, %eax
; CHECK-NEXT:    orl %edx, %eax
; CHECK-NEXT:    movl %ecx, %edx
; CHECK-NEXT:    andl $8, %edx
; CHECK-NEXT:    shll $9, %edx
; CHECK-NEXT:    orl %eax, %edx
; CHECK-NEXT:    movl %ecx, %edi
; CHECK-NEXT:    andl $16, %edi
; CHECK-NEXT:    shll $7, %edi
; CHECK-NEXT:    orl %edx, %edi
; CHECK-NEXT:    movl %ecx, %eax
; CHECK-NEXT:    andl $32, %eax
; CHECK-NEXT:    shll $5, %eax
; CHECK-NEXT:    orl %edi, %eax
; CHECK-NEXT:    movl %ecx, %edx
; CHECK-NEXT:    andl $64, %edx
; CHECK-NEXT:    shll $3, %edx
; CHECK-NEXT:    leal (%ecx,%ecx), %edi
; CHECK-NEXT:    andl $256, %edi # imm = 0x100
; CHECK-NEXT:    orl %edx, %edi
; CHECK-NEXT:    movl %ecx, %edx
; CHECK-NEXT:    shrl %edx
; CHECK-NEXT:    andl $128, %edx
; CHECK-NEXT:    orl %edi, %edx
; CHECK-NEXT:    movl %ecx, %edi
; CHECK-NEXT:    shrl $3, %edi
; CHECK-NEXT:    andl $64, %edi
; CHECK-NEXT:    orl %edx, %edi
; CHECK-NEXT:    movl %ecx, %edx
; CHECK-NEXT:    shrl $5, %edx
; CHECK-NEXT:    andl $32, %edx
; CHECK-NEXT:    orl %edi, %edx
; CHECK-NEXT:    movl %ecx, %edi
; CHECK-NEXT:    shrl $7, %edi
; CHECK-NEXT:    andl $16, %edi
; CHECK-NEXT:    orl %edx, %edi
; CHECK-NEXT:    movl %ecx, %edx
; CHECK-NEXT:    shrl $9, %edx
; CHECK-NEXT:    andl $8, %edx
; CHECK-NEXT:    orl %edi, %edx
; CHECK-NEXT:    movl %ecx, %edi
; CHECK-NEXT:    shrl $11, %edi
; CHECK-NEXT:    andl $4, %edi
; CHECK-NEXT:    orl %edx, %edi
; CHECK-NEXT:    movl %ecx, %edx
; CHECK-NEXT:    shrl $13, %edx
; CHECK-NEXT:    andl $2, %edx
; CHECK-NEXT:    orl %edi, %edx
; CHECK-NEXT:    shrl $15, %ecx
; CHECK-NEXT:    orl %edx, %ecx
; CHECK-NEXT:    orl %eax, %ecx
; CHECK-NEXT:    movl %ecx, %edx
; CHECK-NEXT:    andl $32768, %edx # imm = 0x8000
; CHECK-NEXT:    movl %esi, %eax
; CHECK-NEXT:    shll $15, %eax
; CHECK-NEXT:    movl %esi, %edi
; CHECK-NEXT:    andl $2, %edi
; CHECK-NEXT:    shll $13, %edi
; CHECK-NEXT:    orl %eax, %edi
; CHECK-NEXT:    movl %esi, %eax
; CHECK-NEXT:    andl $4, %eax
; CHECK-NEXT:    shll $11, %eax
; CHECK-NEXT:    orl %edi, %eax
; CHECK-NEXT:    movl %esi, %edi
; CHECK-NEXT:    andl $8, %edi
; CHECK-NEXT:    shll $9, %edi
; CHECK-NEXT:    orl %eax, %edi
; CHECK-NEXT:    movl %esi, %ebx
; CHECK-NEXT:    andl $16, %ebx
; CHECK-NEXT:    shll $7, %ebx
; CHECK-NEXT:    orl %edi, %ebx
; CHECK-NEXT:    movl %esi, %eax
; CHECK-NEXT:    andl $32, %eax
; CHECK-NEXT:    shll $5, %eax
; CHECK-NEXT:    orl %ebx, %eax
; CHECK-NEXT:    movl %esi, %edi
; CHECK-NEXT:    andl $64, %edi
; CHECK-NEXT:    shll $3, %edi
; CHECK-NEXT:    leal (%esi,%esi), %ebx
; CHECK-NEXT:    andl $256, %ebx # imm = 0x100
; CHECK-NEXT:    orl %edi, %ebx
; CHECK-NEXT:    movl %esi, %edi
; CHECK-NEXT:    shrl %edi
; CHECK-NEXT:    andl $128, %edi
; CHECK-NEXT:    orl %ebx, %edi
; CHECK-NEXT:    movl %esi, %ebx
; CHECK-NEXT:    shrl $3, %ebx
; CHECK-NEXT:    andl $64, %ebx
; CHECK-NEXT:    orl %edi, %ebx
; CHECK-NEXT:    movl %esi, %edi
; CHECK-NEXT:    shrl $5, %edi
; CHECK-NEXT:    andl $32, %edi
; CHECK-NEXT:    orl %ebx, %edi
; CHECK-NEXT:    movl %esi, %ebx
; CHECK-NEXT:    shrl $7, %ebx
; CHECK-NEXT:    andl $16, %ebx
; CHECK-NEXT:    orl %edi, %ebx
; CHECK-NEXT:    movl %esi, %edi
; CHECK-NEXT:    shrl $9, %edi
; CHECK-NEXT:    andl $8, %edi
; CHECK-NEXT:    orl %ebx, %edi
; CHECK-NEXT:    movl %esi, %ebx
; CHECK-NEXT:    shrl $11, %ebx
; CHECK-NEXT:    andl $4, %ebx
; CHECK-NEXT:    orl %edi, %ebx
; CHECK-NEXT:    movl %esi, %edi
; CHECK-NEXT:    shrl $13, %edi
; CHECK-NEXT:    andl $2, %edi
; CHECK-NEXT:    orl %ebx, %edi
; CHECK-NEXT:    shrl $15, %esi
; CHECK-NEXT:    orl %edi, %esi
; CHECK-NEXT:    orl %eax, %esi
; CHECK-NEXT:    movl %esi, %eax
; CHECK-NEXT:    andl $32768, %eax # imm = 0x8000
; CHECK-NEXT:    movl %esi, %edi
; CHECK-NEXT:    shll $15, %edi
; CHECK-NEXT:    movl %esi, %ebx
; CHECK-NEXT:    andl $2, %ebx
; CHECK-NEXT:    shll $13, %ebx
; CHECK-NEXT:    orl %edi, %ebx
; CHECK-NEXT:    movl %esi, %edi
; CHECK-NEXT:    andl $4, %edi
; CHECK-NEXT:    shll $11, %edi
; CHECK-NEXT:    orl %ebx, %edi
; CHECK-NEXT:    movl %esi, %ebx
; CHECK-NEXT:    andl $8, %ebx
; CHECK-NEXT:    shll $9, %ebx
; CHECK-NEXT:    orl %edi, %ebx
; CHECK-NEXT:    movl %esi, %ebp
; CHECK-NEXT:    andl $16, %ebp
; CHECK-NEXT:    shll $7, %ebp
; CHECK-NEXT:    orl %ebx, %ebp
; CHECK-NEXT:    movl %esi, %edi
; CHECK-NEXT:    andl $32, %edi
; CHECK-NEXT:    shll $5, %edi
; CHECK-NEXT:    orl %ebp, %edi
; CHECK-NEXT:    movl %esi, %ebx
; CHECK-NEXT:    andl $64, %ebx
; CHECK-NEXT:    shll $3, %ebx
; CHECK-NEXT:    leal (%esi,%esi), %ebp
; CHECK-NEXT:    andl $256, %ebp # imm = 0x100
; CHECK-NEXT:    orl %ebx, %ebp
; CHECK-NEXT:    movl %esi, %ebx
; CHECK-NEXT:    shrl %ebx
; CHECK-NEXT:    andl $128, %ebx
; CHECK-NEXT:    orl %ebp, %ebx
; CHECK-NEXT:    movl %esi, %ebp
; CHECK-NEXT:    shrl $3, %ebp
; CHECK-NEXT:    andl $64, %ebp
; CHECK-NEXT:    orl %ebx, %ebp
; CHECK-NEXT:    movl %esi, %ebx
; CHECK-NEXT:    shrl $5, %ebx
; CHECK-NEXT:    andl $32, %ebx
; CHECK-NEXT:    orl %ebp, %ebx
; CHECK-NEXT:    movl %esi, %ebp
; CHECK-NEXT:    shrl $7, %ebp
; CHECK-NEXT:    andl $16, %ebp
; CHECK-NEXT:    orl %ebx, %ebp
; CHECK-NEXT:    movl %esi, %ebx
; CHECK-NEXT:    shrl $9, %ebx
; CHECK-NEXT:    andl $8, %ebx
; CHECK-NEXT:    orl %ebp, %ebx
; CHECK-NEXT:    movl %esi, %ebp
; CHECK-NEXT:    shrl $11, %ebp
; CHECK-NEXT:    andl $4, %ebp
; CHECK-NEXT:    orl %ebx, %ebp
; CHECK-NEXT:    shrl $13, %esi
; CHECK-NEXT:    andl $2, %esi
; CHECK-NEXT:    orl %ebp, %esi
; CHECK-NEXT:    shrl $15, %eax
; CHECK-NEXT:    orl %esi, %eax
; CHECK-NEXT:    orl %edi, %eax
; CHECK-NEXT:    movl %ecx, %esi
; CHECK-NEXT:    shll $15, %esi
; CHECK-NEXT:    movl %ecx, %edi
; CHECK-NEXT:    andl $2, %edi
; CHECK-NEXT:    shll $13, %edi
; CHECK-NEXT:    orl %esi, %edi
; CHECK-NEXT:    movl %ecx, %esi
; CHECK-NEXT:    andl $4, %esi
; CHECK-NEXT:    shll $11, %esi
; CHECK-NEXT:    orl %edi, %esi
; CHECK-NEXT:    movl %ecx, %edi
; CHECK-NEXT:    andl $8, %edi
; CHECK-NEXT:    shll $9, %edi
; CHECK-NEXT:    orl %esi, %edi
; CHECK-NEXT:    movl %ecx, %ebx
; CHECK-NEXT:    andl $16, %ebx
; CHECK-NEXT:    shll $7, %ebx
; CHECK-NEXT:    orl %edi, %ebx
; CHECK-NEXT:    movl %ecx, %esi
; CHECK-NEXT:    andl $32, %esi
; CHECK-NEXT:    shll $5, %esi
; CHECK-NEXT:    orl %ebx, %esi
; CHECK-NEXT:    movl %ecx, %edi
; CHECK-NEXT:    andl $64, %edi
; CHECK-NEXT:    shll $3, %edi
; CHECK-NEXT:    leal (%ecx,%ecx), %ebx
; CHECK-NEXT:    andl $256, %ebx # imm = 0x100
; CHECK-NEXT:    orl %edi, %ebx
; CHECK-NEXT:    movl %ecx, %edi
; CHECK-NEXT:    shrl %edi
; CHECK-NEXT:    andl $128, %edi
; CHECK-NEXT:    orl %ebx, %edi
; CHECK-NEXT:    movl %ecx, %ebx
; CHECK-NEXT:    shrl $3, %ebx
; CHECK-NEXT:    andl $64, %ebx
; CHECK-NEXT:    orl %edi, %ebx
; CHECK-NEXT:    movl %ecx, %edi
; CHECK-NEXT:    shrl $5, %edi
; CHECK-NEXT:    andl $32, %edi
; CHECK-NEXT:    orl %ebx, %edi
; CHECK-NEXT:    movl %ecx, %ebx
; CHECK-NEXT:    shrl $7, %ebx
; CHECK-NEXT:    andl $16, %ebx
; CHECK-NEXT:    orl %edi, %ebx
; CHECK-NEXT:    movl %ecx, %edi
; CHECK-NEXT:    shrl $9, %edi
; CHECK-NEXT:    andl $8, %edi
; CHECK-NEXT:    orl %ebx, %edi
; CHECK-NEXT:    movl %ecx, %ebx
; CHECK-NEXT:    shrl $11, %ebx
; CHECK-NEXT:    andl $4, %ebx
; CHECK-NEXT:    orl %edi, %ebx
; CHECK-NEXT:    shrl $13, %ecx
; CHECK-NEXT:    andl $2, %ecx
; CHECK-NEXT:    orl %ebx, %ecx
; CHECK-NEXT:    shrl $15, %edx
; CHECK-NEXT:    orl %ecx, %edx
; CHECK-NEXT:    orl %esi, %edx
; CHECK-NEXT:    popl %esi
; CHECK-NEXT:    popl %edi
; CHECK-NEXT:    popl %ebx
; CHECK-NEXT:    popl %ebp
; CHECK-NEXT:    retl
  %b = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %a)
  %c = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %b)
  ret <2 x i16> %c
}
