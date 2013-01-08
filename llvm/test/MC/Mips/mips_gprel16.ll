; This addresses bug 14456. We were not writing
; out the addend to the gprel16 relocation. The
; addend is stored in the instruction immediate 
; field.
;llc gprel16.ll -o gprel16.o -mcpu=mips32r2 -march=mipsel -filetype=obj -relocation-model=static

; RUN: llc -mcpu=mips32r2 -march=mipsel -filetype=obj -relocation-model=static %s -o - \
; RUN: | llvm-objdump -disassemble -mattr +mips32r2 - \
; RUN: | FileCheck %s

target triple = "mipsel-sde--elf-gcc"

@var1 = internal global i32 0, align 4
@var2 = internal global i32 0, align 4

define i32 @testvar1() nounwind {
entry:
; CHECK: lw ${{[0-9]+}}, 0($gp)
  %0 = load i32* @var1, align 4
  %tobool = icmp ne i32 %0, 0
  %cond = select i1 %tobool, i32 1, i32 0
  ret i32 %cond
}

define i32 @testvar2() nounwind {
entry:
; CHECK: lw ${{[0-9]+}}, 4($gp)
  %0 = load i32* @var2, align 4
  %tobool = icmp ne i32 %0, 0
  %cond = select i1 %tobool, i32 1, i32 0
  ret i32 %cond
}

