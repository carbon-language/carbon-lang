; RUN: llc -mtriple=mipsel-sde-elf -march=mipsel -relocation-model=static < %s \
; RUN: | FileCheck %s

@i = internal unnamed_addr global i32 0, align 4

define i32 @geti() nounwind readonly {
entry:
; CHECK: addiu ${{[0-9]+}}, $gp, %gp_rel(i)
  %0 = load i32* @i, align 4
  ret i32 %0
}

