; RUN: llc -march=mipsel -mattr=mips16 -relocation-model=pic < %s | FileCheck %s -check-prefix=16

@i = global i32 1, align 4
@j = global i32 2, align 4
@a = global i32 5, align 4
@.str = private unnamed_addr constant [8 x i8] c"%i = 2\0A\00", align 1
@k = common global i32 0, align 4

define void @t() nounwind "target-cpu"="mips16" "target-features"="+mips16,+o32" {
entry:
  %0 = load i32, i32* @a, align 4
  %cmp = icmp eq i32 %0, 10
  %1 = load i32, i32* @i, align 4
  %2 = load i32, i32* @j, align 4
  %cond = select i1 %cmp, i32 %1, i32 %2
  store i32 %cond, i32* @i, align 4
  ret void
}

attributes #0 = { nounwind "target-cpu"="mips16" "target-features"="+mips16,+o32" }


; 16:	cmpi	${{[0-9]+}}, 10
; 16:	bteqz	$BB{{[0-9]+}}_{{[0-9]}}


