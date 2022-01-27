; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=pic -mips16-constant-islands   < %s | FileCheck %s -check-prefix=cond-b-short

@i = global i32 1, align 4
@j = global i32 2, align 4
@k = common global i32 0, align 4

; Function Attrs: nounwind optsize
define void @t() #0 {
entry:
  %0 = load i32, i32* @i, align 4
  %1 = load i32, i32* @j, align 4
  %cmp = icmp ne i32 %0, %1
  %cond = select i1 %cmp, i32 1, i32 3
  store i32 %cond, i32* @k, align 4
; cond-b-short:	btnez	$BB0_{{[0-9]+}}  # 16 bit inst
  ret void
}

attributes #0 = { nounwind optsize "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }


