; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips16 -soft-float -mips16-hard-float -relocation-model=static -mips16-constant-islands -mips-constant-islands-small-offset=20  < %s | FileCheck %s -check-prefix=offset20

; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips16 -soft-float -mips16-hard-float -relocation-model=static -mips16-constant-islands -mips-constant-islands-small-offset=40  < %s | FileCheck %s -check-prefix=offset40


@i = common global i32 0, align 4
@b = common global i32 0, align 4

; Function Attrs: nounwind
define void @t() #0 {
entry:
  store i32 -559023410, i32* @i, align 4
  %0 = load i32* @b, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.else
; offset20: lw	${{[0-9]+}}, $CPI0_1	# 16 bit inst
; offset20:	b	$BB0_2
; offset20:	.align	2
; offset20: $CPI0_0:
; offset20:	.4byte	3735943886
; offset20: $BB0_2: 

; offset40:	beqz	${{[0-9]+}}, $BB0_3
; offset40:	jal	foo
; offset40:	nop
; offset40:	b	$BB0_4
; offset40:	.align	2
; offset40: $CPI0_0:
; offset40:	.4byte	3735943886
; offset40: $BB0_3:
; offset40:	jal	goo

if.then:                                          ; preds = %entry
  call void bitcast (void (...)* @foo to void ()*)()
  br label %if.end

if.else:                                          ; preds = %entry
  call void bitcast (void (...)* @goo to void ()*)()
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  ret void
}

declare void @foo(...) #1

declare void @goo(...) #1

declare void @hoo(...) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.4 (gitosis@dmz-portal.mips.com:clang.git 3a50d847e098f36e3bf8bc14eea07a6cc35f7803) (gitosis@dmz-portal.mips.com:llvm.git f52db0b69f0c888bdc98bb2f13aaecc1e83288a9)"}
