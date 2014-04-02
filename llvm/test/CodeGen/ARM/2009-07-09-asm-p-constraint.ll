; RUN: llc -mtriple=arm-eabi -mattr=+v6 %s -o /dev/null

define void @test(i8* %x) nounwind {
entry:
	call void asm sideeffect "pld\09${0:a}", "r,~{cc}"(i8* %x) nounwind
	ret void
}
