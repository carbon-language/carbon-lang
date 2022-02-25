; RUN: llc -no-integrated-as < %s | grep "foo 0 0"

define void @bar() nounwind {
	tail call void asm sideeffect "foo ${:uid} ${:uid}", ""() nounwind
	ret void
}
