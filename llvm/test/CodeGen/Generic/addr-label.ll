; RUN: llc %s -o -

;; Reference to a label that gets deleted.
define i8* @test1() nounwind {
entry:
	ret i8* blockaddress(@test1b, %test_label)
}

define i32 @test1b() nounwind {
entry:
	ret i32 -1
test_label:
	br label %ret
ret:
	ret i32 -1
}


;; Issues with referring to a label that gets RAUW'd later.
define i32 @test2a() nounwind {
entry:
        %target = bitcast i8* blockaddress(@test2b, %test_label) to i8*

        call i32 @test2b(i8* %target)

        ret i32 0
}

define i32 @test2b(i8* %target) nounwind {
entry:
        indirectbr i8* %target, [label %test_label]

test_label:
; assume some code here...
        br label %ret

ret:
        ret i32 -1
}
