; RUN: llc < %s -march=x86 -fast-isel | FileCheck %s

declare i64 @test1a(i64)

define i32 @test1(i64 %x) nounwind {
; CHECK: test1:
; CHECK: andb $1, %
	%y = add i64 %x, -3
	%t = call i64 @test1a(i64 %y)
	%s = mul i64 %t, 77
	%z = trunc i64 %s to i1
	br label %next

next:		; preds = %0
	%u = zext i1 %z to i32
	%v = add i32 %u, 1999
	br label %exit

exit:		; preds = %next
	ret i32 %v
}

define void @test2(i8* %a) nounwind {
entry:
; CHECK: test2:
; CHECK: movb {{.*}} %al
; CHECK-NEXT: xorb $1, %al
; CHECK-NEXT: testb $1
  %tmp = load i8* %a, align 1
  %tobool = trunc i8 %tmp to i1
  %tobool2 = xor i1 %tobool, true
  br i1 %tobool2, label %if.then, label %if.end

if.then:
  call void @test2(i8* null)
  br label %if.end

if.end:
  ret void
}
