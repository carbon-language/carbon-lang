; RUN: opt < %s -jump-threading -die -S | grep icmp | count 1

declare void @f1()
declare void @f2()
declare void @f3()

define i32 @test(i32 %A) {
	%tmp455 = icmp eq i32 %A, 42
	br i1 %tmp455, label %BB1, label %BB2
        
BB2:
	call void @f1()
	br label %BB1
        

BB1:
	%tmp459 = icmp eq i32 %A, 42
	br i1 %tmp459, label %BB3, label %BB4

BB3:
	call void @f2()
        ret i32 3

BB4:
	call void @f3()
	ret i32 4
}



