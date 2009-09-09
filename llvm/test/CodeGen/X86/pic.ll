; RUN: llc < %s -mtriple=i686-pc-linux-gnu -relocation-model=pic | FileCheck %s -check-prefix=LINUX

@ptr = external global i32* 
@dst = external global i32 
@src = external global i32 

define void @test1() nounwind {
entry:
    store i32* @dst, i32** @ptr
    %tmp.s = load i32* @src
    store i32 %tmp.s, i32* @dst
    ret void
    
; LINUX:    test1:
; LINUX: .LBB1_0:
; LINUX:	call	.Lllvm$1.$piclabel
; LINUX-NEXT: .Lllvm$1.$piclabel:
; LINUX-NEXT:	popl
; LINUX:	addl	$_GLOBAL_OFFSET_TABLE_ + [.-.Lllvm$1.$piclabel],
; LINUX:	movl	dst@GOT(%eax),
; LINUX:	movl	ptr@GOT(%eax),
; LINUX:	movl	src@GOT(%eax),
; LINUX:	ret
}

@ptr2 = global i32* null
@dst2 = global i32 0
@src2 = global i32 0

define void @test2() nounwind {
entry:
    store i32* @dst2, i32** @ptr2
    %tmp.s = load i32* @src2
    store i32 %tmp.s, i32* @dst2
    ret void
    
; LINUX: test2:
; LINUX:	call	.Lllvm$2.$piclabel
; LINUX-NEXT: .Lllvm$2.$piclabel:
; LINUX-NEXT:	popl
; LINUX:	addl	$_GLOBAL_OFFSET_TABLE_ + [.-.Lllvm$2.$piclabel], %eax
; LINUX:	movl	dst2@GOT(%eax),
; LINUX:	movl	ptr2@GOT(%eax),
; LINUX:	movl	src2@GOT(%eax),
; LINUX:	ret

}

declare i8* @malloc(i32)

define void @test3() nounwind {
entry:
    %ptr = call i8* @malloc(i32 40)
    ret void
; LINUX: test3:
; LINUX: 	pushl	%ebx
; LINUX-NEXT: 	subl	$8, %esp
; LINUX-NEXT: 	call	.Lllvm$3.$piclabel
; LINUX-NEXT: .Lllvm$3.$piclabel:
; LINUX-NEXT: 	popl	%ebx
; LINUX: 	addl	$_GLOBAL_OFFSET_TABLE_ + [.-.Lllvm$3.$piclabel], %ebx
; LINUX: 	movl	$40, (%esp)
; LINUX: 	call	malloc@PLT
; LINUX: 	addl	$8, %esp
; LINUX: 	popl	%ebx
; LINUX: 	ret
}

@pfoo = external global void(...)* 

define void @test4() nounwind {
entry:
    %tmp = call void(...)*(...)* @afoo()
    store void(...)* %tmp, void(...)** @pfoo
    %tmp1 = load void(...)** @pfoo
    call void(...)* %tmp1()
    ret void
; LINUX: test4:
; LINUX: 	call	.Lllvm$4.$piclabel
; LINUX-NEXT: .Lllvm$4.$piclabel:
; LINUX: 	popl
; LINUX: 	addl	$_GLOBAL_OFFSET_TABLE_ + [.-.Lllvm$4.$piclabel],
; LINUX: 	movl	pfoo@GOT(%esi),
; LINUX: 	call	afoo@PLT
; LINUX: 	call	*
}

declare void(...)* @afoo(...)

define void @test5() nounwind {
entry:
    call void(...)* @foo()
    ret void
; LINUX: test5:
; LINUX: call	.Lllvm$5.$piclabel
; LINUX: popl	%ebx
; LINUX: addl	$_GLOBAL_OFFSET_TABLE_ + [.-.Lllvm$5.$piclabel], %ebx
; LINUX: call	foo@PLT
}

declare void @foo(...)


@ptr6 = internal global i32* null
@dst6 = internal global i32 0
@src6 = internal global i32 0

define void @test6() nounwind {
entry:
    store i32* @dst6, i32** @ptr6
    %tmp.s = load i32* @src6
    store i32 %tmp.s, i32* @dst6
    ret void
    
; LINUX: test6:
; LINUX: 	call	.Lllvm$6.$piclabel
; LINUX-NEXT: .Lllvm$6.$piclabel:
; LINUX-NEXT: 	popl	%eax
; LINUX: 	addl	$_GLOBAL_OFFSET_TABLE_ + [.-.Lllvm$6.$piclabel], %eax
; LINUX: 	leal	dst6@GOTOFF(%eax), %ecx
; LINUX: 	movl	%ecx, ptr6@GOTOFF(%eax)
; LINUX: 	movl	src6@GOTOFF(%eax), %ecx
; LINUX: 	movl	%ecx, dst6@GOTOFF(%eax)
; LINUX: 	ret
}

