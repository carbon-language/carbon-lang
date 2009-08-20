; RUN: llvm-as < %s | llc -mtriple=i686-unknown-linux-gnu -march=x86 -relocation-model=static -code-model=small > %t
; RUN: grep leal %t | count 33
; RUN: grep movl %t | count 239
; RUN: grep addl %t | count 20
; RUN: grep subl %t | count 14
; RUN: not grep leaq %t
; RUN: not grep movq %t
; RUN: not grep addq %t
; RUN: not grep subq %t
; RUN: not grep movabs %t
; RUN: not grep largecomm %t
; RUN: not grep _GLOBAL_OFFSET_TABLE_ %t
; RUN: not grep @GOT %t
; RUN: not grep @GOTOFF %t
; RUN: not grep @GOTPCREL %t
; RUN: not grep @GOTPLT %t
; RUN: not grep @PLT %t
; RUN: not grep @PLTOFF %t
; RUN: grep {call	\\\*} %t | count 10
; RUN: not grep %rip %t
; RUN: llvm-as < %s | llc -mtriple=i686-unknown-linux-gnu -march=x86 -relocation-model=pic -code-model=small > %t
; RUN: grep leal %t | count 43
; RUN: grep movl %t | count 377
; RUN: grep addl %t | count 179
; RUN: grep subl %t | count 6
; RUN: not grep leaq %t
; RUN: not grep movq %t
; RUN: not grep addq %t
; RUN: not grep subq %t
; RUN: not grep movabs %t
; RUN: not grep largecomm %t
; RUN: grep _GLOBAL_OFFSET_TABLE_ %t | count 148
; RUN: grep @GOT %t | count 207
; RUN: grep @GOTOFF %t | count 58
; RUN: not grep @GOTPCREL %t
; RUN: not grep @GOTPLT %t
; RUN: grep @PLT %t | count 20
; RUN: not grep @PLTOFF %t
; RUN: grep {call	\\\*} %t | count 10
; RUN: not grep {%rip} %t

; RUN: llvm-as < %s | llc -mtriple=x86_64-unknown-linux-gnu -march=x86-64 -relocation-model=static -code-model=small | FileCheck %s -check-prefix=LINUX-64-STATIC

; RUN: llvm-as < %s | llc -mtriple=x86_64-unknown-linux-gnu -march=x86-64 -relocation-model=pic -code-model=small > %t
; RUN: not grep leal %t
; RUN: grep movl %t | count 98
; RUN: not grep addl %t
; RUN: not grep subl %t
; RUN: grep leaq %t | count 59
; RUN: grep movq %t | count 195
; RUN: grep addq %t | count 36
; RUN: grep subq %t | count 11
; RUN: not grep movabs %t
; RUN: not grep largecomm %t
; RUN: not grep _GLOBAL_OFFSET_TABLE_ %t
; RUN: grep @GOT %t | count 149
; RUN: not grep @GOTOFF %t
; RUN: grep @GOTPCREL %t | count 149
; RUN: not grep @GOTPLT %t
; RUN: grep @PLT %t | count 20
; RUN: not grep @PLTOFF %t
; RUN: grep {call	\\\*} %t | count 10
; RUN: grep {%rip} %t | count 207



; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin -march=x86 -relocation-model=static -code-model=small > %t
; RUN: grep leal %t | count 33
; RUN: grep movl %t | count 239
; RUN: grep addl %t | count 20
; RUN: grep subl %t | count 14
; RUN: not grep leaq %t
; RUN: not grep movq %t
; RUN: not grep addq %t
; RUN: not grep subq %t
; RUN: not grep movabs %t
; RUN: not grep largecomm %t
; RUN: not grep _GLOBAL_OFFSET_TABLE_ %t
; RUN: not grep @GOT %t
; RUN: not grep @GOTOFF %t
; RUN: not grep @GOTPCREL %t
; RUN: not grep @GOTPLT %t
; RUN: not grep @PLT %t
; RUN: not grep @PLTOFF %t
; RUN: grep {call	\\\*} %t | count 10
; RUN: not grep %rip %t
; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin -march=x86 -relocation-model=dynamic-no-pic -code-model=small > %t
; RUN: grep leal %t | count 31
; RUN: grep movl %t | count 312
; RUN: grep addl %t | count 32
; RUN: grep subl %t | count 14
; RUN: not grep leaq %t
; RUN: not grep movq %t
; RUN: not grep addq %t
; RUN: not grep subq %t
; RUN: not grep movabs %t
; RUN: not grep largecomm %t
; RUN: not grep _GLOBAL_OFFSET_TABLE_ %t
; RUN: not grep @GOT %t
; RUN: not grep @GOTOFF %t
; RUN: not grep @GOTPCREL %t
; RUN: not grep @GOTPLT %t
; RUN: not grep @PLT %t
; RUN: not grep @PLTOFF %t
; RUN: grep {call	\\\*} %t | count 10
; RUN: not grep {%rip} %t
; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin -march=x86 -relocation-model=pic -code-model=small > %t
; RUN: grep leal %t | count 57
; RUN: grep movl %t | count 292
; RUN: grep addl %t | count 32
; RUN: grep subl %t | count 14
; RUN: not grep leaq %t
; RUN: not grep movq %t
; RUN: not grep addq %t
; RUN: not grep subq %t
; RUN: not grep movabs %t
; RUN: not grep largecomm %t
; RUN: not grep _GLOBAL_OFFSET_TABLE_ %t
; RUN: not grep @GOT %t
; RUN: not grep @GOTOFF %t
; RUN: not grep @GOTPCREL %t
; RUN: not grep @GOTPLT %t
; RUN: not grep @PLT %t
; RUN: not grep @PLTOFF %t
; RUN: grep {call	\\\*} %t | count 10
; RUN: not grep {%rip} %t
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin -march=x86-64 -relocation-model=dynamic-no-pic -code-model=small > %t
; RUN: not grep leal %t
; RUN: grep movl %t | count 95
; RUN: not grep addl %t
; RUN: not grep subl %t
; RUN: grep leaq %t | count 89
; RUN: grep movq %t | count 142
; RUN: grep addq %t | count 30
; RUN: grep subq %t | count 12
; RUN: not grep movabs %t
; RUN: not grep largecomm %t
; RUN: not grep _GLOBAL_OFFSET_TABLE_ %t
; RUN: grep @GOT %t | count 92
; RUN: not grep @GOTOFF %t
; RUN: grep @GOTPCREL %t | count 92
; RUN: not grep @GOTPLT %t
; RUN: not grep @PLT %t
; RUN: not grep @PLTOFF %t
; RUN: grep {call	\\\*} %t | count 10
; RUN: grep {%rip} %t | count 208
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin -march=x86-64 -relocation-model=pic -code-model=small > %t
; RUN: not grep leal %t
; RUN: grep movl %t | count 95
; RUN: not grep addl %t
; RUN: not grep subl %t
; RUN: grep leaq %t | count 89
; RUN: grep movq %t | count 142
; RUN: grep addq %t | count 30
; RUN: grep subq %t | count 12
; RUN: not grep movabs %t
; RUN: not grep largecomm %t
; RUN: not grep _GLOBAL_OFFSET_TABLE_ %t
; RUN: grep @GOT %t | count 92
; RUN: not grep @GOTOFF %t
; RUN: grep @GOTPCREL %t | count 92
; RUN: not grep @GOTPLT %t
; RUN: not grep @PLT %t
; RUN: not grep @PLTOFF %t
; RUN: grep {call	\\\*} %t | count 10
; RUN: grep {%rip} %t | count 208

@src = external global [131072 x i32]
@dst = external global [131072 x i32]
@xsrc = external global [32 x i32]
@xdst = external global [32 x i32]
@ptr = external global i32*
@dsrc = global [131072 x i32] zeroinitializer, align 32
@ddst = global [131072 x i32] zeroinitializer, align 32
@dptr = global i32* null
@lsrc = internal global [131072 x i32] zeroinitializer
@ldst = internal global [131072 x i32] zeroinitializer
@lptr = internal global i32* null
@ifunc = external global void ()*
@difunc = global void ()* null
@lifunc = internal global void ()* null
@lxsrc = internal global [32 x i32] zeroinitializer, align 32
@lxdst = internal global [32 x i32] zeroinitializer, align 32
@dxsrc = global [32 x i32] zeroinitializer, align 32
@dxdst = global [32 x i32] zeroinitializer, align 32

define void @foo00() nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @src, i32 0, i64 0), align 4
	store i32 %0, i32* getelementptr ([131072 x i32]* @dst, i32 0, i64 0), align 4
	ret void

; LINUX-64-STATIC: foo00:
; LINUX-64-STATIC: movl	src(%rip), %eax
; LINUX-64-STATIC: movl	%eax, dst
; LINUX-64-STATIC: ret
}

define void @fxo00() nounwind {
entry:
	%0 = load i32* getelementptr ([32 x i32]* @xsrc, i32 0, i64 0), align 4
	store i32 %0, i32* getelementptr ([32 x i32]* @xdst, i32 0, i64 0), align 4
	ret void

; LINUX-64-STATIC: fxo00:
; LINUX-64-STATIC: movl	xsrc(%rip), %eax
; LINUX-64-STATIC: movl	%eax, xdst
; LINUX-64-STATIC: ret
}

define void @foo01() nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @dst, i32 0, i32 0), i32** @ptr, align 8
	ret void
; LINUX-64-STATIC: foo01:
; LINUX-64-STATIC: movq	$dst, ptr
; LINUX-64-STATIC: ret
}

define void @fxo01() nounwind {
entry:
	store i32* getelementptr ([32 x i32]* @xdst, i32 0, i32 0), i32** @ptr, align 8
	ret void
; LINUX-64-STATIC: fxo01:
; LINUX-64-STATIC: movq	$xdst, ptr
; LINUX-64-STATIC: ret
}

define void @foo02() nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @src, i32 0, i64 0), align 4
	store i32 %1, i32* %0, align 4
	ret void
; LINUX-64-STATIC: foo02:
; LINUX-64-STATIC: movl    src(%rip), %
; LINUX-64-STATIC: movq    ptr(%rip), %
; LINUX-64-STATIC: movl
; LINUX-64-STATIC: ret
}

define void @fxo02() nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = load i32* getelementptr ([32 x i32]* @xsrc, i32 0, i64 0), align 4
	store i32 %1, i32* %0, align 4
; LINUX-64-STATIC: fxo02:
; LINUX-64-STATIC: movl    xsrc(%rip), %
; LINUX-64-STATIC: movq    ptr(%rip), %
; LINUX-64-STATIC: movl
; LINUX-64-STATIC: ret
	ret void
}

define void @foo03() nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 0), align 32
	store i32 %0, i32* getelementptr ([131072 x i32]* @ddst, i32 0, i64 0), align 32
	ret void
; LINUX-64-STATIC: foo03:
; LINUX-64-STATIC: movl    dsrc(%rip), %eax
; LINUX-64-STATIC: movl    %eax, ddst
; LINUX-64-STATIC: ret
}

define void @foo04() nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @ddst, i32 0, i32 0), i32** @dptr, align 8
	ret void
; LINUX-64-STATIC: foo04:
; LINUX-64-STATIC: movq    $ddst, dptr
; LINUX-64-STATIC: ret
}

define void @foo05() nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 0), align 32
	store i32 %1, i32* %0, align 4
	ret void
; LINUX-64-STATIC: foo05:
; LINUX-64-STATIC: movl    dsrc(%rip), %
; LINUX-64-STATIC: movq    dptr(%rip), %
; LINUX-64-STATIC: movl
; LINUX-64-STATIC: ret
}

define void @foo06() nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 0), align 4
	store i32 %0, i32* getelementptr ([131072 x i32]* @ldst, i32 0, i64 0), align 4
	ret void
; LINUX-64-STATIC: foo06:
; LINUX-64-STATIC: movl    lsrc(%rip), %eax
; LINUX-64-STATIC: movl    %eax, ldst(%rip)
; LINUX-64-STATIC: ret
}

define void @foo07() nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @ldst, i32 0, i32 0), i32** @lptr, align 8
	ret void
; LINUX-64-STATIC: foo07:
; LINUX-64-STATIC: movq    $ldst, lptr
; LINUX-64-STATIC: ret
}

define void @foo08() nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 0), align 4
	store i32 %1, i32* %0, align 4
	ret void
; LINUX-64-STATIC: foo08:
; LINUX-64-STATIC: movl    lsrc(%rip), %
; LINUX-64-STATIC: movq    lptr(%rip), %
; LINUX-64-STATIC: movl
; LINUX-64-STATIC: ret
}

define void @qux00() nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @src, i32 0, i64 16), align 4
	store i32 %0, i32* getelementptr ([131072 x i32]* @dst, i32 0, i64 16), align 4
	ret void
; LINUX-64-STATIC: qux00:
; LINUX-64-STATIC: movl    src+64(%rip), %eax
; LINUX-64-STATIC: movl    %eax, dst+64(%rip)
; LINUX-64-STATIC: ret
}

define void @qxx00() nounwind {
entry:
	%0 = load i32* getelementptr ([32 x i32]* @xsrc, i32 0, i64 16), align 4
	store i32 %0, i32* getelementptr ([32 x i32]* @xdst, i32 0, i64 16), align 4
	ret void
; LINUX-64-STATIC: qxx00:
; LINUX-64-STATIC: movl    xsrc+64(%rip), %eax
; LINUX-64-STATIC: movl    %eax, xdst+64(%rip)
; LINUX-64-STATIC: ret
}

define void @qux01() nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @dst, i32 0, i64 16), i32** @ptr, align 8
	ret void
; LINUX-64-STATIC: qux01:
; LINUX-64-STATIC: movq    $dst+64, ptr
; LINUX-64-STATIC: ret
}

define void @qxx01() nounwind {
entry:
	store i32* getelementptr ([32 x i32]* @xdst, i32 0, i64 16), i32** @ptr, align 8
	ret void
; LINUX-64-STATIC: qxx01:
; LINUX-64-STATIC: movq    $xdst+64, ptr
; LINUX-64-STATIC: ret
}

define void @qux02() nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @src, i32 0, i64 16), align 4
	%2 = getelementptr i32* %0, i64 16
	store i32 %1, i32* %2, align 4
; LINUX-64-STATIC: qux02:
; LINUX-64-STATIC: movl    src+64(%rip), %eax
; LINUX-64-STATIC: movq    ptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, 64(%rcx)
; LINUX-64-STATIC: ret
	ret void
}

define void @qxx02() nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = load i32* getelementptr ([32 x i32]* @xsrc, i32 0, i64 16), align 4
	%2 = getelementptr i32* %0, i64 16
	store i32 %1, i32* %2, align 4
; LINUX-64-STATIC: qxx02:
; LINUX-64-STATIC: movl    xsrc+64(%rip), %eax
; LINUX-64-STATIC: movq    ptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, 64(%rcx)
; LINUX-64-STATIC: ret
	ret void
}

define void @qux03() nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 16), align 32
	store i32 %0, i32* getelementptr ([131072 x i32]* @ddst, i32 0, i64 16), align 32
	ret void
; LINUX-64-STATIC: qux03:
; LINUX-64-STATIC: movl    dsrc+64(%rip), %eax
; LINUX-64-STATIC: movl    %eax, ddst+64(%rip)
; LINUX-64-STATIC: ret
}

define void @qux04() nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @ddst, i32 0, i64 16), i32** @dptr, align 8
	ret void
; LINUX-64-STATIC: qux04:
; LINUX-64-STATIC: movq    $ddst+64, dptr(%rip)
; LINUX-64-STATIC: ret
}

define void @qux05() nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 16), align 32
	%2 = getelementptr i32* %0, i64 16
	store i32 %1, i32* %2, align 4
; LINUX-64-STATIC: qux05:
; LINUX-64-STATIC: movl    dsrc+64(%rip), %eax
; LINUX-64-STATIC: movq    dptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, 64(%rcx)
; LINUX-64-STATIC: ret
	ret void
}

define void @qux06() nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 16), align 4
	store i32 %0, i32* getelementptr ([131072 x i32]* @ldst, i32 0, i64 16), align 4
	ret void
; LINUX-64-STATIC: qux06:
; LINUX-64-STATIC: movl    lsrc+64(%rip), %eax
; LINUX-64-STATIC: movl    %eax, ldst+64
; LINUX-64-STATIC: ret
}

define void @qux07() nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @ldst, i32 0, i64 16), i32** @lptr, align 8
	ret void
; LINUX-64-STATIC: qux07:
; LINUX-64-STATIC: movq    $ldst+64, lptr
; LINUX-64-STATIC: ret
}

define void @qux08() nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 16), align 4
	%2 = getelementptr i32* %0, i64 16
	store i32 %1, i32* %2, align 4
; LINUX-64-STATIC: qux08:
; LINUX-64-STATIC: movl    lsrc+64(%rip), %eax
; LINUX-64-STATIC: movq    lptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, 64(%rcx)
; LINUX-64-STATIC: ret
	ret void
}

define void @ind00(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32]* @src, i64 0, i64 %i
	%1 = load i32* %0, align 4
	%2 = getelementptr [131072 x i32]* @dst, i64 0, i64 %i
	store i32 %1, i32* %2, align 4
	ret void
; LINUX-64-STATIC: ind00:
; LINUX-64-STATIC: movl    src(,%rdi,4), %eax
; LINUX-64-STATIC: movl    %eax, dst(,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @ixd00(i64 %i) nounwind {
entry:
	%0 = getelementptr [32 x i32]* @xsrc, i64 0, i64 %i
	%1 = load i32* %0, align 4
	%2 = getelementptr [32 x i32]* @xdst, i64 0, i64 %i
	store i32 %1, i32* %2, align 4
	ret void
; LINUX-64-STATIC: ixd00:
; LINUX-64-STATIC: movl    xsrc(,%rdi,4), %eax
; LINUX-64-STATIC: movl    %eax, xdst(,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @ind01(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32]* @dst, i64 0, i64 %i
	store i32* %0, i32** @ptr, align 8
	ret void
; LINUX-64-STATIC: ind01:
; LINUX-64-STATIC: leaq    dst(,%rdi,4), %rax
; LINUX-64-STATIC: movq    %rax, ptr
; LINUX-64-STATIC: ret
}

define void @ixd01(i64 %i) nounwind {
entry:
	%0 = getelementptr [32 x i32]* @xdst, i64 0, i64 %i
	store i32* %0, i32** @ptr, align 8
	ret void
; LINUX-64-STATIC: ixd01:
; LINUX-64-STATIC: leaq    xdst(,%rdi,4), %rax
; LINUX-64-STATIC: movq    %rax, ptr
; LINUX-64-STATIC: ret
}

define void @ind02(i64 %i) nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = getelementptr [131072 x i32]* @src, i64 0, i64 %i
	%2 = load i32* %1, align 4
	%3 = getelementptr i32* %0, i64 %i
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC: ind02:
; LINUX-64-STATIC: movl    src(,%rdi,4), %eax
; LINUX-64-STATIC: movq    ptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, (%rcx,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @ixd02(i64 %i) nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = getelementptr [32 x i32]* @xsrc, i64 0, i64 %i
	%2 = load i32* %1, align 4
	%3 = getelementptr i32* %0, i64 %i
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC: ixd02:
; LINUX-64-STATIC: movl    xsrc(,%rdi,4), %eax
; LINUX-64-STATIC: movq    ptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, (%rcx,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @ind03(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32]* @dsrc, i64 0, i64 %i
	%1 = load i32* %0, align 4
	%2 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %i
	store i32 %1, i32* %2, align 4
	ret void
; LINUX-64-STATIC: ind03:
; LINUX-64-STATIC: movl    dsrc(,%rdi,4), %eax
; LINUX-64-STATIC: movl    %eax, ddst(,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @ind04(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %i
	store i32* %0, i32** @dptr, align 8
	ret void
; LINUX-64-STATIC: ind04:
; LINUX-64-STATIC: leaq    ddst(,%rdi,4), %rax
; LINUX-64-STATIC: movq    %rax, dptr
; LINUX-64-STATIC: ret
}

define void @ind05(i64 %i) nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = getelementptr [131072 x i32]* @dsrc, i64 0, i64 %i
	%2 = load i32* %1, align 4
	%3 = getelementptr i32* %0, i64 %i
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC: ind05:
; LINUX-64-STATIC: movl    dsrc(,%rdi,4), %eax
; LINUX-64-STATIC: movq    dptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, (%rcx,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @ind06(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32]* @lsrc, i64 0, i64 %i
	%1 = load i32* %0, align 4
	%2 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %i
	store i32 %1, i32* %2, align 4
	ret void
; LINUX-64-STATIC: ind06:
; LINUX-64-STATIC: movl    lsrc(,%rdi,4), %eax
; LINUX-64-STATIC: movl    %eax, ldst(,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @ind07(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %i
	store i32* %0, i32** @lptr, align 8
	ret void
; LINUX-64-STATIC: ind07:
; LINUX-64-STATIC: leaq    ldst(,%rdi,4), %rax
; LINUX-64-STATIC: movq    %rax, lptr
; LINUX-64-STATIC: ret
}

define void @ind08(i64 %i) nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = getelementptr [131072 x i32]* @lsrc, i64 0, i64 %i
	%2 = load i32* %1, align 4
	%3 = getelementptr i32* %0, i64 %i
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC: ind08:
; LINUX-64-STATIC: movl    lsrc(,%rdi,4), %eax
; LINUX-64-STATIC: movq    lptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, (%rcx,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @off00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @src, i64 0, i64 %0
	%2 = load i32* %1, align 4
	%3 = getelementptr [131072 x i32]* @dst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC: off00:
; LINUX-64-STATIC: movl    src+64(,%rdi,4), %eax
; LINUX-64-STATIC: movl    %eax, dst+64(,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @oxf00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [32 x i32]* @xsrc, i64 0, i64 %0
	%2 = load i32* %1, align 4
	%3 = getelementptr [32 x i32]* @xdst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC: oxf00:
; LINUX-64-STATIC: movl    xsrc+64(,%rdi,4), %eax
; LINUX-64-STATIC: movl    %eax, xdst+64(,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @off01(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 16
	%0 = getelementptr [131072 x i32]* @dst, i64 0, i64 %.sum
	store i32* %0, i32** @ptr, align 8
	ret void
; LINUX-64-STATIC: off01:
; LINUX-64-STATIC: leaq    dst+64(,%rdi,4), %rax
; LINUX-64-STATIC: movq    %rax, ptr
; LINUX-64-STATIC: ret
}

define void @oxf01(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 16
	%0 = getelementptr [32 x i32]* @xdst, i64 0, i64 %.sum
	store i32* %0, i32** @ptr, align 8
	ret void
; LINUX-64-STATIC: oxf01:
; LINUX-64-STATIC: leaq    xdst+64(,%rdi,4), %rax
; LINUX-64-STATIC: movq    %rax, ptr
; LINUX-64-STATIC: ret
}

define void @off02(i64 %i) nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr [131072 x i32]* @src, i64 0, i64 %1
	%3 = load i32* %2, align 4
	%4 = getelementptr i32* %0, i64 %1
	store i32 %3, i32* %4, align 4
	ret void
; LINUX-64-STATIC: off02:
; LINUX-64-STATIC: movl    src+64(,%rdi,4), %eax
; LINUX-64-STATIC: movq    ptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, 64(%rcx,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @oxf02(i64 %i) nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr [32 x i32]* @xsrc, i64 0, i64 %1
	%3 = load i32* %2, align 4
	%4 = getelementptr i32* %0, i64 %1
	store i32 %3, i32* %4, align 4
	ret void
; LINUX-64-STATIC: oxf02:
; LINUX-64-STATIC: movl    xsrc+64(,%rdi,4), %eax
; LINUX-64-STATIC: movq    ptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, 64(%rcx,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @off03(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @dsrc, i64 0, i64 %0
	%2 = load i32* %1, align 4
	%3 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC: off03:
; LINUX-64-STATIC: movl    dsrc+64(,%rdi,4), %eax
; LINUX-64-STATIC: movl    %eax, ddst+64(,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @off04(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 16
	%0 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %.sum
	store i32* %0, i32** @dptr, align 8
	ret void
; LINUX-64-STATIC: off04:
; LINUX-64-STATIC: leaq    ddst+64(,%rdi,4), %rax
; LINUX-64-STATIC: movq    %rax, dptr
; LINUX-64-STATIC: ret
}

define void @off05(i64 %i) nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr [131072 x i32]* @dsrc, i64 0, i64 %1
	%3 = load i32* %2, align 4
	%4 = getelementptr i32* %0, i64 %1
	store i32 %3, i32* %4, align 4
	ret void
; LINUX-64-STATIC: off05:
; LINUX-64-STATIC: movl    dsrc+64(,%rdi,4), %eax
; LINUX-64-STATIC: movq    dptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, 64(%rcx,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @off06(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @lsrc, i64 0, i64 %0
	%2 = load i32* %1, align 4
	%3 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC: off06:
; LINUX-64-STATIC: movl    lsrc+64(,%rdi,4), %eax
; LINUX-64-STATIC: movl    %eax, ldst+64(,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @off07(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 16
	%0 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %.sum
	store i32* %0, i32** @lptr, align 8
	ret void
; LINUX-64-STATIC: off07:
; LINUX-64-STATIC: leaq    ldst+64(,%rdi,4), %rax
; LINUX-64-STATIC: movq    %rax, lptr
; LINUX-64-STATIC: ret
}

define void @off08(i64 %i) nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr [131072 x i32]* @lsrc, i64 0, i64 %1
	%3 = load i32* %2, align 4
	%4 = getelementptr i32* %0, i64 %1
	store i32 %3, i32* %4, align 4
	ret void
; LINUX-64-STATIC: off08:
; LINUX-64-STATIC: movl    lsrc+64(,%rdi,4), %eax
; LINUX-64-STATIC: movq    lptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, 64(%rcx,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @moo00(i64 %i) nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @src, i32 0, i64 65536), align 4
	store i32 %0, i32* getelementptr ([131072 x i32]* @dst, i32 0, i64 65536), align 4
	ret void
; LINUX-64-STATIC: moo00:
; LINUX-64-STATIC: movl    src+262144(%rip), %eax
; LINUX-64-STATIC: movl    %eax, dst+262144(%rip)
; LINUX-64-STATIC: ret
}

define void @moo01(i64 %i) nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @dst, i32 0, i64 65536), i32** @ptr, align 8
	ret void
; LINUX-64-STATIC: moo01:
; LINUX-64-STATIC: movq    $dst+262144, ptr(%rip)
; LINUX-64-STATIC: ret
}

define void @moo02(i64 %i) nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @src, i32 0, i64 65536), align 4
	%2 = getelementptr i32* %0, i64 65536
	store i32 %1, i32* %2, align 4
	ret void
; LINUX-64-STATIC: moo02:
; LINUX-64-STATIC: movl    src+262144(%rip), %eax
; LINUX-64-STATIC: movq    ptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, 262144(%rcx)
; LINUX-64-STATIC: ret
}

define void @moo03(i64 %i) nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 65536), align 32
	store i32 %0, i32* getelementptr ([131072 x i32]* @ddst, i32 0, i64 65536), align 32
	ret void
; LINUX-64-STATIC: moo03:
; LINUX-64-STATIC: movl    dsrc+262144(%rip), %eax
; LINUX-64-STATIC: movl    %eax, ddst+262144(%rip)
; LINUX-64-STATIC: ret
}

define void @moo04(i64 %i) nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @ddst, i32 0, i64 65536), i32** @dptr, align 8
	ret void
; LINUX-64-STATIC: moo04:
; LINUX-64-STATIC: movq    $ddst+262144, dptr
; LINUX-64-STATIC: ret
}

define void @moo05(i64 %i) nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 65536), align 32
	%2 = getelementptr i32* %0, i64 65536
	store i32 %1, i32* %2, align 4
	ret void
; LINUX-64-STATIC: moo05:
; LINUX-64-STATIC: movl    dsrc+262144(%rip), %eax
; LINUX-64-STATIC: movq    dptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, 262144(%rcx)
; LINUX-64-STATIC: ret
}

define void @moo06(i64 %i) nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 65536), align 4
	store i32 %0, i32* getelementptr ([131072 x i32]* @ldst, i32 0, i64 65536), align 4
	ret void
; LINUX-64-STATIC: moo06:
; LINUX-64-STATIC: movl    lsrc+262144(%rip), %eax
; LINUX-64-STATIC: movl    %eax, ldst+262144(%rip)
; LINUX-64-STATIC: ret
}

define void @moo07(i64 %i) nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @ldst, i32 0, i64 65536), i32** @lptr, align 8
	ret void
; LINUX-64-STATIC: moo07:
; LINUX-64-STATIC: movq    $ldst+262144, lptr
; LINUX-64-STATIC: ret
}

define void @moo08(i64 %i) nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 65536), align 4
	%2 = getelementptr i32* %0, i64 65536
	store i32 %1, i32* %2, align 4
	ret void
; LINUX-64-STATIC: moo08:
; LINUX-64-STATIC: movl    lsrc+262144(%rip), %eax
; LINUX-64-STATIC: movq    lptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, 262144(%rcx)
; LINUX-64-STATIC: ret
}

define void @big00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @src, i64 0, i64 %0
	%2 = load i32* %1, align 4
	%3 = getelementptr [131072 x i32]* @dst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC: big00:
; LINUX-64-STATIC: movl    src+262144(,%rdi,4), %eax
; LINUX-64-STATIC: movl    %eax, dst+262144(,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @big01(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 65536
	%0 = getelementptr [131072 x i32]* @dst, i64 0, i64 %.sum
	store i32* %0, i32** @ptr, align 8
	ret void
; LINUX-64-STATIC: big01:
; LINUX-64-STATIC: leaq    dst+262144(,%rdi,4), %rax
; LINUX-64-STATIC: movq    %rax, ptr(%rip)
; LINUX-64-STATIC: ret
}

define void @big02(i64 %i) nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr [131072 x i32]* @src, i64 0, i64 %1
	%3 = load i32* %2, align 4
	%4 = getelementptr i32* %0, i64 %1
	store i32 %3, i32* %4, align 4
	ret void
; LINUX-64-STATIC: big02:
; LINUX-64-STATIC: movl    src+262144(,%rdi,4), %eax
; LINUX-64-STATIC: movq    ptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, 262144(%rcx,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @big03(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @dsrc, i64 0, i64 %0
	%2 = load i32* %1, align 4
	%3 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC: big03:
; LINUX-64-STATIC: movl    dsrc+262144(,%rdi,4), %eax
; LINUX-64-STATIC: movl    %eax, ddst+262144(,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @big04(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 65536
	%0 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %.sum
	store i32* %0, i32** @dptr, align 8
	ret void
; LINUX-64-STATIC: big04:
; LINUX-64-STATIC: leaq    ddst+262144(,%rdi,4), %rax
; LINUX-64-STATIC: movq    %rax, dptr
; LINUX-64-STATIC: ret
}

define void @big05(i64 %i) nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr [131072 x i32]* @dsrc, i64 0, i64 %1
	%3 = load i32* %2, align 4
	%4 = getelementptr i32* %0, i64 %1
	store i32 %3, i32* %4, align 4
	ret void
; LINUX-64-STATIC: big05:
; LINUX-64-STATIC: movl    dsrc+262144(,%rdi,4), %eax
; LINUX-64-STATIC: movq    dptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, 262144(%rcx,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @big06(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @lsrc, i64 0, i64 %0
	%2 = load i32* %1, align 4
	%3 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC: big06:
; LINUX-64-STATIC: movl    lsrc+262144(,%rdi,4), %eax
; LINUX-64-STATIC: movl    %eax, ldst+262144(,%rdi,4)
; LINUX-64-STATIC: ret
}

define void @big07(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 65536
	%0 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %.sum
	store i32* %0, i32** @lptr, align 8
	ret void
; LINUX-64-STATIC: big07:
; LINUX-64-STATIC: leaq    ldst+262144(,%rdi,4), %rax
; LINUX-64-STATIC: movq    %rax, lptr
; LINUX-64-STATIC: ret
}

define void @big08(i64 %i) nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr [131072 x i32]* @lsrc, i64 0, i64 %1
	%3 = load i32* %2, align 4
	%4 = getelementptr i32* %0, i64 %1
	store i32 %3, i32* %4, align 4
	ret void
; LINUX-64-STATIC: big08:
; LINUX-64-STATIC: movl    lsrc+262144(,%rdi,4), %eax
; LINUX-64-STATIC: movq    lptr(%rip), %rcx
; LINUX-64-STATIC: movl    %eax, 262144(%rcx,%rdi,4)
; LINUX-64-STATIC: ret
}

define i8* @bar00() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @src to i8*)
; LINUX-64-STATIC: bar00:
; LINUX-64-STATIC: movl    $src, %eax
; LINUX-64-STATIC: ret
}

define i8* @bxr00() nounwind {
entry:
	ret i8* bitcast ([32 x i32]* @xsrc to i8*)
; LINUX-64-STATIC: bxr00:
; LINUX-64-STATIC: movl    $xsrc, %eax
; LINUX-64-STATIC: ret
}

define i8* @bar01() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @dst to i8*)
; LINUX-64-STATIC: bar01:
; LINUX-64-STATIC: movl    $dst, %eax
; LINUX-64-STATIC: ret
}

define i8* @bxr01() nounwind {
entry:
	ret i8* bitcast ([32 x i32]* @xdst to i8*)
; LINUX-64-STATIC: bxr01:
; LINUX-64-STATIC: movl    $xdst, %eax
; LINUX-64-STATIC: ret
}

define i8* @bar02() nounwind {
entry:
	ret i8* bitcast (i32** @ptr to i8*)
; LINUX-64-STATIC: bar02:
; LINUX-64-STATIC: movl    $ptr, %eax
; LINUX-64-STATIC: ret
}

define i8* @bar03() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @dsrc to i8*)
; LINUX-64-STATIC: bar03:
; LINUX-64-STATIC: movl    $dsrc, %eax
; LINUX-64-STATIC: ret
}

define i8* @bar04() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @ddst to i8*)
; LINUX-64-STATIC: bar04:
; LINUX-64-STATIC: movl    $ddst, %eax
; LINUX-64-STATIC: ret
}

define i8* @bar05() nounwind {
entry:
	ret i8* bitcast (i32** @dptr to i8*)
; LINUX-64-STATIC: bar05:
; LINUX-64-STATIC: movl    $dptr, %eax
; LINUX-64-STATIC: ret
}

define i8* @bar06() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @lsrc to i8*)
; LINUX-64-STATIC: bar06:
; LINUX-64-STATIC: movl    $lsrc, %eax
; LINUX-64-STATIC: ret
}

define i8* @bar07() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @ldst to i8*)
; LINUX-64-STATIC: bar07:
; LINUX-64-STATIC: movl    $ldst, %eax
; LINUX-64-STATIC: ret
}

define i8* @bar08() nounwind {
entry:
	ret i8* bitcast (i32** @lptr to i8*)
; LINUX-64-STATIC: bar08:
; LINUX-64-STATIC: movl    $lptr, %eax
; LINUX-64-STATIC: ret
}

define i8* @har00() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @src to i8*)
; LINUX-64-STATIC: har00:
; LINUX-64-STATIC: movl    $src, %eax
; LINUX-64-STATIC: ret
}

define i8* @hxr00() nounwind {
entry:
	ret i8* bitcast ([32 x i32]* @xsrc to i8*)
; LINUX-64-STATIC: hxr00:
; LINUX-64-STATIC: movl    $xsrc, %eax
; LINUX-64-STATIC: ret
}

define i8* @har01() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @dst to i8*)
; LINUX-64-STATIC: har01:
; LINUX-64-STATIC: movl    $dst, %eax
; LINUX-64-STATIC: ret
}

define i8* @hxr01() nounwind {
entry:
	ret i8* bitcast ([32 x i32]* @xdst to i8*)
; LINUX-64-STATIC: hxr01:
; LINUX-64-STATIC: movl    $xdst, %eax
; LINUX-64-STATIC: ret
}

define i8* @har02() nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = bitcast i32* %0 to i8*
	ret i8* %1
; LINUX-64-STATIC: har02:
; LINUX-64-STATIC: movq    ptr(%rip), %rax
; LINUX-64-STATIC: ret
}

define i8* @har03() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @dsrc to i8*)
; LINUX-64-STATIC: har03:
; LINUX-64-STATIC: movl    $dsrc, %eax
; LINUX-64-STATIC: ret
}

define i8* @har04() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @ddst to i8*)
; LINUX-64-STATIC: har04:
; LINUX-64-STATIC: movl    $ddst, %eax
; LINUX-64-STATIC: ret
}

define i8* @har05() nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = bitcast i32* %0 to i8*
	ret i8* %1
; LINUX-64-STATIC: har05:
; LINUX-64-STATIC: movq    dptr(%rip), %rax
; LINUX-64-STATIC: ret
}

define i8* @har06() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @lsrc to i8*)
; LINUX-64-STATIC: har06:
; LINUX-64-STATIC: movl    $lsrc, %eax
; LINUX-64-STATIC: ret
}

define i8* @har07() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @ldst to i8*)
; LINUX-64-STATIC: har07:
; LINUX-64-STATIC: movl    $ldst, %eax
; LINUX-64-STATIC: ret
}

define i8* @har08() nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = bitcast i32* %0 to i8*
	ret i8* %1
; LINUX-64-STATIC: har08:
; LINUX-64-STATIC: movq    lptr(%rip), %rax
; LINUX-64-STATIC: ret
}

define i8* @bat00() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @src, i32 0, i64 16) to i8*)
; LINUX-64-STATIC: bat00:
; LINUX-64-STATIC: movl    $src+64, %eax
; LINUX-64-STATIC: ret
}

define i8* @bxt00() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([32 x i32]* @xsrc, i32 0, i64 16) to i8*)
; LINUX-64-STATIC: bxt00:
; LINUX-64-STATIC: movl    $xsrc+64, %eax
; LINUX-64-STATIC: ret
}

define i8* @bat01() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @dst, i32 0, i64 16) to i8*)
; LINUX-64-STATIC: bat01:
; LINUX-64-STATIC: movl    $dst+64, %eax
; LINUX-64-STATIC: ret
}

define i8* @bxt01() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([32 x i32]* @xdst, i32 0, i64 16) to i8*)
; LINUX-64-STATIC: bxt01:
; LINUX-64-STATIC: movl    $xdst+64, %eax
; LINUX-64-STATIC: ret
}

define i8* @bat02() nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = getelementptr i32* %0, i64 16
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: bat02:
; LINUX-64-STATIC: movq    ptr(%rip), %rax
; LINUX-64-STATIC: addq    $64, %rax
; LINUX-64-STATIC: ret
}

define i8* @bat03() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 16) to i8*)
; LINUX-64-STATIC: bat03:
; LINUX-64-STATIC: movl    $dsrc+64, %eax
; LINUX-64-STATIC: ret
}

define i8* @bat04() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @ddst, i32 0, i64 16) to i8*)
; LINUX-64-STATIC: bat04:
; LINUX-64-STATIC: movl    $ddst+64, %eax
; LINUX-64-STATIC: ret
}

define i8* @bat05() nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = getelementptr i32* %0, i64 16
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: bat05:
; LINUX-64-STATIC: movq    dptr(%rip), %rax
; LINUX-64-STATIC: addq    $64, %rax
; LINUX-64-STATIC: ret
}

define i8* @bat06() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 16) to i8*)
; LINUX-64-STATIC: bat06:
; LINUX-64-STATIC: movl    $lsrc+64, %eax
; LINUX-64-STATIC: ret
}

define i8* @bat07() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @ldst, i32 0, i64 16) to i8*)
; LINUX-64-STATIC: bat07:
; LINUX-64-STATIC: movl    $ldst+64, %eax
; LINUX-64-STATIC: ret
}

define i8* @bat08() nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = getelementptr i32* %0, i64 16
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: bat08:
; LINUX-64-STATIC: movq    lptr(%rip), %rax
; LINUX-64-STATIC: addq    $64, %rax
; LINUX-64-STATIC: ret
}

define i8* @bam00() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @src, i32 0, i64 65536) to i8*)
; LINUX-64-STATIC: bam00:
; LINUX-64-STATIC: movl    $src+262144, %eax
; LINUX-64-STATIC: ret
}

define i8* @bam01() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @dst, i32 0, i64 65536) to i8*)
; LINUX-64-STATIC: bam01:
; LINUX-64-STATIC: movl    $dst+262144, %eax
; LINUX-64-STATIC: ret
}

define i8* @bxm01() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([32 x i32]* @xdst, i32 0, i64 65536) to i8*)
; LINUX-64-STATIC: bxm01:
; LINUX-64-STATIC: movl    $xdst+262144, %eax
; LINUX-64-STATIC: ret
}

define i8* @bam02() nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = getelementptr i32* %0, i64 65536
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: bam02:
; LINUX-64-STATIC: movl    $262144, %eax
; LINUX-64-STATIC: addq    ptr(%rip), %rax
; LINUX-64-STATIC: ret
}

define i8* @bam03() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 65536) to i8*)
; LINUX-64-STATIC: bam03:
; LINUX-64-STATIC: movl    $dsrc+262144, %eax
; LINUX-64-STATIC: ret
}

define i8* @bam04() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @ddst, i32 0, i64 65536) to i8*)
; LINUX-64-STATIC: bam04:
; LINUX-64-STATIC: movl    $ddst+262144, %eax
; LINUX-64-STATIC: ret
}

define i8* @bam05() nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = getelementptr i32* %0, i64 65536
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: bam05:
; LINUX-64-STATIC: movl    $262144, %eax
; LINUX-64-STATIC: addq    dptr(%rip), %rax
; LINUX-64-STATIC: ret
}

define i8* @bam06() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 65536) to i8*)
; LINUX-64-STATIC: bam06:
; LINUX-64-STATIC: movl    $lsrc+262144, %eax
; LINUX-64-STATIC: ret
}

define i8* @bam07() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @ldst, i32 0, i64 65536) to i8*)
; LINUX-64-STATIC: bam07:
; LINUX-64-STATIC: movl    $ldst+262144, %eax
; LINUX-64-STATIC: ret
}

define i8* @bam08() nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = getelementptr i32* %0, i64 65536
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: bam08:
; LINUX-64-STATIC: movl    $262144, %eax
; LINUX-64-STATIC: addq    lptr(%rip), %rax
; LINUX-64-STATIC: ret
}

define i8* @cat00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @src, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cat00:
; LINUX-64-STATIC: leaq    src+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cxt00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [32 x i32]* @xsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cxt00:
; LINUX-64-STATIC: leaq    xsrc+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cat01(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @dst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cat01:
; LINUX-64-STATIC: leaq    dst+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cxt01(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [32 x i32]* @xdst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cxt01:
; LINUX-64-STATIC: leaq    xdst+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cat02(i64 %i) nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
; LINUX-64-STATIC: cat02:
; LINUX-64-STATIC: movq    ptr(%rip), %rax
; LINUX-64-STATIC: leaq    64(%rax,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cat03(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @dsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cat03:
; LINUX-64-STATIC: leaq    dsrc+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cat04(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cat04:
; LINUX-64-STATIC: leaq    ddst+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cat05(i64 %i) nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
; LINUX-64-STATIC: cat05:
; LINUX-64-STATIC: movq    dptr(%rip), %rax
; LINUX-64-STATIC: leaq    64(%rax,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cat06(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @lsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cat06:
; LINUX-64-STATIC: leaq    lsrc+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cat07(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cat07:
; LINUX-64-STATIC: leaq    ldst+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cat08(i64 %i) nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
; LINUX-64-STATIC: cat08:
; LINUX-64-STATIC: movq    lptr(%rip), %rax
; LINUX-64-STATIC: leaq    64(%rax,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cam00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @src, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cam00:
; LINUX-64-STATIC: leaq    src+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cxm00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [32 x i32]* @xsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cxm00:
; LINUX-64-STATIC: leaq    xsrc+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cam01(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @dst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cam01:
; LINUX-64-STATIC: leaq    dst+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cxm01(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [32 x i32]* @xdst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cxm01:
; LINUX-64-STATIC: leaq    xdst+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cam02(i64 %i) nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
; LINUX-64-STATIC: cam02:
; LINUX-64-STATIC: movq    ptr(%rip), %rax
; LINUX-64-STATIC: leaq    262144(%rax,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cam03(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @dsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cam03:
; LINUX-64-STATIC: leaq    dsrc+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cam04(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cam04:
; LINUX-64-STATIC: leaq    ddst+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cam05(i64 %i) nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
; LINUX-64-STATIC: cam05:
; LINUX-64-STATIC: movq    dptr(%rip), %rax
; LINUX-64-STATIC: leaq    262144(%rax,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cam06(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @lsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cam06:
; LINUX-64-STATIC: leaq    lsrc+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cam07(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC: cam07:
; LINUX-64-STATIC: leaq    ldst+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define i8* @cam08(i64 %i) nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
; LINUX-64-STATIC: cam08:
; LINUX-64-STATIC: movq    lptr(%rip), %rax
; LINUX-64-STATIC: leaq    262144(%rax,%rdi,4), %rax
; LINUX-64-STATIC: ret
}

define void @lcallee() nounwind {
entry:
	tail call void @x() nounwind
	tail call void @x() nounwind
	tail call void @x() nounwind
	tail call void @x() nounwind
	tail call void @x() nounwind
	tail call void @x() nounwind
	tail call void @x() nounwind
	ret void
; LINUX-64-STATIC: lcallee:
; LINUX-64-STATIC: call    x
; LINUX-64-STATIC: call    x
; LINUX-64-STATIC: call    x
; LINUX-64-STATIC: call    x
; LINUX-64-STATIC: call    x
; LINUX-64-STATIC: call    x
; LINUX-64-STATIC: call    x
; LINUX-64-STATIC: ret
}

declare void @x()

define internal void @dcallee() nounwind {
entry:
	tail call void @y() nounwind
	tail call void @y() nounwind
	tail call void @y() nounwind
	tail call void @y() nounwind
	tail call void @y() nounwind
	tail call void @y() nounwind
	tail call void @y() nounwind
	ret void
; LINUX-64-STATIC: dcallee:
; LINUX-64-STATIC: call    y
; LINUX-64-STATIC: call    y
; LINUX-64-STATIC: call    y
; LINUX-64-STATIC: call    y
; LINUX-64-STATIC: call    y
; LINUX-64-STATIC: call    y
; LINUX-64-STATIC: call    y
; LINUX-64-STATIC: ret
}

declare void @y()

define void ()* @address() nounwind {
entry:
	ret void ()* @callee
; LINUX-64-STATIC: address:
; LINUX-64-STATIC: movl    $callee, %eax
; LINUX-64-STATIC: ret
}

declare void @callee()

define void ()* @laddress() nounwind {
entry:
	ret void ()* @lcallee
; LINUX-64-STATIC: laddress:
; LINUX-64-STATIC: movl    $lcallee, %eax
; LINUX-64-STATIC: ret
}

define void ()* @daddress() nounwind {
entry:
	ret void ()* @dcallee
; LINUX-64-STATIC: daddress:
; LINUX-64-STATIC: movl    $dcallee, %eax
; LINUX-64-STATIC: ret
}

define void @caller() nounwind {
entry:
	tail call void @callee() nounwind
	tail call void @callee() nounwind
	ret void
; LINUX-64-STATIC: caller:
; LINUX-64-STATIC: call    callee
; LINUX-64-STATIC: call    callee
; LINUX-64-STATIC: ret
}

define void @dcaller() nounwind {
entry:
	tail call void @dcallee() nounwind
	tail call void @dcallee() nounwind
	ret void
; LINUX-64-STATIC: dcaller:
; LINUX-64-STATIC: call    dcallee
; LINUX-64-STATIC: call    dcallee
; LINUX-64-STATIC: ret
}

define void @lcaller() nounwind {
entry:
	tail call void @lcallee() nounwind
	tail call void @lcallee() nounwind
	ret void
; LINUX-64-STATIC: lcaller:
; LINUX-64-STATIC: call    lcallee
; LINUX-64-STATIC: call    lcallee
; LINUX-64-STATIC: ret
}

define void @tailcaller() nounwind {
entry:
	tail call void @callee() nounwind
	ret void
; LINUX-64-STATIC: tailcaller:
; LINUX-64-STATIC: call    callee
; LINUX-64-STATIC: ret
}

define void @dtailcaller() nounwind {
entry:
	tail call void @dcallee() nounwind
	ret void
; LINUX-64-STATIC: dtailcaller:
; LINUX-64-STATIC: call    dcallee
; LINUX-64-STATIC: ret
}

define void @ltailcaller() nounwind {
entry:
	tail call void @lcallee() nounwind
	ret void
; LINUX-64-STATIC: ltailcaller:
; LINUX-64-STATIC: call    lcallee
; LINUX-64-STATIC: ret
}

define void @icaller() nounwind {
entry:
	%0 = load void ()** @ifunc, align 8
	tail call void %0() nounwind
	%1 = load void ()** @ifunc, align 8
	tail call void %1() nounwind
	ret void
; LINUX-64-STATIC: icaller:
; LINUX-64-STATIC: call    *ifunc
; LINUX-64-STATIC: call    *ifunc
; LINUX-64-STATIC: ret
}

define void @dicaller() nounwind {
entry:
	%0 = load void ()** @difunc, align 8
	tail call void %0() nounwind
	%1 = load void ()** @difunc, align 8
	tail call void %1() nounwind
	ret void
; LINUX-64-STATIC: dicaller:
; LINUX-64-STATIC: call    *difunc
; LINUX-64-STATIC: call    *difunc
; LINUX-64-STATIC: ret
}

define void @licaller() nounwind {
entry:
	%0 = load void ()** @lifunc, align 8
	tail call void %0() nounwind
	%1 = load void ()** @lifunc, align 8
	tail call void %1() nounwind
	ret void
; LINUX-64-STATIC: licaller:
; LINUX-64-STATIC: call    *lifunc
; LINUX-64-STATIC: call    *lifunc
; LINUX-64-STATIC: ret
}

define void @itailcaller() nounwind {
entry:
	%0 = load void ()** @ifunc, align 8
	tail call void %0() nounwind
	%1 = load void ()** @ifunc, align 8
	tail call void %1() nounwind
	ret void
; LINUX-64-STATIC: itailcaller:
; LINUX-64-STATIC: call    *ifunc
; LINUX-64-STATIC: call    *ifunc
; LINUX-64-STATIC: ret
}

define void @ditailcaller() nounwind {
entry:
	%0 = load void ()** @difunc, align 8
	tail call void %0() nounwind
	ret void
; LINUX-64-STATIC: ditailcaller:
; LINUX-64-STATIC: call    *difunc
; LINUX-64-STATIC: ret
}

define void @litailcaller() nounwind {
entry:
	%0 = load void ()** @lifunc, align 8
	tail call void %0() nounwind
	ret void
; LINUX-64-STATIC: litailcaller:
; LINUX-64-STATIC: call    *lifunc
; LINUX-64-STATIC: ret
}
