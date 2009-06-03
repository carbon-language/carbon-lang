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
; RUN: llvm-as < %s | llc -mtriple=x86_64-unknown-linux-gnu -march=x86-64 -relocation-model=static -code-model=small > %t
; RUN: not grep leal %t
; RUN: grep movl %t | count 91
; RUN: not grep addl %t
; RUN: not grep subl %t
; RUN: grep leaq %t | count 70
; RUN: grep movq %t | count 56
; RUN: grep addq %t | count 20
; RUN: grep subq %t | count 14
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
; RUN: grep {%rip} %t | count 139
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
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin -march=x86-64 -relocation-model=static -code-model=small > %t
; RUN: not grep leal %t
; RUN: grep movl %t | count 91
; RUN: not grep addl %t
; RUN: not grep subl %t
; RUN: grep leaq %t | count 70
; RUN: grep movq %t | count 56
; RUN: grep addq %t | count 20
; RUN: grep subq %t | count 14
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
; RUN: grep {%rip} %t | count 139
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
}

define void @fxo00() nounwind {
entry:
	%0 = load i32* getelementptr ([32 x i32]* @xsrc, i32 0, i64 0), align 4
	store i32 %0, i32* getelementptr ([32 x i32]* @xdst, i32 0, i64 0), align 4
	ret void
}

define void @foo01() nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @dst, i32 0, i32 0), i32** @ptr, align 8
	ret void
}

define void @fxo01() nounwind {
entry:
	store i32* getelementptr ([32 x i32]* @xdst, i32 0, i32 0), i32** @ptr, align 8
	ret void
}

define void @foo02() nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @src, i32 0, i64 0), align 4
	store i32 %1, i32* %0, align 4
	ret void
}

define void @fxo02() nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = load i32* getelementptr ([32 x i32]* @xsrc, i32 0, i64 0), align 4
	store i32 %1, i32* %0, align 4
	ret void
}

define void @foo03() nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 0), align 32
	store i32 %0, i32* getelementptr ([131072 x i32]* @ddst, i32 0, i64 0), align 32
	ret void
}

define void @foo04() nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @ddst, i32 0, i32 0), i32** @dptr, align 8
	ret void
}

define void @foo05() nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 0), align 32
	store i32 %1, i32* %0, align 4
	ret void
}

define void @foo06() nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 0), align 4
	store i32 %0, i32* getelementptr ([131072 x i32]* @ldst, i32 0, i64 0), align 4
	ret void
}

define void @foo07() nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @ldst, i32 0, i32 0), i32** @lptr, align 8
	ret void
}

define void @foo08() nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 0), align 4
	store i32 %1, i32* %0, align 4
	ret void
}

define void @qux00() nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @src, i32 0, i64 16), align 4
	store i32 %0, i32* getelementptr ([131072 x i32]* @dst, i32 0, i64 16), align 4
	ret void
}

define void @qxx00() nounwind {
entry:
	%0 = load i32* getelementptr ([32 x i32]* @xsrc, i32 0, i64 16), align 4
	store i32 %0, i32* getelementptr ([32 x i32]* @xdst, i32 0, i64 16), align 4
	ret void
}

define void @qux01() nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @dst, i32 0, i64 16), i32** @ptr, align 8
	ret void
}

define void @qxx01() nounwind {
entry:
	store i32* getelementptr ([32 x i32]* @xdst, i32 0, i64 16), i32** @ptr, align 8
	ret void
}

define void @qux02() nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @src, i32 0, i64 16), align 4
	%2 = getelementptr i32* %0, i64 16
	store i32 %1, i32* %2, align 4
	ret void
}

define void @qxx02() nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = load i32* getelementptr ([32 x i32]* @xsrc, i32 0, i64 16), align 4
	%2 = getelementptr i32* %0, i64 16
	store i32 %1, i32* %2, align 4
	ret void
}

define void @qux03() nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 16), align 32
	store i32 %0, i32* getelementptr ([131072 x i32]* @ddst, i32 0, i64 16), align 32
	ret void
}

define void @qux04() nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @ddst, i32 0, i64 16), i32** @dptr, align 8
	ret void
}

define void @qux05() nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 16), align 32
	%2 = getelementptr i32* %0, i64 16
	store i32 %1, i32* %2, align 4
	ret void
}

define void @qux06() nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 16), align 4
	store i32 %0, i32* getelementptr ([131072 x i32]* @ldst, i32 0, i64 16), align 4
	ret void
}

define void @qux07() nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @ldst, i32 0, i64 16), i32** @lptr, align 8
	ret void
}

define void @qux08() nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 16), align 4
	%2 = getelementptr i32* %0, i64 16
	store i32 %1, i32* %2, align 4
	ret void
}

define void @ind00(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32]* @src, i64 0, i64 %i
	%1 = load i32* %0, align 4
	%2 = getelementptr [131072 x i32]* @dst, i64 0, i64 %i
	store i32 %1, i32* %2, align 4
	ret void
}

define void @ixd00(i64 %i) nounwind {
entry:
	%0 = getelementptr [32 x i32]* @xsrc, i64 0, i64 %i
	%1 = load i32* %0, align 4
	%2 = getelementptr [32 x i32]* @xdst, i64 0, i64 %i
	store i32 %1, i32* %2, align 4
	ret void
}

define void @ind01(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32]* @dst, i64 0, i64 %i
	store i32* %0, i32** @ptr, align 8
	ret void
}

define void @ixd01(i64 %i) nounwind {
entry:
	%0 = getelementptr [32 x i32]* @xdst, i64 0, i64 %i
	store i32* %0, i32** @ptr, align 8
	ret void
}

define void @ind02(i64 %i) nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = getelementptr [131072 x i32]* @src, i64 0, i64 %i
	%2 = load i32* %1, align 4
	%3 = getelementptr i32* %0, i64 %i
	store i32 %2, i32* %3, align 4
	ret void
}

define void @ixd02(i64 %i) nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = getelementptr [32 x i32]* @xsrc, i64 0, i64 %i
	%2 = load i32* %1, align 4
	%3 = getelementptr i32* %0, i64 %i
	store i32 %2, i32* %3, align 4
	ret void
}

define void @ind03(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32]* @dsrc, i64 0, i64 %i
	%1 = load i32* %0, align 4
	%2 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %i
	store i32 %1, i32* %2, align 4
	ret void
}

define void @ind04(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %i
	store i32* %0, i32** @dptr, align 8
	ret void
}

define void @ind05(i64 %i) nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = getelementptr [131072 x i32]* @dsrc, i64 0, i64 %i
	%2 = load i32* %1, align 4
	%3 = getelementptr i32* %0, i64 %i
	store i32 %2, i32* %3, align 4
	ret void
}

define void @ind06(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32]* @lsrc, i64 0, i64 %i
	%1 = load i32* %0, align 4
	%2 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %i
	store i32 %1, i32* %2, align 4
	ret void
}

define void @ind07(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %i
	store i32* %0, i32** @lptr, align 8
	ret void
}

define void @ind08(i64 %i) nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = getelementptr [131072 x i32]* @lsrc, i64 0, i64 %i
	%2 = load i32* %1, align 4
	%3 = getelementptr i32* %0, i64 %i
	store i32 %2, i32* %3, align 4
	ret void
}

define void @off00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @src, i64 0, i64 %0
	%2 = load i32* %1, align 4
	%3 = getelementptr [131072 x i32]* @dst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
}

define void @oxf00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [32 x i32]* @xsrc, i64 0, i64 %0
	%2 = load i32* %1, align 4
	%3 = getelementptr [32 x i32]* @xdst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
}

define void @off01(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 16
	%0 = getelementptr [131072 x i32]* @dst, i64 0, i64 %.sum
	store i32* %0, i32** @ptr, align 8
	ret void
}

define void @oxf01(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 16
	%0 = getelementptr [32 x i32]* @xdst, i64 0, i64 %.sum
	store i32* %0, i32** @ptr, align 8
	ret void
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
}

define void @off03(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @dsrc, i64 0, i64 %0
	%2 = load i32* %1, align 4
	%3 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
}

define void @off04(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 16
	%0 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %.sum
	store i32* %0, i32** @dptr, align 8
	ret void
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
}

define void @off06(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @lsrc, i64 0, i64 %0
	%2 = load i32* %1, align 4
	%3 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
}

define void @off07(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 16
	%0 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %.sum
	store i32* %0, i32** @lptr, align 8
	ret void
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
}

define void @moo00(i64 %i) nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @src, i32 0, i64 65536), align 4
	store i32 %0, i32* getelementptr ([131072 x i32]* @dst, i32 0, i64 65536), align 4
	ret void
}

define void @moo01(i64 %i) nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @dst, i32 0, i64 65536), i32** @ptr, align 8
	ret void
}

define void @moo02(i64 %i) nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @src, i32 0, i64 65536), align 4
	%2 = getelementptr i32* %0, i64 65536
	store i32 %1, i32* %2, align 4
	ret void
}

define void @moo03(i64 %i) nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 65536), align 32
	store i32 %0, i32* getelementptr ([131072 x i32]* @ddst, i32 0, i64 65536), align 32
	ret void
}

define void @moo04(i64 %i) nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @ddst, i32 0, i64 65536), i32** @dptr, align 8
	ret void
}

define void @moo05(i64 %i) nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 65536), align 32
	%2 = getelementptr i32* %0, i64 65536
	store i32 %1, i32* %2, align 4
	ret void
}

define void @moo06(i64 %i) nounwind {
entry:
	%0 = load i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 65536), align 4
	store i32 %0, i32* getelementptr ([131072 x i32]* @ldst, i32 0, i64 65536), align 4
	ret void
}

define void @moo07(i64 %i) nounwind {
entry:
	store i32* getelementptr ([131072 x i32]* @ldst, i32 0, i64 65536), i32** @lptr, align 8
	ret void
}

define void @moo08(i64 %i) nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = load i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 65536), align 4
	%2 = getelementptr i32* %0, i64 65536
	store i32 %1, i32* %2, align 4
	ret void
}

define void @big00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @src, i64 0, i64 %0
	%2 = load i32* %1, align 4
	%3 = getelementptr [131072 x i32]* @dst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
}

define void @big01(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 65536
	%0 = getelementptr [131072 x i32]* @dst, i64 0, i64 %.sum
	store i32* %0, i32** @ptr, align 8
	ret void
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
}

define void @big03(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @dsrc, i64 0, i64 %0
	%2 = load i32* %1, align 4
	%3 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
}

define void @big04(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 65536
	%0 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %.sum
	store i32* %0, i32** @dptr, align 8
	ret void
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
}

define void @big06(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @lsrc, i64 0, i64 %0
	%2 = load i32* %1, align 4
	%3 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
}

define void @big07(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 65536
	%0 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %.sum
	store i32* %0, i32** @lptr, align 8
	ret void
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
}

define i8* @bar00() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @src to i8*)
}

define i8* @bxr00() nounwind {
entry:
	ret i8* bitcast ([32 x i32]* @xsrc to i8*)
}

define i8* @bar01() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @dst to i8*)
}

define i8* @bxr01() nounwind {
entry:
	ret i8* bitcast ([32 x i32]* @xdst to i8*)
}

define i8* @bar02() nounwind {
entry:
	ret i8* bitcast (i32** @ptr to i8*)
}

define i8* @bar03() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @dsrc to i8*)
}

define i8* @bar04() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @ddst to i8*)
}

define i8* @bar05() nounwind {
entry:
	ret i8* bitcast (i32** @dptr to i8*)
}

define i8* @bar06() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @lsrc to i8*)
}

define i8* @bar07() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @ldst to i8*)
}

define i8* @bar08() nounwind {
entry:
	ret i8* bitcast (i32** @lptr to i8*)
}

define i8* @har00() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @src to i8*)
}

define i8* @hxr00() nounwind {
entry:
	ret i8* bitcast ([32 x i32]* @xsrc to i8*)
}

define i8* @har01() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @dst to i8*)
}

define i8* @hxr01() nounwind {
entry:
	ret i8* bitcast ([32 x i32]* @xdst to i8*)
}

define i8* @har02() nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = bitcast i32* %0 to i8*
	ret i8* %1
}

define i8* @har03() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @dsrc to i8*)
}

define i8* @har04() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @ddst to i8*)
}

define i8* @har05() nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = bitcast i32* %0 to i8*
	ret i8* %1
}

define i8* @har06() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @lsrc to i8*)
}

define i8* @har07() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @ldst to i8*)
}

define i8* @har08() nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = bitcast i32* %0 to i8*
	ret i8* %1
}

define i8* @bat00() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @src, i32 0, i64 16) to i8*)
}

define i8* @bxt00() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([32 x i32]* @xsrc, i32 0, i64 16) to i8*)
}

define i8* @bat01() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @dst, i32 0, i64 16) to i8*)
}

define i8* @bxt01() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([32 x i32]* @xdst, i32 0, i64 16) to i8*)
}

define i8* @bat02() nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = getelementptr i32* %0, i64 16
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @bat03() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 16) to i8*)
}

define i8* @bat04() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @ddst, i32 0, i64 16) to i8*)
}

define i8* @bat05() nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = getelementptr i32* %0, i64 16
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @bat06() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 16) to i8*)
}

define i8* @bat07() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @ldst, i32 0, i64 16) to i8*)
}

define i8* @bat08() nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = getelementptr i32* %0, i64 16
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @bam00() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @src, i32 0, i64 65536) to i8*)
}

define i8* @bam01() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @dst, i32 0, i64 65536) to i8*)
}

define i8* @bxm01() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([32 x i32]* @xdst, i32 0, i64 65536) to i8*)
}

define i8* @bam02() nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = getelementptr i32* %0, i64 65536
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @bam03() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @dsrc, i32 0, i64 65536) to i8*)
}

define i8* @bam04() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @ddst, i32 0, i64 65536) to i8*)
}

define i8* @bam05() nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = getelementptr i32* %0, i64 65536
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @bam06() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @lsrc, i32 0, i64 65536) to i8*)
}

define i8* @bam07() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32]* @ldst, i32 0, i64 65536) to i8*)
}

define i8* @bam08() nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = getelementptr i32* %0, i64 65536
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cat00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @src, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cxt00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [32 x i32]* @xsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cat01(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @dst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cxt01(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [32 x i32]* @xdst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cat02(i64 %i) nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
}

define i8* @cat03(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @dsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cat04(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cat05(i64 %i) nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
}

define i8* @cat06(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @lsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cat07(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cat08(i64 %i) nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
}

define i8* @cam00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @src, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cxm00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [32 x i32]* @xsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cam01(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @dst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cxm01(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [32 x i32]* @xdst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cam02(i64 %i) nounwind {
entry:
	%0 = load i32** @ptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
}

define i8* @cam03(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @dsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cam04(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @ddst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cam05(i64 %i) nounwind {
entry:
	%0 = load i32** @dptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
}

define i8* @cam06(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @lsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cam07(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32]* @ldst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
}

define i8* @cam08(i64 %i) nounwind {
entry:
	%0 = load i32** @lptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
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
}

declare void @y()

define void ()* @address() nounwind {
entry:
	ret void ()* @callee
}

declare void @callee()

define void ()* @laddress() nounwind {
entry:
	ret void ()* @lcallee
}

define void ()* @daddress() nounwind {
entry:
	ret void ()* @dcallee
}

define void @caller() nounwind {
entry:
	tail call void @callee() nounwind
	tail call void @callee() nounwind
	ret void
}

define void @dcaller() nounwind {
entry:
	tail call void @dcallee() nounwind
	tail call void @dcallee() nounwind
	ret void
}

define void @lcaller() nounwind {
entry:
	tail call void @lcallee() nounwind
	tail call void @lcallee() nounwind
	ret void
}

define void @tailcaller() nounwind {
entry:
	tail call void @callee() nounwind
	ret void
}

define void @dtailcaller() nounwind {
entry:
	tail call void @dcallee() nounwind
	ret void
}

define void @ltailcaller() nounwind {
entry:
	tail call void @lcallee() nounwind
	ret void
}

define void @icaller() nounwind {
entry:
	%0 = load void ()** @ifunc, align 8
	tail call void %0() nounwind
	%1 = load void ()** @ifunc, align 8
	tail call void %1() nounwind
	ret void
}

define void @dicaller() nounwind {
entry:
	%0 = load void ()** @difunc, align 8
	tail call void %0() nounwind
	%1 = load void ()** @difunc, align 8
	tail call void %1() nounwind
	ret void
}

define void @licaller() nounwind {
entry:
	%0 = load void ()** @lifunc, align 8
	tail call void %0() nounwind
	%1 = load void ()** @lifunc, align 8
	tail call void %1() nounwind
	ret void
}

define void @itailcaller() nounwind {
entry:
	%0 = load void ()** @ifunc, align 8
	tail call void %0() nounwind
	%1 = load void ()** @ifunc, align 8
	tail call void %1() nounwind
	ret void
}

define void @ditailcaller() nounwind {
entry:
	%0 = load void ()** @difunc, align 8
	tail call void %0() nounwind
	ret void
}

define void @litailcaller() nounwind {
entry:
	%0 = load void ()** @lifunc, align 8
	tail call void %0() nounwind
	ret void
}
