// FIXME: For now this test just checks that llvm-mc -triple i386-unknown-unknown works. Once we have .macro,
// .if, and .abort we can write a better test (without resorting to miles of
// greps).
        
// RUN: llvm-mc -triple i386-unknown-unknown %s > %t

        .text
g:
h:
j:
k:      
        .data
        .byte !1 + 2
        .byte !0
        .byte ~0
        .byte -1
        .byte +1
        .byte 1 + 2
        .byte 1 & 3
        .byte 4 / 2
        .byte 4 / -2
        .byte 1 == 1
        .byte 1 == 0
        .byte 1 > 0
        .byte 1 >= 1
        .byte 1 < 2
        .byte 1 <= 1
        .byte 4 % 3
        .byte 2 * 2
        .byte 2 != 2
        .byte 2 <> 2
        .byte 1 | 2
        .byte 1 << 1
        .byte 2 >> 1
        .byte ~0 >> 1
        .byte 3 - 2
        .byte 1 ^ 3
        .byte 1 && 2
        .byte 3 && 0
        .byte 1 || 2
        .byte 0 || 0

        .set c, 10
        .byte c + 1

        d = e + 10
        .long d

        f = g - h + 5
        .long f

        i = (j + 10) - (k + 2)
        .long i
        
        l = m - n + 4
        
        .text
m:
n:
        nop
        
        
        movw	$8, (42)+66(%eax)
