Fix jump table support.  currently it uses 64bit absolute address. 
gcc uses gprel32.  This way I won't keep fighting Evan as he keeps
breaking 64bit entries in jump tables...

#include <string.h>
#include <setjmp.h>

int main(int x, char** y)
{
char* foo;
switch(x) {
case 1:
foo = "1";
break;
case 2:
foo = "2";
break;
case 3:
foo = "3";
break;
case 4:
foo = "4";
break;
case 5:
foo = "5";
break;
case 6:
foo = "6";
break;
case 7:
foo = "7";
break;
case 8:
foo = "8";
break;
};
print(foo);
return 0;

}


        .set noreorder
        .set volatile
        .set noat
        .set nomacro
        .section        .rodata.str1.1,"aMS",@progbits,1
$LC6:
        .ascii "7\0"
$LC7:
        .ascii "8\0"
$LC0:
        .ascii "1\0"
$LC1:
        .ascii "2\0"
$LC2:
        .ascii "3\0"
$LC3:
        .ascii "4\0"
$LC4:
        .ascii "5\0"
$LC5:
        .ascii "6\0"
        .text
        .align 2
        .align 4
        .globl main
        .ent main
main:
        .frame $30,16,$26,0
        .mask 0x4000000,-16
        ldah $29,0($27)         !gpdisp!1
        lda $29,0($29)          !gpdisp!1
$main..ng:
        zapnot $16,15,$16
        lda $30,-16($30)
        cmpule $16,8,$1
        stq $26,0($30)
        .prologue 1
        beq $1,$L2
        ldah $6,$L11($29)               !gprelhigh
        lda $5,$L11($6)         !gprellow
        s4addq $16,$5,$0
        ldl $2,0($0)
        addq $29,$2,$3
        jmp $31,($3),$L2
        .section        .rodata
        .align 2
        .align 2
$L11:
        .gprel32 $L2
        .gprel32 $L3
        .gprel32 $L4
        .gprel32 $L5
        .gprel32 $L6
        .gprel32 $L7
        .gprel32 $L8
        .gprel32 $L9
        .gprel32 $L10
        .text
$L9:
        ldah $20,$LC6($29)              !gprelhigh
        lda $4,$LC6($20)                !gprellow
        .align 4
$L2:
        mov $4,$16
        ldq $27,print($29)              !literal!2
        jsr $26,($27),print             !lituse_jsr!2
        ldah $29,0($26)         !gpdisp!3
        mov $31,$0
        bis $31,$31,$31
        lda $29,0($29)          !gpdisp!3
        ldq $26,0($30)
        lda $30,16($30)
        ret $31,($26),1
$L10:
        ldah $21,$LC7($29)              !gprelhigh
        lda $4,$LC7($21)                !gprellow
        br $31,$L2
$L3:
        ldah $7,$LC0($29)               !gprelhigh
        lda $4,$LC0($7)         !gprellow
        br $31,$L2
$L4:
        ldah $8,$LC1($29)               !gprelhigh
        lda $4,$LC1($8)         !gprellow
        br $31,$L2
$L5:
        ldah $16,$LC2($29)              !gprelhigh
        lda $4,$LC2($16)                !gprellow
        br $31,$L2
$L6:
        ldah $17,$LC3($29)              !gprelhigh
        lda $4,$LC3($17)                !gprellow
        br $31,$L2
$L7:
        ldah $18,$LC4($29)              !gprelhigh
        lda $4,$LC4($18)                !gprellow
        br $31,$L2
$L8:
        ldah $19,$LC5($29)              !gprelhigh
        lda $4,$LC5($19)                !gprellow
        br $31,$L2
        .end main
        .section        .note.GNU-stack,"",@progbits
        .ident  "GCC: (GNU) 3.4.4 20050314 (prerelease) (Debian 3.4.3-13)"




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fix Ordered/Unordered FP stuff

