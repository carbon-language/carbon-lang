        .text
        .globl  main
        .align  16, 0x90
        .type   main,@function
main:                                   # @main
        callq   GOTTPOFF
        addl    %fs:tls1@TPOFF, %eax
        addl    %fs:tls0@TPOFF, %eax
        addl    %fs:tls2@TPOFF, %eax
        ret

        .text
        .globl GOTTPOFF
        .type  GOTTPOFF,@function
GOTTPOFF:
        movq tls2@GOTTPOFF(%rip), %rax
        movl %fs:0(%rax), %eax
        ret

        .type   tls0,@object            # @tls0
        .section        .tbss,"awT",@nobits
        .globl  tls0
        .align  4
tls0:
        .long   0                       # 0x0
        .size   tls0, 4

        .type   tls1,@object            # @tls1
        .globl  tls1
        .align  4
tls1:
        .long   0                       # 0x0
        .size   tls1, 4

        .type   tls2,@object            # @tls2
        .section        .tdata,"awT",@progbits
        .globl  tls2
        .align  4
tls2:
        .long   1                       # 0x1
        .size   tls2, 4
