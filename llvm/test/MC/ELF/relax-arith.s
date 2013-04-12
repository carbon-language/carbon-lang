// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -sd | FileCheck  %s

// Test that we correctly relax these instructions into versions that use
// 16 or 32 bit immediate values.

bar:
// CHECK:      Name: imul
// CHECK:      SectionData (
// CHECK-NEXT:   0000: 6669DB00 0066691C 25000000 00000069
// CHECK-NEXT:   0010: DB000000 00691C25 00000000 00000000
// CHECK-NEXT:   0020: 4869DB00 00000048 691C2500 00000000
// CHECK-NEXT:   0030: 000000
// CHECK-NEXT: )
        .section imul
        imul $foo, %bx,  %bx
        imul $foo, bar,  %bx
        imul $foo, %ebx, %ebx
        imul $foo, bar,  %ebx
        imul $foo, %rbx, %rbx
        imul $foo, bar,  %rbx


// CHECK:      Name: and
// CHECK:      SectionData (
// CHECK-NEXT:   0000: 6681E300 00668124 25000000 00000081
// CHECK-NEXT:   0010: E3000000 00812425 00000000 00000000
// CHECK-NEXT:   0020: 4881E300 00000048 81242500 00000000
// CHECK-NEXT:   0030: 000000
// CHECK-NEXT: )
        .section and
        and  $foo, %bx
        andw $foo, bar
        and  $foo, %ebx
        andl $foo, bar
        and  $foo, %rbx
        andq $foo, bar

// CHECK:      Name: or
// CHECK:      SectionData (
// CHECK-NEXT:   0000: 6681CB00 0066810C 25000000 00000081
// CHECK-NEXT:   0010: CB000000 00810C25 00000000 00000000
// CHECK-NEXT:   0020: 4881CB00 00000048 810C2500 00000000
// CHECK-NEXT:   0030: 000000
// CHECK-NEXT: )
        .section or
        or  $foo, %bx
        orw $foo, bar
        or  $foo, %ebx
        orl $foo, bar
        or  $foo, %rbx
        orq $foo, bar

// CHECK:      Name: xor
// CHECK:      SectionData (
// CHECK-NEXT:   0000: 6681F300 00668134 25000000 00000081
// CHECK-NEXT:   0010: F3000000 00813425 00000000 00000000
// CHECK-NEXT:   0020: 4881F300 00000048 81342500 00000000
// CHECK-NEXT:   0030: 000000
// CHECK-NEXT: )
        .section xor
        xor  $foo, %bx
        xorw $foo, bar
        xor  $foo, %ebx
        xorl $foo, bar
        xor  $foo, %rbx
        xorq $foo, bar

// CHECK:      Name: add
// CHECK:      SectionData (
// CHECK-NEXT:   0000: 6681C300 00668104 25000000 00000081
// CHECK-NEXT:   0010: C3000000 00810425 00000000 00000000
// CHECK-NEXT:   0020: 4881C300 00000048 81042500 00000000
// CHECK-NEXT:   0030: 000000
// CHECK-NEXT: )
        .section add
        add  $foo, %bx
        addw $foo, bar
        add  $foo, %ebx
        addl $foo, bar
        add  $foo, %rbx
        addq $foo, bar

// CHECK:      Name: sub
// CHECK:      SectionData (
// CHECK-NEXT:   000: 6681EB00 0066812C 25000000 00000081
// CHECK-NEXT:   010: EB000000 00812C25 00000000 00000000
// CHECK-NEXT:   020: 4881EB00 00000048 812C2500 00000000
// CHECK-NEXT:   030: 000000
// CHECK-NEXT: )
        .section sub
        sub  $foo, %bx
        subw $foo, bar
        sub  $foo, %ebx
        subl $foo, bar
        sub  $foo, %rbx
        subq $foo, bar

// CHECK:      Name: cmp
// CHECK:      SectionData (
// CHECK-NEXT:   0000: 6681FB00 0066813C 25000000 00000081
// CHECK-NEXT:   0010: FB000000 00813C25 00000000 00000000
// CHECK-NEXT:   0020: 4881FB00 00000048 813C2500 00000000
// CHECK-NEXT:   0030: 000000
// CHECK-NEXT: )
        .section cmp
        cmp  $foo, %bx
        cmpw $foo, bar
        cmp  $foo, %ebx
        cmpl $foo, bar
        cmp  $foo, %rbx
        cmpq $foo, bar
