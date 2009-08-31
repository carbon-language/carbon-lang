// FIXME: Actually test that we get the expected results.
        
// RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# Immediates
# CHECK: addl $1, %eax
        addl $1, %eax
# CHECK: addl $(1 + 2), %eax
        addl $(1+2), %eax
# CHECK: addl $a, %eax
        addl $a, %eax
# CHECK: addl $(1 + 2), %eax
        addl $1 + 2, %eax
        
# Disambiguation

        # FIXME: Add back when we can match this.
        #addl $1, 4+4
        # FIXME: Add back when we can match this.
        #addl $1, (4+4)
# CHECK: addl $1, (4 + 4)(%eax)
        addl $1, 4+4(%eax)
# CHECK: addl $1, (4 + 4)(%eax)
        addl $1, (4+4)(%eax)
# CHECK: addl $1, 8(%eax)
        addl $1, 8(%eax)
# CHECK: addl $1, 0(%eax)
        addl $1, (%eax)
# CHECK: addl $1, (4 + 4)(,%eax)
        addl $1, (4+4)(,%eax)
        
# Indirect Memory Operands
# CHECK: addl $1, 1(%eax)
        addl $1, 1(%eax)
# CHECK: addl $1, 1(%eax,%ebx)
        addl $1, 1(%eax,%ebx)
# CHECK: addl $1, 1(%eax,%ebx)
        addl $1, 1(%eax,%ebx,)
# CHECK: addl $1, 1(%eax,%ebx,4)
        addl $1, 1(%eax,%ebx,4)
# CHECK: addl $1, 1(,%ebx)
        addl $1, 1(,%ebx)
# CHECK: addl $1, 1(,%ebx)
        addl $1, 1(,%ebx,)
# CHECK: addl $1, 1(,%ebx,4)
        addl $1, 1(,%ebx,4)
# CHECK: addl $1, 1(,%ebx,4)
        addl $1, 1(,%ebx,(2+2))

# '*'
# CHECK: call a
        call a
# CHECK: call *%eax
        call *%eax
# CHECK: call *4(%eax)
        call *4(%eax)

        
