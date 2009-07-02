// FIXME: Actually test that we get the expected results.
        
// RUN: llvm-mc %s > %t

# Immediates
        push $1
        push $(1+2)
        push $a
        push $1 + 2
        
# Disambiguation
        push 4+4
        push (4+4)
        push (4+4)(%eax)
        push 8(%eax)
        push (%eax)
        push (4+4)(,%eax)
        
# Indirect Memory Operands
        push 1(%eax)
        push 1(%eax,%ebx)
        push 1(%eax,%ebx,)
        push 1(%eax,%ebx,4)
        push 1(,%ebx)
        push 1(,%ebx,)
        push 1(,%ebx,4)
        push 1(,%ebx,(2+2))

# '*'
        call a
        call *a
        call *%eax
        call 4(%eax) # FIXME: Warn or reject.
        call *4(%eax)

        
