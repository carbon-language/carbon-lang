; RUN: llc -march=ppc64 -mcpu=pwr7 -O0 -relocation-model=pic < %s | FileCheck -check-prefix=OPT0 %s
; RUN: llc -march=ppc64 -mcpu=pwr7 -O1 -relocation-model=pic < %s | FileCheck -check-prefix=OPT1 %s

target triple = "powerpc64-unknown-linux-gnu"
; Test correct assembly code generation for thread-local storage using
; the local dynamic model.

@a = hidden thread_local global i32 0, align 4

define signext i32 @main() nounwind {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = load i32* @a, align 4
  ret i32 %0
}

; OPT0-LABEL: main:
; OPT0:      addis [[REG:[0-9]+]], 2, a@got@tlsld@ha
; OPT0-NEXT: addi 3, [[REG]], a@got@tlsld@l
; OPT0:      bl __tls_get_addr(a@tlsld)
; OPT0-NEXT: nop
; OPT0:      addis [[REG2:[0-9]+]], 3, a@dtprel@ha
; OPT0-NEXT: addi {{[0-9]+}}, [[REG2]], a@dtprel@l

; Test peephole optimization for thread-local storage using the
; local dynamic model.

; OPT1-LABEL: main:
; OPT1:      addis [[REG:[0-9]+]], 2, a@got@tlsld@ha
; OPT1-NEXT: addi 3, [[REG]], a@got@tlsld@l
; OPT1:      bl __tls_get_addr(a@tlsld)
; OPT1-NEXT: nop
; OPT1:      addis [[REG2:[0-9]+]], 3, a@dtprel@ha
; OPT1-NEXT: lwa {{[0-9]+}}, a@dtprel@l([[REG2]])

; Test correct assembly code generation for thread-local storage using
; the general dynamic model.

@a2 = thread_local global i32 0, align 4

define signext i32 @main2() nounwind {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = load i32* @a2, align 4
  ret i32 %0
}

; OPT1-LABEL: main2
; OPT1: addis [[REG:[0-9]+]], 2, a2@got@tlsgd@ha
; OPT1-NEXT: addi 3, [[REG]], a2@got@tlsgd@l
; OPT1:      bl __tls_get_addr(a2@tlsgd)
; OPT1-NEXT: nop

