; RUN: llvm-as < %s | llc -march=x86 | grep 18446744073709551615 | count 14
; RUN: llvm-as < %s | llc -march=ppc32 | grep 4294967295 | count 28

; These static initializers are too big to hand off to assemblers
; as monolithic blobs.

@x = global i128 -1
@y = global i256 -1
@z = global i512 -1
