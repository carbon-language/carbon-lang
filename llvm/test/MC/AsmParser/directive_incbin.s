# RUN: llvm-mc -triple i386-unknown-unknown %s -I %p | FileCheck %s

.data
.incbin "incbin\137abcd"  # "\137" is underscore "_"

# CHECK: .ascii	 "abcd\n"
