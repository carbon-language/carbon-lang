// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s 2> %t2 -filetype=obj \
// RUN:   >/dev/null
// RUN: FileCheck --check-prefix=CHECK-ERROR %s < %t2

   ldr w24, [x23, :tlsdesc_lo12:sym]
   ldr s22, [x21, :tlsdesc_lo12:sym]

// CHECK-ERROR: error: LP64 4 byte TLSDESC load/store relocation not supported (ILP32 eqv: TLSDESC_LD64_LO12)
// CHECK-ERROR:   ldr w24, [x23, :tlsdesc_lo12:sym]
// CHECK-ERROR:   ^
// CHECK-ERROR: error: LP64 4 byte TLSDESC load/store relocation not supported (ILP32 eqv: TLSDESC_LD64_LO12)
// CHECK-ERROR:   ldr s22, [x21, :tlsdesc_lo12:sym]
// CHECK-ERROR:   ^
