// Assembler generated object test.
// This tests .eh_frame descriptors minimally.

// What we really need is a prettyprinter output check not unlike what
// gnu's readobj generates instead of checking the bits for .eh_frame.

// RUN: llvm-mc -filetype=obj -mcpu=mips32r2 -triple mipsel-unknown-linux -arch=mipsel %s -o - \
// RUN: | llvm-objdump -s - | FileCheck -check-prefix=CHECK-LEO32 %s

// RUN: llvm-mc -filetype=obj -mcpu=mips32r2 -triple mips-unknown-linux -arch=mips %s -o - \
// RUN: | llvm-objdump -s - | FileCheck -check-prefix=CHECK-BEO32 %s

// RUN: llvm-mc -filetype=obj -mcpu=mips64r2 -mattr=n64 -arch=mips64el %s -o - \
// RUN: | llvm-objdump -s - | FileCheck -check-prefix=CHECK-LE64 %s

// RUN: llvm-mc -filetype=obj -mcpu=mips64r2 -mattr=n64 -arch=mips64 %s -o - \
// RUN: | llvm-objdump -s - | FileCheck -check-prefix=CHECK-BE64 %s

// O32 little endian
// CHECK-LEO32: Contents of section .eh_frame:
// CHECK-LEO32-NEXT: 0000 10000000 00000000 017a5200 017c1f01  .........zR..|..
// CHECK-LEO32-NEXT: 0010 000c1d00 10000000 18000000 00000000  ................
// CHECK-LEO32-NEXT: 0020 00000000 00000000                    ........

// O32 big endian
// CHECK-BEO32: Contents of section .eh_frame:
// CHECK-BEO32-NEXT 0000 00000010 00000000 017a5200 017c1f01  .........zR..|..
// CHECK-BEO32-NEXT 0010 000c1d00 00000010 00000018 00000000  ................
// CHECK-BEO32-NEXT 0020 00000000 00000000                    ........

// N64 little endian
// CHECK-LE64: Contents of section .eh_frame:
// CHECK-LE64-NEXT: 0000 10000000 00000000 017a5200 01781f01  .........zR..x..
// CHECK-LE64-NEXT: 0010 000c1d00 18000000 18000000 00000000  ................
// CHECK-LE64-NEXT: 0020 00000000 00000000 00000000 00000000  ................

// N64 big endian
// CHECK-BE64: Contents of section .eh_frame:
// CHECK-BE64-NEXT: 0000 00000010 00000000 017a5200 01781f01  .........zR..x..
// CHECK-BE64-NEXT: 0010 000c1d00 00000018 00000018 00000000  ................
// CHECK-BE64-NEXT: 0020 00000000 00000000 00000000 00000000  ................

func:
	.cfi_startproc
	.cfi_endproc

