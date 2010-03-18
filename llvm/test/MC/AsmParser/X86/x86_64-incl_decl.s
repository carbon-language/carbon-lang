// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck --check-prefix=CHECK-X86_32 %s
// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck --check-prefix=CHECK-X86_64 %s

# CHECK-X86_32:	incb	%al # encoding: [0xfe,0xc0]
# CHECK-X86_64:	incb	%al # encoding: [0xfe,0xc0]
	incb %al

# CHECK-X86_32:	incw	%ax # encoding: [0x66,0x40]
# CHECK-X86_64:	incw	%ax # encoding: [0x66,0xff,0xc0]
	incw %ax

# CHECK-X86_32:	incl	%eax # encoding: [0x40]
# CHECK-X86_64:	incl	%eax # encoding: [0xff,0xc0]
	incl %eax

# CHECK-X86_32:	decb	%al # encoding: [0xfe,0xc8]
# CHECK-X86_64:	decb	%al # encoding: [0xfe,0xc8]
	decb %al

# CHECK-X86_32:	decw	%ax # encoding: [0x66,0x48]
# CHECK-X86_64:	decw	%ax # encoding: [0x66,0xff,0xc8]
	decw %ax

# CHECK-X86_32:	decl	%eax # encoding: [0x48]
# CHECK-X86_64:	decl	%eax # encoding: [0xff,0xc8]
	decl %eax
