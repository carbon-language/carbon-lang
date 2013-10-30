// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -sr -t | FileCheck  %s

// Test that we produce the correct relocation.

bar:
        movl	$bar, %edx        # R_X86_64_32
        movq	$bar, %rdx        # R_X86_64_32S
        movq	$bar, bar(%rip)   # R_X86_64_32S
        movl	bar, %edx         # R_X86_64_32S
        movq	bar, %rdx         # R_X86_64_32S
.long bar                         # R_X86_64_32
        leaq	foo@GOTTPOFF(%rip), %rax # R_X86_64_GOTTPOFF
        leaq	foo@TLSGD(%rip), %rax    # R_X86_64_TLSGD
        leaq	foo@TPOFF(%rax), %rax    # R_X86_64_TPOFF32
        leaq	foo@TLSLD(%rip), %rdi    # R_X86_64_TLSLD
        leaq	foo@dtpoff(%rax), %rcx   # R_X86_64_DTPOFF32
        movabs  foo@GOT, %rax		 # R_X86_64_GOT64
        movabs  foo@GOTOFF, %rax	 # R_X86_64_GOTOFF64
        pushq    $bar
        movq	foo(%rip), %rdx
        leaq    foo-bar(%r14),%r14
        addq	$bar,%rax         # R_X86_64_32S
	.quad	foo@DTPOFF
        movabsq	$baz@TPOFF, %rax

// CHECK:        Section {
// CHECK:          Name: .rela.text
// CHECK:          Relocations [
// CHECK-NEXT:       0x1 R_X86_64_32        .text
// CHECK-NEXT:       0x8 R_X86_64_32S       .text
// CHECK-NEXT:       0x13 R_X86_64_32S      .text
// CHECK-NEXT:       0x1A R_X86_64_32S      .text
// CHECK-NEXT:       0x22 R_X86_64_32S      .text
// CHECK-NEXT:       0x26 R_X86_64_32       .text
// CHECK-NEXT:       0x2D R_X86_64_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x34 R_X86_64_TLSGD    foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x3B R_X86_64_TPOFF32  foo 0x0
// CHECK-NEXT:       0x42 R_X86_64_TLSLD    foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x49 R_X86_64_DTPOFF32 foo 0x0
// CHECK-NEXT:       0x4F R_X86_64_GOT64 foo 0x0
// CHECK-NEXT:       0x59 R_X86_64_GOTOFF64 foo 0x0
// CHECK-NEXT:       0x62 R_X86_64_32S .text 0x0
// CHECK-NEXT:       0x69 R_X86_64_PC32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x70 R_X86_64_PC32 foo 0x70
// CHECK-NEXT:       0x77 R_X86_64_32S .text 0x0
// CHECK-NEXT:       0x7B R_X86_64_DTPOFF64 foo 0x0
// CHECK-NEXT:       0x85 R_X86_64_TPOFF64 baz 0x0
// CHECK-NEXT:     ]
// CHECK-NEXT:   }

// CHECK:        Symbol {
// CHECK:          Name: .text (0)
// CHECK-NEXT:     Value:
// CHECK-NEXT:     Size:
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: Section
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text (0x1)
// CHECK-NEXT:   }
