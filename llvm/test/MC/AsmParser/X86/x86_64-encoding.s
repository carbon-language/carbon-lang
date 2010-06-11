// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: crc32b 	%bl, %eax
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf0,0xc3]
        crc32b	%bl, %eax

// CHECK: crc32b 	4(%rbx), %eax
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf0,0x43,0x04]
        crc32b	4(%rbx), %eax

// CHECK: crc32w 	%bx, %eax
// CHECK:  encoding: [0x66,0xf2,0x0f,0x38,0xf1,0xc3]
        crc32w	%bx, %eax

// CHECK: crc32w 	4(%rbx), %eax
// CHECK:  encoding: [0x66,0xf2,0x0f,0x38,0xf1,0x43,0x04]
        crc32w	4(%rbx), %eax

// CHECK: crc32l 	%ebx, %eax
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf1,0xc3]
        crc32l	%ebx, %eax

// CHECK: crc32l 	4(%rbx), %eax
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf1,0x43,0x04]
        crc32l	4(%rbx), %eax

// CHECK: crc32l 	3735928559(%rbx,%rcx,8), %ecx
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf1,0x8c,0xcb,0xef,0xbe,0xad,0xde]
        	crc32l   0xdeadbeef(%rbx,%rcx,8),%ecx

// CHECK: crc32l 	69, %ecx
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf1,0x0c,0x25,0x45,0x00,0x00,0x00]
        	crc32l   0x45,%ecx

// CHECK: crc32l 	32493, %ecx
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf1,0x0c,0x25,0xed,0x7e,0x00,0x00]
        	crc32l   0x7eed,%ecx

// CHECK: crc32l 	3133065982, %ecx
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf1,0x0c,0x25,0xfe,0xca,0xbe,0xba]
        	crc32l   0xbabecafe,%ecx

// CHECK: crc32l 	%ecx, %ecx
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf1,0xc9]
        	crc32l   %ecx,%ecx

// CHECK: crc32b 	%r11b, %eax
// CHECK:  encoding: [0xf2,0x41,0x0f,0x38,0xf0,0xc3]
        crc32b	%r11b, %eax

// CHECK: crc32b 	4(%rbx), %eax
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf0,0x43,0x04]
        crc32b	4(%rbx), %eax

// CHECK: crc32b 	%dil, %rax
// CHECK:  encoding: [0xf2,0x48,0x0f,0x38,0xf0,0xc7]
        crc32b	%dil,%rax

// CHECK: crc32b 	%r11b, %rax
// CHECK:  encoding: [0xf2,0x49,0x0f,0x38,0xf0,0xc3]
        crc32b	%r11b,%rax

// CHECK: crc32b 	4(%rbx), %rax
// CHECK:  encoding: [0xf2,0x48,0x0f,0x38,0xf0,0x43,0x04]
        crc32b	4(%rbx), %rax

// CHECK: crc32q 	%rbx, %rax
// CHECK:  encoding: [0xf2,0x48,0x0f,0x38,0xf1,0xc3]
        crc32q	%rbx, %rax

// CHECK: crc32q 	4(%rbx), %rax
// CHECK:  encoding: [0xf2,0x48,0x0f,0x38,0xf1,0x43,0x04]
        crc32q	4(%rbx), %rax

// CHECK: movd %r8, %mm1
// CHECK:  encoding: [0x49,0x0f,0x6e,0xc8]
movd %r8, %mm1

// CHECK: movd %r8d, %mm1
// CHECK:  encoding: [0x41,0x0f,0x6e,0xc8]
movd %r8d, %mm1

// CHECK: movd %rdx, %mm1
// CHECK:  encoding: [0x48,0x0f,0x6e,0xca]
movd %rdx, %mm1

// CHECK: movd %edx, %mm1
// CHECK:  encoding: [0x0f,0x6e,0xca]
movd %edx, %mm1

// CHECK: movd %mm1, %r8
// CHECK:  encoding: [0x49,0x0f,0x7e,0xc8]
movd %mm1, %r8

// CHECK: movd %mm1, %r8d
// CHECK:  encoding: [0x41,0x0f,0x7e,0xc8]
movd %mm1, %r8d

// CHECK: movd %mm1, %rdx
// CHECK:  encoding: [0x48,0x0f,0x7e,0xca]
movd %mm1, %rdx

// CHECK: movd %mm1, %edx
// CHECK:  encoding: [0x0f,0x7e,0xca]
movd %mm1, %edx

// CHECK: vaddss  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x32,0x58,0xd0]
vaddss  %xmm8, %xmm9, %xmm10

// CHECK: vmulss  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x32,0x59,0xd0]
vmulss  %xmm8, %xmm9, %xmm10

// CHECK: vsubss  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x32,0x5c,0xd0]
vsubss  %xmm8, %xmm9, %xmm10

// CHECK: vdivss  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x32,0x5e,0xd0]
vdivss  %xmm8, %xmm9, %xmm10

// CHECK: vaddsd  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x33,0x58,0xd0]
vaddsd  %xmm8, %xmm9, %xmm10

// CHECK: vmulsd  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x33,0x59,0xd0]
vmulsd  %xmm8, %xmm9, %xmm10

// CHECK: vsubsd  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x33,0x5c,0xd0]
vsubsd  %xmm8, %xmm9, %xmm10

// CHECK: vdivsd  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x33,0x5e,0xd0]
vdivsd  %xmm8, %xmm9, %xmm10

// CHECK:   vaddss  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2a,0x58,0x5c,0xd9,0xfc]
vaddss  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK:   vsubss  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2a,0x5c,0x5c,0xd9,0xfc]
vsubss  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK:   vmulss  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2a,0x59,0x5c,0xd9,0xfc]
vmulss  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK:   vdivss  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2a,0x5e,0x5c,0xd9,0xfc]
vdivss  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK:   vaddsd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2b,0x58,0x5c,0xd9,0xfc]
vaddsd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK:   vsubsd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2b,0x5c,0x5c,0xd9,0xfc]
vsubsd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK:   vmulsd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2b,0x59,0x5c,0xd9,0xfc]
vmulsd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK:   vdivsd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2b,0x5e,0x5c,0xd9,0xfc]
vdivsd  -4(%rcx,%rbx,8), %xmm10, %xmm11
