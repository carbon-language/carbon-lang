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

// CHECK: vaddps  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x20,0x58,0xfa]
vaddps  %xmm10, %xmm11, %xmm15

// CHECK: vsubps  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x20,0x5c,0xfa]
vsubps  %xmm10, %xmm11, %xmm15

// CHECK: vmulps  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x20,0x59,0xfa]
vmulps  %xmm10, %xmm11, %xmm15

// CHECK: vdivps  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x20,0x5e,0xfa]
vdivps  %xmm10, %xmm11, %xmm15

// CHECK: vaddpd  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x21,0x58,0xfa]
vaddpd  %xmm10, %xmm11, %xmm15

// CHECK: vsubpd  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x21,0x5c,0xfa]
vsubpd  %xmm10, %xmm11, %xmm15

// CHECK: vmulpd  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x21,0x59,0xfa]
vmulpd  %xmm10, %xmm11, %xmm15

// CHECK: vdivpd  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x21,0x5e,0xfa]
vdivpd  %xmm10, %xmm11, %xmm15

// CHECK: vaddps  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x28,0x58,0x5c,0xd9,0xfc]
vaddps  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vsubps  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x28,0x5c,0x5c,0xd9,0xfc]
vsubps  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vmulps  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x28,0x59,0x5c,0xd9,0xfc]
vmulps  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vdivps  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x28,0x5e,0x5c,0xd9,0xfc]
vdivps  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vaddpd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x29,0x58,0x5c,0xd9,0xfc]
vaddpd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vsubpd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x29,0x5c,0x5c,0xd9,0xfc]
vsubpd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vmulpd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x29,0x59,0x5c,0xd9,0xfc]
vmulpd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vdivpd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x29,0x5e,0x5c,0xd9,0xfc]
vdivpd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vmaxss  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x0a,0x5f,0xe2]
          vmaxss  %xmm10, %xmm14, %xmm12

// CHECK: vmaxsd  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x0b,0x5f,0xe2]
          vmaxsd  %xmm10, %xmm14, %xmm12

// CHECK: vminss  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x0a,0x5d,0xe2]
          vminss  %xmm10, %xmm14, %xmm12

// CHECK: vminsd  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x0b,0x5d,0xe2]
          vminsd  %xmm10, %xmm14, %xmm12

// CHECK: vmaxss  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x1a,0x5f,0x54,0xcb,0xfc]
          vmaxss  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vmaxsd  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x1b,0x5f,0x54,0xcb,0xfc]
          vmaxsd  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vminss  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x1a,0x5d,0x54,0xcb,0xfc]
          vminss  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vminsd  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x1b,0x5d,0x54,0xcb,0xfc]
          vminsd  -4(%rbx,%rcx,8), %xmm12, %xmm10

