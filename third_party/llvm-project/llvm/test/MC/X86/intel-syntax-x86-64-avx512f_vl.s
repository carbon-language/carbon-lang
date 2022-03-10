// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK:  vcmppd k3, xmm27, xmm23, 171
// CHECK:  encoding: [0x62,0xb1,0xa5,0x00,0xc2,0xdf,0xab]
          vcmppd k3,xmm27,xmm23,0xab

// CHECK:  vcmppd k3 {k5}, xmm27, xmm23, 171
// CHECK:  encoding: [0x62,0xb1,0xa5,0x05,0xc2,0xdf,0xab]
          vcmppd k3{k5},xmm27,xmm23,0xab

// CHECK:  vcmppd k3, xmm27, xmm23, 123
// CHECK:  encoding: [0x62,0xb1,0xa5,0x00,0xc2,0xdf,0x7b]
          vcmppd k3,xmm27,xmm23,0x7b

// CHECK:  vcmppd k3, xmm27, xmmword ptr [rcx], 123
// CHECK:  encoding: [0x62,0xf1,0xa5,0x00,0xc2,0x19,0x7b]
          vcmppd k3,xmm27,XMMWORD PTR [rcx],0x7b

// CHECK:  vcmppd k3, xmm27, xmmword ptr [rax + 8*r14 + 291], 123
// CHECK:  encoding: [0x62,0xb1,0xa5,0x00,0xc2,0x9c,0xf0,0x23,0x01,0x00,0x00,0x7b]
          vcmppd k3,xmm27,XMMWORD PTR [rax+r14*8+0x123],0x7b

// CHECK:  vcmppd k3, xmm27, qword ptr [rcx]{1to2}, 123
// CHECK:  encoding: [0x62,0xf1,0xa5,0x10,0xc2,0x19,0x7b]
          vcmppd k3,xmm27,QWORD PTR [rcx]{1to2},0x7b

// CHECK:  vcmppd k3, xmm27, xmmword ptr [rdx + 2032], 123
// CHECK:  encoding: [0x62,0xf1,0xa5,0x00,0xc2,0x5a,0x7f,0x7b]
          vcmppd k3,xmm27,XMMWORD PTR [rdx+0x7f0],0x7b

// CHECK:  vcmppd k3, xmm27, xmmword ptr [rdx + 2048], 123
// CHECK:  encoding: [0x62,0xf1,0xa5,0x00,0xc2,0x9a,0x00,0x08,0x00,0x00,0x7b]
          vcmppd k3,xmm27,XMMWORD PTR [rdx+0x800],0x7b

// CHECK:  vcmppd k3, xmm27, xmmword ptr [rdx - 2048], 123
// CHECK:  encoding: [0x62,0xf1,0xa5,0x00,0xc2,0x5a,0x80,0x7b]
          vcmppd k3,xmm27,XMMWORD PTR [rdx-0x800],0x7b

// CHECK:  vcmppd k3, xmm27, xmmword ptr [rdx - 2064], 123
// CHECK:  encoding: [0x62,0xf1,0xa5,0x00,0xc2,0x9a,0xf0,0xf7,0xff,0xff,0x7b]
          vcmppd k3,xmm27,XMMWORD PTR [rdx-0x810],0x7b

// CHECK:  vcmppd k3, xmm27, qword ptr [rdx + 1016]{1to2}, 123
// CHECK:  encoding: [0x62,0xf1,0xa5,0x10,0xc2,0x5a,0x7f,0x7b]
          vcmppd k3,xmm27,QWORD PTR [rdx+0x3f8]{1to2},0x7b

// CHECK:  vcmppd k3, xmm27, qword ptr [rdx + 1024]{1to2}, 123
// CHECK:  encoding: [0x62,0xf1,0xa5,0x10,0xc2,0x9a,0x00,0x04,0x00,0x00,0x7b]
          vcmppd k3,xmm27,QWORD PTR [rdx+0x400]{1to2},0x7b

// CHECK:  vcmppd k3, xmm27, qword ptr [rdx - 1024]{1to2}, 123
// CHECK:  encoding: [0x62,0xf1,0xa5,0x10,0xc2,0x5a,0x80,0x7b]
          vcmppd k3,xmm27,QWORD PTR [rdx-0x400]{1to2},0x7b

// CHECK:  vcmppd k3, xmm27, qword ptr [rdx - 1032]{1to2}, 123
// CHECK:  encoding: [0x62,0xf1,0xa5,0x10,0xc2,0x9a,0xf8,0xfb,0xff,0xff,0x7b]
          vcmppd k3,xmm27,QWORD PTR [rdx-0x408]{1to2},0x7b

// CHECK:  vcmppd k4, ymm17, ymm27, 171
// CHECK:  encoding: [0x62,0x91,0xf5,0x20,0xc2,0xe3,0xab]
          vcmppd k4,ymm17,ymm27,0xab

// CHECK:  vcmppd k4 {k7}, ymm17, ymm27, 171
// CHECK:  encoding: [0x62,0x91,0xf5,0x27,0xc2,0xe3,0xab]
          vcmppd k4{k7},ymm17,ymm27,0xab

// CHECK:  vcmppd k4, ymm17, ymm27, 123
// CHECK:  encoding: [0x62,0x91,0xf5,0x20,0xc2,0xe3,0x7b]
          vcmppd k4,ymm17,ymm27,0x7b

// CHECK:  vcmppd k4, ymm17, ymmword ptr [rcx], 123
// CHECK:  encoding: [0x62,0xf1,0xf5,0x20,0xc2,0x21,0x7b]
          vcmppd k4,ymm17,YMMWORD PTR [rcx],0x7b

// CHECK:  vcmppd k4, ymm17, ymmword ptr [rax + 8*r14 + 291], 123
// CHECK:  encoding: [0x62,0xb1,0xf5,0x20,0xc2,0xa4,0xf0,0x23,0x01,0x00,0x00,0x7b]
          vcmppd k4,ymm17,YMMWORD PTR [rax+r14*8+0x123],0x7b

// CHECK:  vcmppd k4, ymm17, qword ptr [rcx]{1to4}, 123
// CHECK:  encoding: [0x62,0xf1,0xf5,0x30,0xc2,0x21,0x7b]
          vcmppd k4,ymm17,QWORD PTR [rcx]{1to4},0x7b

// CHECK:  vcmppd k4, ymm17, ymmword ptr [rdx + 4064], 123
// CHECK:  encoding: [0x62,0xf1,0xf5,0x20,0xc2,0x62,0x7f,0x7b]
          vcmppd k4,ymm17,YMMWORD PTR [rdx+0xfe0],0x7b

// CHECK:  vcmppd k4, ymm17, ymmword ptr [rdx + 4096], 123
// CHECK:  encoding: [0x62,0xf1,0xf5,0x20,0xc2,0xa2,0x00,0x10,0x00,0x00,0x7b]
          vcmppd k4,ymm17,YMMWORD PTR [rdx+0x1000],0x7b

// CHECK:  vcmppd k4, ymm17, ymmword ptr [rdx - 4096], 123
// CHECK:  encoding: [0x62,0xf1,0xf5,0x20,0xc2,0x62,0x80,0x7b]
          vcmppd k4,ymm17,YMMWORD PTR [rdx-0x1000],0x7b

// CHECK:  vcmppd k4, ymm17, ymmword ptr [rdx - 4128], 123
// CHECK:  encoding: [0x62,0xf1,0xf5,0x20,0xc2,0xa2,0xe0,0xef,0xff,0xff,0x7b]
          vcmppd k4,ymm17,YMMWORD PTR [rdx-0x1020],0x7b

// CHECK:  vcmppd k4, ymm17, qword ptr [rdx + 1016]{1to4}, 123
// CHECK:  encoding: [0x62,0xf1,0xf5,0x30,0xc2,0x62,0x7f,0x7b]
          vcmppd k4,ymm17,QWORD PTR [rdx+0x3f8]{1to4},0x7b

// CHECK:  vcmppd k4, ymm17, qword ptr [rdx + 1024]{1to4}, 123
// CHECK:  encoding: [0x62,0xf1,0xf5,0x30,0xc2,0xa2,0x00,0x04,0x00,0x00,0x7b]
          vcmppd k4,ymm17,QWORD PTR [rdx+0x400]{1to4},0x7b

// CHECK:  vcmppd k4, ymm17, qword ptr [rdx - 1024]{1to4}, 123
// CHECK:  encoding: [0x62,0xf1,0xf5,0x30,0xc2,0x62,0x80,0x7b]
          vcmppd k4,ymm17,QWORD PTR [rdx-0x400]{1to4},0x7b

// CHECK:  vcmppd k4, ymm17, qword ptr [rdx - 1032]{1to4}, 123
// CHECK:  encoding: [0x62,0xf1,0xf5,0x30,0xc2,0xa2,0xf8,0xfb,0xff,0xff,0x7b]
          vcmppd k4,ymm17,QWORD PTR [rdx-0x408]{1to4},0x7b

// CHECK:  vcmpps k4, xmm29, xmm28, 171
// CHECK:  encoding: [0x62,0x91,0x14,0x00,0xc2,0xe4,0xab]
          vcmpps k4,xmm29,xmm28,0xab

// CHECK:  vcmpps k4 {k2}, xmm29, xmm28, 171
// CHECK:  encoding: [0x62,0x91,0x14,0x02,0xc2,0xe4,0xab]
          vcmpps k4{k2},xmm29,xmm28,0xab

// CHECK:  vcmpps k4, xmm29, xmm28, 123
// CHECK:  encoding: [0x62,0x91,0x14,0x00,0xc2,0xe4,0x7b]
          vcmpps k4,xmm29,xmm28,0x7b

// CHECK:  vcmpps k4, xmm29, xmmword ptr [rcx], 123
// CHECK:  encoding: [0x62,0xf1,0x14,0x00,0xc2,0x21,0x7b]
          vcmpps k4,xmm29,XMMWORD PTR [rcx],0x7b

// CHECK:  vcmpps k4, xmm29, xmmword ptr [rax + 8*r14 + 291], 123
// CHECK:  encoding: [0x62,0xb1,0x14,0x00,0xc2,0xa4,0xf0,0x23,0x01,0x00,0x00,0x7b]
          vcmpps k4,xmm29,XMMWORD PTR [rax+r14*8+0x123],0x7b

// CHECK:  vcmpps k4, xmm29, dword ptr [rcx]{1to4}, 123
// CHECK:  encoding: [0x62,0xf1,0x14,0x10,0xc2,0x21,0x7b]
          vcmpps k4,xmm29,DWORD PTR [rcx]{1to4},0x7b

// CHECK:  vcmpps k4, xmm29, xmmword ptr [rdx + 2032], 123
// CHECK:  encoding: [0x62,0xf1,0x14,0x00,0xc2,0x62,0x7f,0x7b]
          vcmpps k4,xmm29,XMMWORD PTR [rdx+0x7f0],0x7b

// CHECK:  vcmpps k4, xmm29, xmmword ptr [rdx + 2048], 123
// CHECK:  encoding: [0x62,0xf1,0x14,0x00,0xc2,0xa2,0x00,0x08,0x00,0x00,0x7b]
          vcmpps k4,xmm29,XMMWORD PTR [rdx+0x800],0x7b

// CHECK:  vcmpps k4, xmm29, xmmword ptr [rdx - 2048], 123
// CHECK:  encoding: [0x62,0xf1,0x14,0x00,0xc2,0x62,0x80,0x7b]
          vcmpps k4,xmm29,XMMWORD PTR [rdx-0x800],0x7b

// CHECK:  vcmpps k4, xmm29, xmmword ptr [rdx - 2064], 123
// CHECK:  encoding: [0x62,0xf1,0x14,0x00,0xc2,0xa2,0xf0,0xf7,0xff,0xff,0x7b]
          vcmpps k4,xmm29,XMMWORD PTR [rdx-0x810],0x7b

// CHECK:  vcmpps k4, xmm29, dword ptr [rdx + 508]{1to4}, 123
// CHECK:  encoding: [0x62,0xf1,0x14,0x10,0xc2,0x62,0x7f,0x7b]
          vcmpps k4,xmm29,DWORD PTR [rdx+0x1fc]{1to4},0x7b

// CHECK:  vcmpps k4, xmm29, dword ptr [rdx + 512]{1to4}, 123
// CHECK:  encoding: [0x62,0xf1,0x14,0x10,0xc2,0xa2,0x00,0x02,0x00,0x00,0x7b]
          vcmpps k4,xmm29,DWORD PTR [rdx+0x200]{1to4},0x7b

// CHECK:  vcmpps k4, xmm29, dword ptr [rdx - 512]{1to4}, 123
// CHECK:  encoding: [0x62,0xf1,0x14,0x10,0xc2,0x62,0x80,0x7b]
          vcmpps k4,xmm29,DWORD PTR [rdx-0x200]{1to4},0x7b

// CHECK:  vcmpps k4, xmm29, dword ptr [rdx - 516]{1to4}, 123
// CHECK:  encoding: [0x62,0xf1,0x14,0x10,0xc2,0xa2,0xfc,0xfd,0xff,0xff,0x7b]
          vcmpps k4,xmm29,DWORD PTR [rdx-0x204]{1to4},0x7b

// CHECK:  vcmpps k4, ymm19, ymm18, 171
// CHECK:  encoding: [0x62,0xb1,0x64,0x20,0xc2,0xe2,0xab]
          vcmpps k4,ymm19,ymm18,0xab

// CHECK:  vcmpps k4 {k1}, ymm19, ymm18, 171
// CHECK:  encoding: [0x62,0xb1,0x64,0x21,0xc2,0xe2,0xab]
          vcmpps k4{k1},ymm19,ymm18,0xab

// CHECK:  vcmpps k4, ymm19, ymm18, 123
// CHECK:  encoding: [0x62,0xb1,0x64,0x20,0xc2,0xe2,0x7b]
          vcmpps k4,ymm19,ymm18,0x7b

// CHECK:  vcmpps k4, ymm19, ymmword ptr [rcx], 123
// CHECK:  encoding: [0x62,0xf1,0x64,0x20,0xc2,0x21,0x7b]
          vcmpps k4,ymm19,YMMWORD PTR [rcx],0x7b

// CHECK:  vcmpps k4, ymm19, ymmword ptr [rax + 8*r14 + 291], 123
// CHECK:  encoding: [0x62,0xb1,0x64,0x20,0xc2,0xa4,0xf0,0x23,0x01,0x00,0x00,0x7b]
          vcmpps k4,ymm19,YMMWORD PTR [rax+r14*8+0x123],0x7b

// CHECK: vcmpps k4, ymm19, dword ptr [rcx]{1to8}, 123
// CHECK:  encoding: [0x62,0xf1,0x64,0x30,0xc2,0x21,0x7b]
          vcmpps k4,ymm19,DWORD PTR [rcx]{1to8},0x7b

// CHECK:  vcmpps k4, ymm19, ymmword ptr [rdx + 4064], 123
// CHECK:  encoding: [0x62,0xf1,0x64,0x20,0xc2,0x62,0x7f,0x7b]
          vcmpps k4,ymm19,YMMWORD PTR [rdx+0xfe0],0x7b

// CHECK:  vcmpps k4, ymm19, ymmword ptr [rdx + 4096], 123
// CHECK:  encoding: [0x62,0xf1,0x64,0x20,0xc2,0xa2,0x00,0x10,0x00,0x00,0x7b]
          vcmpps k4,ymm19,YMMWORD PTR [rdx+0x1000],0x7b

// CHECK:  vcmpps k4, ymm19, ymmword ptr [rdx - 4096], 123
// CHECK:  encoding: [0x62,0xf1,0x64,0x20,0xc2,0x62,0x80,0x7b]
          vcmpps k4,ymm19,YMMWORD PTR [rdx-0x1000],0x7b

// CHECK:  vcmpps k4, ymm19, ymmword ptr [rdx - 4128], 123
// CHECK:  encoding: [0x62,0xf1,0x64,0x20,0xc2,0xa2,0xe0,0xef,0xff,0xff,0x7b]
          vcmpps k4,ymm19,YMMWORD PTR [rdx-0x1020],0x7b

// CHECK:  vcmpps k4, ymm19, dword ptr [rdx + 508]{1to8}, 123
// CHECK:  encoding: [0x62,0xf1,0x64,0x30,0xc2,0x62,0x7f,0x7b]
          vcmpps k4,ymm19,DWORD PTR [rdx+0x1fc]{1to8},0x7b

// CHECK:  vcmpps k4, ymm19, dword ptr [rdx + 512]{1to8}, 123
// CHECK:  encoding: [0x62,0xf1,0x64,0x30,0xc2,0xa2,0x00,0x02,0x00,0x00,0x7b]
          vcmpps k4,ymm19,DWORD PTR [rdx+0x200]{1to8},0x7b

// CHECK:  vcmpps k4, ymm19, dword ptr [rdx - 512]{1to8}, 123
// CHECK:  encoding: [0x62,0xf1,0x64,0x30,0xc2,0x62,0x80,0x7b]
          vcmpps k4,ymm19,DWORD PTR [rdx-0x200]{1to8},0x7b

// CHECK:  vcmpps k4, ymm19, dword ptr [rdx - 516]{1to8}, 123
// CHECK:  encoding: [0x62,0xf1,0x64,0x30,0xc2,0xa2,0xfc,0xfd,0xff,0xff,0x7b]
          vcmpps k4,ymm19,DWORD PTR [rdx-0x204]{1to8},0x7b

// CHECK: vgatherdpd	xmm17 {k1}, xmmword ptr [r14 + 8*xmm31 + 123] 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0x92,0x8c,0xfe,0x7b,0x00,0x00,0x00]
          vgatherdpd	xmm17 {k1}, xmmword ptr [r14 + 8*xmm31 + 123] 

// CHECK: vgatherdpd	xmm17 {k1}, xmmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0x92,0x4c,0x39,0x20]
          vgatherdpd	xmm17 {k1}, xmmword ptr [r9 + xmm31 + 256] 

// CHECK: vgatherdpd	xmm17 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0xfd,0x01,0x92,0x8c,0xb9,0x00,0x04,0x00,0x00]
          vgatherdpd	xmm17 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vgatherdpd	ymm23 {k1}, ymmword ptr [r14 + 8*xmm31 + 123] 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0x92,0xbc,0xfe,0x7b,0x00,0x00,0x00]
          vgatherdpd	ymm23 {k1}, ymmword ptr [r14 + 8*xmm31 + 123] 

// CHECK: vgatherdpd	ymm23 {k1}, ymmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0x92,0x7c,0x39,0x20]
          vgatherdpd	ymm23 {k1}, ymmword ptr [r9 + xmm31 + 256] 

// CHECK: vgatherdpd	ymm23 {k1}, ymmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0xfd,0x21,0x92,0xbc,0xb9,0x00,0x04,0x00,0x00]
          vgatherdpd	ymm23 {k1}, ymmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vgatherdpd	xmm23 {k1}, xmmword ptr [r14 + 8*xmm31 - 123] 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0x92,0xbc,0xfe,0x85,0xff,0xff,0xff]
          vgatherdpd	xmm23 {k1}, xmmword ptr [r14 + 8*xmm31 - 123] 

// CHECK: vgatherdpd	xmm23 {k1}, xmmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0x92,0x7c,0x39,0x20]
          vgatherdpd	xmm23 {k1}, xmmword ptr [r9 + xmm31 + 256] 

// CHECK: vgatherdpd	xmm23 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0xfd,0x01,0x92,0xbc,0xb9,0x00,0x04,0x00,0x00]
          vgatherdpd	xmm23 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vgatherdpd	ymm18 {k1}, ymmword ptr [r14 + 8*xmm31 - 123] 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0x92,0x94,0xfe,0x85,0xff,0xff,0xff]
          vgatherdpd	ymm18 {k1}, ymmword ptr [r14 + 8*xmm31 - 123] 

// CHECK: vgatherdpd	ymm18 {k1}, ymmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0x92,0x54,0x39,0x20]
          vgatherdpd	ymm18 {k1}, ymmword ptr [r9 + xmm31 + 256] 

// CHECK: vgatherdpd	ymm18 {k1}, ymmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0xfd,0x21,0x92,0x94,0xb9,0x00,0x04,0x00,0x00]
          vgatherdpd	ymm18 {k1}, ymmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vgatherdps	xmm18 {k1}, xmmword ptr [r14 + 8*xmm31 + 123] 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0x92,0x94,0xfe,0x7b,0x00,0x00,0x00]
          vgatherdps	xmm18 {k1}, xmmword ptr [r14 + 8*xmm31 + 123] 

// CHECK: vgatherdps	xmm18 {k1}, xmmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0x92,0x54,0x39,0x40]
          vgatherdps	xmm18 {k1}, xmmword ptr [r9 + xmm31 + 256] 

// CHECK: vgatherdps	xmm18 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0x7d,0x01,0x92,0x94,0xb9,0x00,0x04,0x00,0x00]
          vgatherdps	xmm18 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vgatherdps	ymm27 {k1}, ymmword ptr [r14 + 8*ymm31 + 123] 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0x92,0x9c,0xfe,0x7b,0x00,0x00,0x00]
          vgatherdps	ymm27 {k1}, ymmword ptr [r14 + 8*ymm31 + 123] 

// CHECK: vgatherdps	ymm27 {k1}, ymmword ptr [r9 + ymm31 + 256] 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0x92,0x5c,0x39,0x40]
          vgatherdps	ymm27 {k1}, ymmword ptr [r9 + ymm31 + 256] 

// CHECK: vgatherdps	ymm27 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 
// CHECK: encoding: [0x62,0x22,0x7d,0x21,0x92,0x9c,0xb9,0x00,0x04,0x00,0x00]
          vgatherdps	ymm27 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 

// CHECK: vgatherdps	xmm29 {k1}, xmmword ptr [r14 + 8*xmm31 - 123] 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0x92,0xac,0xfe,0x85,0xff,0xff,0xff]
          vgatherdps	xmm29 {k1}, xmmword ptr [r14 + 8*xmm31 - 123] 

// CHECK: vgatherdps	xmm29 {k1}, xmmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0x92,0x6c,0x39,0x40]
          vgatherdps	xmm29 {k1}, xmmword ptr [r9 + xmm31 + 256] 

// CHECK: vgatherdps	xmm29 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0x22,0x7d,0x01,0x92,0xac,0xb9,0x00,0x04,0x00,0x00]
          vgatherdps	xmm29 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vgatherdps	ymm21 {k1}, ymmword ptr [r14 + 8*ymm31 - 123] 
// CHECK: encoding: [0x62,0x82,0x7d,0x21,0x92,0xac,0xfe,0x85,0xff,0xff,0xff]
          vgatherdps	ymm21 {k1}, ymmword ptr [r14 + 8*ymm31 - 123] 

// CHECK: vgatherdps	ymm21 {k1}, ymmword ptr [r9 + ymm31 + 256] 
// CHECK: encoding: [0x62,0x82,0x7d,0x21,0x92,0x6c,0x39,0x40]
          vgatherdps	ymm21 {k1}, ymmword ptr [r9 + ymm31 + 256] 

// CHECK: vgatherdps	ymm21 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0x7d,0x21,0x92,0xac,0xb9,0x00,0x04,0x00,0x00]
          vgatherdps	ymm21 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 

// CHECK: vgatherqpd	xmm17 {k1}, xmmword ptr [r14 + 8*xmm31 + 123] 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0x93,0x8c,0xfe,0x7b,0x00,0x00,0x00]
          vgatherqpd	xmm17 {k1}, xmmword ptr [r14 + 8*xmm31 + 123] 

// CHECK: vgatherqpd	xmm17 {k1}, xmmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0x93,0x4c,0x39,0x20]
          vgatherqpd	xmm17 {k1}, xmmword ptr [r9 + xmm31 + 256] 

// CHECK: vgatherqpd	xmm17 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0xfd,0x01,0x93,0x8c,0xb9,0x00,0x04,0x00,0x00]
          vgatherqpd	xmm17 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vgatherqpd	ymm29 {k1}, ymmword ptr [r14 + 8*ymm31 + 123] 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0x93,0xac,0xfe,0x7b,0x00,0x00,0x00]
          vgatherqpd	ymm29 {k1}, ymmword ptr [r14 + 8*ymm31 + 123] 

// CHECK: vgatherqpd	ymm29 {k1}, ymmword ptr [r9 + ymm31 + 256] 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0x93,0x6c,0x39,0x20]
          vgatherqpd	ymm29 {k1}, ymmword ptr [r9 + ymm31 + 256] 

// CHECK: vgatherqpd	ymm29 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 
// CHECK: encoding: [0x62,0x22,0xfd,0x21,0x93,0xac,0xb9,0x00,0x04,0x00,0x00]
          vgatherqpd	ymm29 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 

// CHECK: vgatherqpd	xmm18 {k1}, xmmword ptr [r14 + 8*xmm31 - 123] 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0x93,0x94,0xfe,0x85,0xff,0xff,0xff]
          vgatherqpd	xmm18 {k1}, xmmword ptr [r14 + 8*xmm31 - 123] 

// CHECK: vgatherqpd	xmm18 {k1}, xmmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0x93,0x54,0x39,0x20]
          vgatherqpd	xmm18 {k1}, xmmword ptr [r9 + xmm31 + 256] 

// CHECK: vgatherqpd	xmm18 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0xfd,0x01,0x93,0x94,0xb9,0x00,0x04,0x00,0x00]
          vgatherqpd	xmm18 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vgatherqpd	ymm21 {k1}, ymmword ptr [r14 + 8*ymm31 - 123] 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0x93,0xac,0xfe,0x85,0xff,0xff,0xff]
          vgatherqpd	ymm21 {k1}, ymmword ptr [r14 + 8*ymm31 - 123] 

// CHECK: vgatherqpd	ymm21 {k1}, ymmword ptr [r9 + ymm31 + 256] 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0x93,0x6c,0x39,0x20]
          vgatherqpd	ymm21 {k1}, ymmword ptr [r9 + ymm31 + 256] 

// CHECK: vgatherqpd	ymm21 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0xfd,0x21,0x93,0xac,0xb9,0x00,0x04,0x00,0x00]
          vgatherqpd	ymm21 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 

// CHECK: vgatherqps	xmm21 {k1}, qword ptr [r14 + 8*xmm31 + 123] 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0x93,0xac,0xfe,0x7b,0x00,0x00,0x00]
          vgatherqps	xmm21 {k1}, qword ptr [r14 + 8*xmm31 + 123] 

// CHECK: vgatherqps	xmm21 {k1}, qword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0x93,0x6c,0x39,0x40]
          vgatherqps	xmm21 {k1}, qword ptr [r9 + xmm31 + 256] 

// CHECK: vgatherqps	xmm21 {k1}, qword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0x7d,0x01,0x93,0xac,0xb9,0x00,0x04,0x00,0x00]
          vgatherqps	xmm21 {k1}, qword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vgatherqps	xmm19 {k1}, xmmword ptr [r14 + 8*ymm31 + 123] 
// CHECK: encoding: [0x62,0x82,0x7d,0x21,0x93,0x9c,0xfe,0x7b,0x00,0x00,0x00]
          vgatherqps	xmm19 {k1}, xmmword ptr [r14 + 8*ymm31 + 123] 

// CHECK: vgatherqps	xmm19 {k1}, xmmword ptr [r9 + ymm31 + 256] 
// CHECK: encoding: [0x62,0x82,0x7d,0x21,0x93,0x5c,0x39,0x40]
          vgatherqps	xmm19 {k1}, xmmword ptr [r9 + ymm31 + 256] 

// CHECK: vgatherqps	xmm19 {k1}, xmmword ptr [rcx + 4*ymm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0x7d,0x21,0x93,0x9c,0xb9,0x00,0x04,0x00,0x00]
          vgatherqps	xmm19 {k1}, xmmword ptr [rcx + 4*ymm31 + 1024] 

// CHECK: vgatherqps	xmm22 {k1}, qword ptr [r14 + 8*xmm31 - 123] 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0x93,0xb4,0xfe,0x85,0xff,0xff,0xff]
          vgatherqps	xmm22 {k1}, qword ptr [r14 + 8*xmm31 - 123] 

// CHECK: vgatherqps	xmm22 {k1}, qword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0x93,0x74,0x39,0x40]
          vgatherqps	xmm22 {k1}, qword ptr [r9 + xmm31 + 256] 

// CHECK: vgatherqps	xmm22 {k1}, qword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0x7d,0x01,0x93,0xb4,0xb9,0x00,0x04,0x00,0x00]
          vgatherqps	xmm22 {k1}, qword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vgatherqps	xmm30 {k1}, xmmword ptr [r14 + 8*ymm31 - 123] 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0x93,0xb4,0xfe,0x85,0xff,0xff,0xff]
          vgatherqps	xmm30 {k1}, xmmword ptr [r14 + 8*ymm31 - 123] 

// CHECK: vgatherqps	xmm30 {k1}, xmmword ptr [r9 + ymm31 + 256] 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0x93,0x74,0x39,0x40]
          vgatherqps	xmm30 {k1}, xmmword ptr [r9 + ymm31 + 256] 

// CHECK: vgatherqps	xmm30 {k1}, xmmword ptr [rcx + 4*ymm31 + 1024] 
// CHECK: encoding: [0x62,0x22,0x7d,0x21,0x93,0xb4,0xb9,0x00,0x04,0x00,0x00]
          vgatherqps	xmm30 {k1}, xmmword ptr [rcx + 4*ymm31 + 1024] 

// CHECK: vpgatherdd	xmm17 {k1}, xmmword ptr [r14 + 8*xmm31 + 123] 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0x90,0x8c,0xfe,0x7b,0x00,0x00,0x00]
          vpgatherdd	xmm17 {k1}, xmmword ptr [r14 + 8*xmm31 + 123] 

// CHECK: vpgatherdd	xmm17 {k1}, xmmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0x90,0x4c,0x39,0x40]
          vpgatherdd	xmm17 {k1}, xmmword ptr [r9 + xmm31 + 256] 

// CHECK: vpgatherdd	xmm17 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0x7d,0x01,0x90,0x8c,0xb9,0x00,0x04,0x00,0x00]
          vpgatherdd	xmm17 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vpgatherdd	ymm19 {k1}, ymmword ptr [r14 + 8*ymm31 + 123] 
// CHECK: encoding: [0x62,0x82,0x7d,0x21,0x90,0x9c,0xfe,0x7b,0x00,0x00,0x00]
          vpgatherdd	ymm19 {k1}, ymmword ptr [r14 + 8*ymm31 + 123] 

// CHECK: vpgatherdd	ymm19 {k1}, ymmword ptr [r9 + ymm31 + 256] 
// CHECK: encoding: [0x62,0x82,0x7d,0x21,0x90,0x5c,0x39,0x40]
          vpgatherdd	ymm19 {k1}, ymmword ptr [r9 + ymm31 + 256] 

// CHECK: vpgatherdd	ymm19 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0x7d,0x21,0x90,0x9c,0xb9,0x00,0x04,0x00,0x00]
          vpgatherdd	ymm19 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 

// CHECK: vpgatherdd	xmm22 {k1}, xmmword ptr [r14 + 8*xmm31 - 123] 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0x90,0xb4,0xfe,0x85,0xff,0xff,0xff]
          vpgatherdd	xmm22 {k1}, xmmword ptr [r14 + 8*xmm31 - 123] 

// CHECK: vpgatherdd	xmm22 {k1}, xmmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0x90,0x74,0x39,0x40]
          vpgatherdd	xmm22 {k1}, xmmword ptr [r9 + xmm31 + 256] 

// CHECK: vpgatherdd	xmm22 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0x7d,0x01,0x90,0xb4,0xb9,0x00,0x04,0x00,0x00]
          vpgatherdd	xmm22 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vpgatherdd	ymm29 {k1}, ymmword ptr [r14 + 8*ymm31 - 123] 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0x90,0xac,0xfe,0x85,0xff,0xff,0xff]
          vpgatherdd	ymm29 {k1}, ymmword ptr [r14 + 8*ymm31 - 123] 

// CHECK: vpgatherdd	ymm29 {k1}, ymmword ptr [r9 + ymm31 + 256] 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0x90,0x6c,0x39,0x40]
          vpgatherdd	ymm29 {k1}, ymmword ptr [r9 + ymm31 + 256] 

// CHECK: vpgatherdd	ymm29 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 
// CHECK: encoding: [0x62,0x22,0x7d,0x21,0x90,0xac,0xb9,0x00,0x04,0x00,0x00]
          vpgatherdd	ymm29 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 

// CHECK: vpgatherdq	xmm17 {k1}, xmmword ptr [r14 + 8*xmm31 + 123] 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0x90,0x8c,0xfe,0x7b,0x00,0x00,0x00]
          vpgatherdq	xmm17 {k1}, xmmword ptr [r14 + 8*xmm31 + 123] 

// CHECK: vpgatherdq	xmm17 {k1}, xmmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0x90,0x4c,0x39,0x20]
          vpgatherdq	xmm17 {k1}, xmmword ptr [r9 + xmm31 + 256] 

// CHECK: vpgatherdq	xmm17 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0xfd,0x01,0x90,0x8c,0xb9,0x00,0x04,0x00,0x00]
          vpgatherdq	xmm17 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vpgatherdq	ymm26 {k1}, ymmword ptr [r14 + 8*xmm31 + 123] 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0x90,0x94,0xfe,0x7b,0x00,0x00,0x00]
          vpgatherdq	ymm26 {k1}, ymmword ptr [r14 + 8*xmm31 + 123] 

// CHECK: vpgatherdq	ymm26 {k1}, ymmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0x90,0x54,0x39,0x20]
          vpgatherdq	ymm26 {k1}, ymmword ptr [r9 + xmm31 + 256] 

// CHECK: vpgatherdq	ymm26 {k1}, ymmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0x22,0xfd,0x21,0x90,0x94,0xb9,0x00,0x04,0x00,0x00]
          vpgatherdq	ymm26 {k1}, ymmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vpgatherdq	xmm25 {k1}, xmmword ptr [r14 + 8*xmm31 - 123] 
// CHECK: encoding: [0x62,0x02,0xfd,0x01,0x90,0x8c,0xfe,0x85,0xff,0xff,0xff]
          vpgatherdq	xmm25 {k1}, xmmword ptr [r14 + 8*xmm31 - 123] 

// CHECK: vpgatherdq	xmm25 {k1}, xmmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x02,0xfd,0x01,0x90,0x4c,0x39,0x20]
          vpgatherdq	xmm25 {k1}, xmmword ptr [r9 + xmm31 + 256] 

// CHECK: vpgatherdq	xmm25 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0x22,0xfd,0x01,0x90,0x8c,0xb9,0x00,0x04,0x00,0x00]
          vpgatherdq	xmm25 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vpgatherdq	ymm22 {k1}, ymmword ptr [r14 + 8*xmm31 - 123] 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0x90,0xb4,0xfe,0x85,0xff,0xff,0xff]
          vpgatherdq	ymm22 {k1}, ymmword ptr [r14 + 8*xmm31 - 123] 

// CHECK: vpgatherdq	ymm22 {k1}, ymmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0x90,0x74,0x39,0x20]
          vpgatherdq	ymm22 {k1}, ymmword ptr [r9 + xmm31 + 256] 

// CHECK: vpgatherdq	ymm22 {k1}, ymmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0xfd,0x21,0x90,0xb4,0xb9,0x00,0x04,0x00,0x00]
          vpgatherdq	ymm22 {k1}, ymmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vpgatherqd	xmm21 {k1}, qword ptr [r14 + 8*xmm31 + 123] 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0x91,0xac,0xfe,0x7b,0x00,0x00,0x00]
          vpgatherqd	xmm21 {k1}, qword ptr [r14 + 8*xmm31 + 123] 

// CHECK: vpgatherqd	xmm21 {k1}, qword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0x91,0x6c,0x39,0x40]
          vpgatherqd	xmm21 {k1}, qword ptr [r9 + xmm31 + 256] 

// CHECK: vpgatherqd	xmm21 {k1}, qword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0x7d,0x01,0x91,0xac,0xb9,0x00,0x04,0x00,0x00]
          vpgatherqd	xmm21 {k1}, qword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vpgatherqd	xmm25 {k1}, xmmword ptr [r14 + 8*ymm31 + 123] 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0x91,0x8c,0xfe,0x7b,0x00,0x00,0x00]
          vpgatherqd	xmm25 {k1}, xmmword ptr [r14 + 8*ymm31 + 123] 

// CHECK: vpgatherqd	xmm25 {k1}, xmmword ptr [r9 + ymm31 + 256] 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0x91,0x4c,0x39,0x40]
          vpgatherqd	xmm25 {k1}, xmmword ptr [r9 + ymm31 + 256] 

// CHECK: vpgatherqd	xmm25 {k1}, xmmword ptr [rcx + 4*ymm31 + 1024] 
// CHECK: encoding: [0x62,0x22,0x7d,0x21,0x91,0x8c,0xb9,0x00,0x04,0x00,0x00]
          vpgatherqd	xmm25 {k1}, xmmword ptr [rcx + 4*ymm31 + 1024] 

// CHECK: vpgatherqd	xmm30 {k1}, qword ptr [r14 + 8*xmm31 - 123] 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0x91,0xb4,0xfe,0x85,0xff,0xff,0xff]
          vpgatherqd	xmm30 {k1}, qword ptr [r14 + 8*xmm31 - 123] 

// CHECK: vpgatherqd	xmm30 {k1}, qword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0x91,0x74,0x39,0x40]
          vpgatherqd	xmm30 {k1}, qword ptr [r9 + xmm31 + 256] 

// CHECK: vpgatherqd	xmm30 {k1}, qword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0x22,0x7d,0x01,0x91,0xb4,0xb9,0x00,0x04,0x00,0x00]
          vpgatherqd	xmm30 {k1}, qword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vpgatherqd	xmm28 {k1}, xmmword ptr [r14 + 8*ymm31 - 123] 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0x91,0xa4,0xfe,0x85,0xff,0xff,0xff]
          vpgatherqd	xmm28 {k1}, xmmword ptr [r14 + 8*ymm31 - 123] 

// CHECK: vpgatherqd	xmm28 {k1}, xmmword ptr [r9 + ymm31 + 256] 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0x91,0x64,0x39,0x40]
          vpgatherqd	xmm28 {k1}, xmmword ptr [r9 + ymm31 + 256] 

// CHECK: vpgatherqd	xmm28 {k1}, xmmword ptr [rcx + 4*ymm31 + 1024] 
// CHECK: encoding: [0x62,0x22,0x7d,0x21,0x91,0xa4,0xb9,0x00,0x04,0x00,0x00]
          vpgatherqd	xmm28 {k1}, xmmword ptr [rcx + 4*ymm31 + 1024] 

// CHECK: vpgatherqq	xmm18 {k1}, xmmword ptr [r14 + 8*xmm31 + 123] 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0x91,0x94,0xfe,0x7b,0x00,0x00,0x00]
          vpgatherqq	xmm18 {k1}, xmmword ptr [r14 + 8*xmm31 + 123] 

// CHECK: vpgatherqq	xmm18 {k1}, xmmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0x91,0x54,0x39,0x20]
          vpgatherqq	xmm18 {k1}, xmmword ptr [r9 + xmm31 + 256] 

// CHECK: vpgatherqq	xmm18 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0xfd,0x01,0x91,0x94,0xb9,0x00,0x04,0x00,0x00]
          vpgatherqq	xmm18 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vpgatherqq	ymm19 {k1}, ymmword ptr [r14 + 8*ymm31 + 123] 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0x91,0x9c,0xfe,0x7b,0x00,0x00,0x00]
          vpgatherqq	ymm19 {k1}, ymmword ptr [r14 + 8*ymm31 + 123] 

// CHECK: vpgatherqq	ymm19 {k1}, ymmword ptr [r9 + ymm31 + 256] 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0x91,0x5c,0x39,0x20]
          vpgatherqq	ymm19 {k1}, ymmword ptr [r9 + ymm31 + 256] 

// CHECK: vpgatherqq	ymm19 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0xfd,0x21,0x91,0x9c,0xb9,0x00,0x04,0x00,0x00]
          vpgatherqq	ymm19 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 

// CHECK: vpgatherqq	xmm23 {k1}, xmmword ptr [r14 + 8*xmm31 - 123] 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0x91,0xbc,0xfe,0x85,0xff,0xff,0xff]
          vpgatherqq	xmm23 {k1}, xmmword ptr [r14 + 8*xmm31 - 123] 

// CHECK: vpgatherqq	xmm23 {k1}, xmmword ptr [r9 + xmm31 + 256] 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0x91,0x7c,0x39,0x20]
          vpgatherqq	xmm23 {k1}, xmmword ptr [r9 + xmm31 + 256] 

// CHECK: vpgatherqq	xmm23 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 
// CHECK: encoding: [0x62,0xa2,0xfd,0x01,0x91,0xbc,0xb9,0x00,0x04,0x00,0x00]
          vpgatherqq	xmm23 {k1}, xmmword ptr [rcx + 4*xmm31 + 1024] 

// CHECK: vpgatherqq	ymm26 {k1}, ymmword ptr [r14 + 8*ymm31 - 123] 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0x91,0x94,0xfe,0x85,0xff,0xff,0xff]
          vpgatherqq	ymm26 {k1}, ymmword ptr [r14 + 8*ymm31 - 123] 

// CHECK: vpgatherqq	ymm26 {k1}, ymmword ptr [r9 + ymm31 + 256] 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0x91,0x54,0x39,0x20]
          vpgatherqq	ymm26 {k1}, ymmword ptr [r9 + ymm31 + 256] 

// CHECK: vpgatherqq	ymm26 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 
// CHECK: encoding: [0x62,0x22,0xfd,0x21,0x91,0x94,0xb9,0x00,0x04,0x00,0x00]
          vpgatherqq	ymm26 {k1}, ymmword ptr [rcx + 4*ymm31 + 1024] 

// CHECK: vpscatterdd	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm20 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0xa0,0xa4,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterdd	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm20 

// CHECK: vpscatterdd	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm20 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0xa0,0xa4,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterdd	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm20 

// CHECK: vpscatterdd	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm20 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0xa0,0x64,0x39,0x40]
          vpscatterdd	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm20 

// CHECK: vpscatterdd	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm20 
// CHECK: encoding: [0x62,0xa2,0x7d,0x01,0xa0,0xa4,0xb9,0x00,0x04,0x00,0x00]
          vpscatterdd	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm20 

// CHECK: vpscatterdd	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm28 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa0,0xa4,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterdd	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm28 

// CHECK: vpscatterdd	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm28 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa0,0xa4,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterdd	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm28 

// CHECK: vpscatterdd	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm28 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa0,0x64,0x39,0x40]
          vpscatterdd	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm28 

// CHECK: vpscatterdd	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm28 
// CHECK: encoding: [0x62,0x22,0x7d,0x21,0xa0,0xa4,0xb9,0x00,0x04,0x00,0x00]
          vpscatterdd	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm28 

// CHECK: vpscatterdd	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm17 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0xa0,0x8c,0xfe,0x85,0xff,0xff,0xff]
          vpscatterdd	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm17 

// CHECK: vpscatterdd	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm17 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0xa0,0x8c,0xfe,0x85,0xff,0xff,0xff]
          vpscatterdd	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm17 

// CHECK: vpscatterdd	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm17 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0xa0,0x4c,0x39,0x40]
          vpscatterdd	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm17 

// CHECK: vpscatterdd	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm17 
// CHECK: encoding: [0x62,0xa2,0x7d,0x01,0xa0,0x8c,0xb9,0x00,0x04,0x00,0x00]
          vpscatterdd	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm17 

// CHECK: vpscatterdd	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm24 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa0,0x84,0xfe,0x85,0xff,0xff,0xff]
          vpscatterdd	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm24 

// CHECK: vpscatterdd	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm24 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa0,0x84,0xfe,0x85,0xff,0xff,0xff]
          vpscatterdd	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm24 

// CHECK: vpscatterdd	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm24 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa0,0x44,0x39,0x40]
          vpscatterdd	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm24 

// CHECK: vpscatterdd	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm24 
// CHECK: encoding: [0x62,0x22,0x7d,0x21,0xa0,0x84,0xb9,0x00,0x04,0x00,0x00]
          vpscatterdd	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm24 

// CHECK: vpscatterdq	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm21 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa0,0xac,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterdq	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm21 

// CHECK: vpscatterdq	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm21 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa0,0xac,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterdq	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm21 

// CHECK: vpscatterdq	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm21 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa0,0x6c,0x39,0x20]
          vpscatterdq	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm21 

// CHECK: vpscatterdq	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm21 
// CHECK: encoding: [0x62,0xa2,0xfd,0x01,0xa0,0xac,0xb9,0x00,0x04,0x00,0x00]
          vpscatterdq	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm21 

// CHECK: vpscatterdq	ymmword ptr [r14 + 8*xmm31 + 123] {k1}, ymm28 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0xa0,0xa4,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterdq	ymmword ptr [r14 + 8*xmm31 + 123] {k1}, ymm28 

// CHECK: vpscatterdq	ymmword ptr [r14 + 8*xmm31 + 123] {k1}, ymm28 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0xa0,0xa4,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterdq	ymmword ptr [r14 + 8*xmm31 + 123] {k1}, ymm28 

// CHECK: vpscatterdq	ymmword ptr [r9 + xmm31 + 256] {k1}, ymm28 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0xa0,0x64,0x39,0x20]
          vpscatterdq	ymmword ptr [r9 + xmm31 + 256] {k1}, ymm28 

// CHECK: vpscatterdq	ymmword ptr [rcx + 4*xmm31 + 1024] {k1}, ymm28 
// CHECK: encoding: [0x62,0x22,0xfd,0x21,0xa0,0xa4,0xb9,0x00,0x04,0x00,0x00]
          vpscatterdq	ymmword ptr [rcx + 4*xmm31 + 1024] {k1}, ymm28 

// CHECK: vpscatterdq	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm28 
// CHECK: encoding: [0x62,0x02,0xfd,0x01,0xa0,0xa4,0xfe,0x85,0xff,0xff,0xff]
          vpscatterdq	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm28 

// CHECK: vpscatterdq	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm28 
// CHECK: encoding: [0x62,0x02,0xfd,0x01,0xa0,0xa4,0xfe,0x85,0xff,0xff,0xff]
          vpscatterdq	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm28 

// CHECK: vpscatterdq	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm28 
// CHECK: encoding: [0x62,0x02,0xfd,0x01,0xa0,0x64,0x39,0x20]
          vpscatterdq	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm28 

// CHECK: vpscatterdq	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm28 
// CHECK: encoding: [0x62,0x22,0xfd,0x01,0xa0,0xa4,0xb9,0x00,0x04,0x00,0x00]
          vpscatterdq	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm28 

// CHECK: vpscatterdq	ymmword ptr [r14 + 8*xmm31 - 123] {k1}, ymm20 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0xa0,0xa4,0xfe,0x85,0xff,0xff,0xff]
          vpscatterdq	ymmword ptr [r14 + 8*xmm31 - 123] {k1}, ymm20 

// CHECK: vpscatterdq	ymmword ptr [r14 + 8*xmm31 - 123] {k1}, ymm20 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0xa0,0xa4,0xfe,0x85,0xff,0xff,0xff]
          vpscatterdq	ymmword ptr [r14 + 8*xmm31 - 123] {k1}, ymm20 

// CHECK: vpscatterdq	ymmword ptr [r9 + xmm31 + 256] {k1}, ymm20 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0xa0,0x64,0x39,0x20]
          vpscatterdq	ymmword ptr [r9 + xmm31 + 256] {k1}, ymm20 

// CHECK: vpscatterdq	ymmword ptr [rcx + 4*xmm31 + 1024] {k1}, ymm20 
// CHECK: encoding: [0x62,0xa2,0xfd,0x21,0xa0,0xa4,0xb9,0x00,0x04,0x00,0x00]
          vpscatterdq	ymmword ptr [rcx + 4*xmm31 + 1024] {k1}, ymm20 

// CHECK: vpscatterqd	qword ptr [r14 + 8*xmm31 + 123] {k1}, xmm22 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0xa1,0xb4,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterqd	qword ptr [r14 + 8*xmm31 + 123] {k1}, xmm22 

// CHECK: vpscatterqd	qword ptr [r14 + 8*xmm31 + 123] {k1}, xmm22 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0xa1,0xb4,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterqd	qword ptr [r14 + 8*xmm31 + 123] {k1}, xmm22 

// CHECK: vpscatterqd	qword ptr [r9 + xmm31 + 256] {k1}, xmm22 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0xa1,0x74,0x39,0x40]
          vpscatterqd	qword ptr [r9 + xmm31 + 256] {k1}, xmm22 

// CHECK: vpscatterqd	qword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm22 
// CHECK: encoding: [0x62,0xa2,0x7d,0x01,0xa1,0xb4,0xb9,0x00,0x04,0x00,0x00]
          vpscatterqd	qword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm22 

// CHECK: vpscatterqd	xmmword ptr [r14 + 8*ymm31 + 123] {k1}, xmm24 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa1,0x84,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterqd	xmmword ptr [r14 + 8*ymm31 + 123] {k1}, xmm24 

// CHECK: vpscatterqd	xmmword ptr [r14 + 8*ymm31 + 123] {k1}, xmm24 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa1,0x84,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterqd	xmmword ptr [r14 + 8*ymm31 + 123] {k1}, xmm24 

// CHECK: vpscatterqd	xmmword ptr [r9 + ymm31 + 256] {k1}, xmm24 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa1,0x44,0x39,0x40]
          vpscatterqd	xmmword ptr [r9 + ymm31 + 256] {k1}, xmm24 

// CHECK: vpscatterqd	xmmword ptr [rcx + 4*ymm31 + 1024] {k1}, xmm24 
// CHECK: encoding: [0x62,0x22,0x7d,0x21,0xa1,0x84,0xb9,0x00,0x04,0x00,0x00]
          vpscatterqd	xmmword ptr [rcx + 4*ymm31 + 1024] {k1}, xmm24 

// CHECK: vpscatterqd	qword ptr [r14 + 8*xmm31 - 123] {k1}, xmm22 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0xa1,0xb4,0xfe,0x85,0xff,0xff,0xff]
          vpscatterqd	qword ptr [r14 + 8*xmm31 - 123] {k1}, xmm22 

// CHECK: vpscatterqd	qword ptr [r14 + 8*xmm31 - 123] {k1}, xmm22 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0xa1,0xb4,0xfe,0x85,0xff,0xff,0xff]
          vpscatterqd	qword ptr [r14 + 8*xmm31 - 123] {k1}, xmm22 

// CHECK: vpscatterqd	qword ptr [r9 + xmm31 + 256] {k1}, xmm22 
// CHECK: encoding: [0x62,0x82,0x7d,0x01,0xa1,0x74,0x39,0x40]
          vpscatterqd	qword ptr [r9 + xmm31 + 256] {k1}, xmm22 

// CHECK: vpscatterqd	qword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm22 
// CHECK: encoding: [0x62,0xa2,0x7d,0x01,0xa1,0xb4,0xb9,0x00,0x04,0x00,0x00]
          vpscatterqd	qword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm22 

// CHECK: vpscatterqd	xmmword ptr [r14 + 8*ymm31 - 123] {k1}, xmm29 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa1,0xac,0xfe,0x85,0xff,0xff,0xff]
          vpscatterqd	xmmword ptr [r14 + 8*ymm31 - 123] {k1}, xmm29 

// CHECK: vpscatterqd	xmmword ptr [r14 + 8*ymm31 - 123] {k1}, xmm29 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa1,0xac,0xfe,0x85,0xff,0xff,0xff]
          vpscatterqd	xmmword ptr [r14 + 8*ymm31 - 123] {k1}, xmm29 

// CHECK: vpscatterqd	xmmword ptr [r9 + ymm31 + 256] {k1}, xmm29 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa1,0x6c,0x39,0x40]
          vpscatterqd	xmmword ptr [r9 + ymm31 + 256] {k1}, xmm29 

// CHECK: vpscatterqd	xmmword ptr [rcx + 4*ymm31 + 1024] {k1}, xmm29 
// CHECK: encoding: [0x62,0x22,0x7d,0x21,0xa1,0xac,0xb9,0x00,0x04,0x00,0x00]
          vpscatterqd	xmmword ptr [rcx + 4*ymm31 + 1024] {k1}, xmm29 

// CHECK: vpscatterqq	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm28 
// CHECK: encoding: [0x62,0x02,0xfd,0x01,0xa1,0xa4,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterqq	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm28 

// CHECK: vpscatterqq	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm28 
// CHECK: encoding: [0x62,0x02,0xfd,0x01,0xa1,0xa4,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterqq	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm28 

// CHECK: vpscatterqq	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm28 
// CHECK: encoding: [0x62,0x02,0xfd,0x01,0xa1,0x64,0x39,0x20]
          vpscatterqq	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm28 

// CHECK: vpscatterqq	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm28 
// CHECK: encoding: [0x62,0x22,0xfd,0x01,0xa1,0xa4,0xb9,0x00,0x04,0x00,0x00]
          vpscatterqq	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm28 

// CHECK: vpscatterqq	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm19 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0xa1,0x9c,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterqq	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm19 

// CHECK: vpscatterqq	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm19 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0xa1,0x9c,0xfe,0x7b,0x00,0x00,0x00]
          vpscatterqq	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm19 

// CHECK: vpscatterqq	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm19 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0xa1,0x5c,0x39,0x20]
          vpscatterqq	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm19 

// CHECK: vpscatterqq	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm19 
// CHECK: encoding: [0x62,0xa2,0xfd,0x21,0xa1,0x9c,0xb9,0x00,0x04,0x00,0x00]
          vpscatterqq	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm19 

// CHECK: vpscatterqq	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm24 
// CHECK: encoding: [0x62,0x02,0xfd,0x01,0xa1,0x84,0xfe,0x85,0xff,0xff,0xff]
          vpscatterqq	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm24 

// CHECK: vpscatterqq	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm24 
// CHECK: encoding: [0x62,0x02,0xfd,0x01,0xa1,0x84,0xfe,0x85,0xff,0xff,0xff]
          vpscatterqq	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm24 

// CHECK: vpscatterqq	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm24 
// CHECK: encoding: [0x62,0x02,0xfd,0x01,0xa1,0x44,0x39,0x20]
          vpscatterqq	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm24 

// CHECK: vpscatterqq	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm24 
// CHECK: encoding: [0x62,0x22,0xfd,0x01,0xa1,0x84,0xb9,0x00,0x04,0x00,0x00]
          vpscatterqq	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm24 

// CHECK: vpscatterqq	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm17 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0xa1,0x8c,0xfe,0x85,0xff,0xff,0xff]
          vpscatterqq	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm17 

// CHECK: vpscatterqq	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm17 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0xa1,0x8c,0xfe,0x85,0xff,0xff,0xff]
          vpscatterqq	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm17 

// CHECK: vpscatterqq	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm17 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0xa1,0x4c,0x39,0x20]
          vpscatterqq	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm17 

// CHECK: vpscatterqq	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm17 
// CHECK: encoding: [0x62,0xa2,0xfd,0x21,0xa1,0x8c,0xb9,0x00,0x04,0x00,0x00]
          vpscatterqq	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm17 

// CHECK: vscatterdpd	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm18 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa2,0x94,0xfe,0x7b,0x00,0x00,0x00]
          vscatterdpd	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm18 

// CHECK: vscatterdpd	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm18 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa2,0x94,0xfe,0x7b,0x00,0x00,0x00]
          vscatterdpd	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm18 

// CHECK: vscatterdpd	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm18 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa2,0x54,0x39,0x20]
          vscatterdpd	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm18 

// CHECK: vscatterdpd	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm18 
// CHECK: encoding: [0x62,0xa2,0xfd,0x01,0xa2,0x94,0xb9,0x00,0x04,0x00,0x00]
          vscatterdpd	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm18 

// CHECK: vscatterdpd	ymmword ptr [r14 + 8*xmm31 + 123] {k1}, ymm30 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0xa2,0xb4,0xfe,0x7b,0x00,0x00,0x00]
          vscatterdpd	ymmword ptr [r14 + 8*xmm31 + 123] {k1}, ymm30 

// CHECK: vscatterdpd	ymmword ptr [r14 + 8*xmm31 + 123] {k1}, ymm30 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0xa2,0xb4,0xfe,0x7b,0x00,0x00,0x00]
          vscatterdpd	ymmword ptr [r14 + 8*xmm31 + 123] {k1}, ymm30 

// CHECK: vscatterdpd	ymmword ptr [r9 + xmm31 + 256] {k1}, ymm30 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0xa2,0x74,0x39,0x20]
          vscatterdpd	ymmword ptr [r9 + xmm31 + 256] {k1}, ymm30 

// CHECK: vscatterdpd	ymmword ptr [rcx + 4*xmm31 + 1024] {k1}, ymm30 
// CHECK: encoding: [0x62,0x22,0xfd,0x21,0xa2,0xb4,0xb9,0x00,0x04,0x00,0x00]
          vscatterdpd	ymmword ptr [rcx + 4*xmm31 + 1024] {k1}, ymm30 

// CHECK: vscatterdpd	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm19 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa2,0x9c,0xfe,0x85,0xff,0xff,0xff]
          vscatterdpd	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm19 

// CHECK: vscatterdpd	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm19 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa2,0x9c,0xfe,0x85,0xff,0xff,0xff]
          vscatterdpd	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm19 

// CHECK: vscatterdpd	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm19 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa2,0x5c,0x39,0x20]
          vscatterdpd	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm19 

// CHECK: vscatterdpd	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm19 
// CHECK: encoding: [0x62,0xa2,0xfd,0x01,0xa2,0x9c,0xb9,0x00,0x04,0x00,0x00]
          vscatterdpd	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm19 

// CHECK: vscatterdpd	ymmword ptr [r14 + 8*xmm31 - 123] {k1}, ymm26 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0xa2,0x94,0xfe,0x85,0xff,0xff,0xff]
          vscatterdpd	ymmword ptr [r14 + 8*xmm31 - 123] {k1}, ymm26 

// CHECK: vscatterdpd	ymmword ptr [r14 + 8*xmm31 - 123] {k1}, ymm26 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0xa2,0x94,0xfe,0x85,0xff,0xff,0xff]
          vscatterdpd	ymmword ptr [r14 + 8*xmm31 - 123] {k1}, ymm26 

// CHECK: vscatterdpd	ymmword ptr [r9 + xmm31 + 256] {k1}, ymm26 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0xa2,0x54,0x39,0x20]
          vscatterdpd	ymmword ptr [r9 + xmm31 + 256] {k1}, ymm26 

// CHECK: vscatterdpd	ymmword ptr [rcx + 4*xmm31 + 1024] {k1}, ymm26 
// CHECK: encoding: [0x62,0x22,0xfd,0x21,0xa2,0x94,0xb9,0x00,0x04,0x00,0x00]
          vscatterdpd	ymmword ptr [rcx + 4*xmm31 + 1024] {k1}, ymm26 

// CHECK: vscatterdps	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm24 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0xa2,0x84,0xfe,0x7b,0x00,0x00,0x00]
          vscatterdps	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm24 

// CHECK: vscatterdps	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm24 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0xa2,0x84,0xfe,0x7b,0x00,0x00,0x00]
          vscatterdps	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm24 

// CHECK: vscatterdps	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm24 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0xa2,0x44,0x39,0x40]
          vscatterdps	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm24 

// CHECK: vscatterdps	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm24 
// CHECK: encoding: [0x62,0x22,0x7d,0x01,0xa2,0x84,0xb9,0x00,0x04,0x00,0x00]
          vscatterdps	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm24 

// CHECK: vscatterdps	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm23 
// CHECK: encoding: [0x62,0x82,0x7d,0x21,0xa2,0xbc,0xfe,0x7b,0x00,0x00,0x00]
          vscatterdps	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm23 

// CHECK: vscatterdps	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm23 
// CHECK: encoding: [0x62,0x82,0x7d,0x21,0xa2,0xbc,0xfe,0x7b,0x00,0x00,0x00]
          vscatterdps	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm23 

// CHECK: vscatterdps	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm23 
// CHECK: encoding: [0x62,0x82,0x7d,0x21,0xa2,0x7c,0x39,0x40]
          vscatterdps	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm23 

// CHECK: vscatterdps	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm23 
// CHECK: encoding: [0x62,0xa2,0x7d,0x21,0xa2,0xbc,0xb9,0x00,0x04,0x00,0x00]
          vscatterdps	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm23 

// CHECK: vscatterdps	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm28 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0xa2,0xa4,0xfe,0x85,0xff,0xff,0xff]
          vscatterdps	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm28 

// CHECK: vscatterdps	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm28 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0xa2,0xa4,0xfe,0x85,0xff,0xff,0xff]
          vscatterdps	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm28 

// CHECK: vscatterdps	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm28 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0xa2,0x64,0x39,0x40]
          vscatterdps	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm28 

// CHECK: vscatterdps	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm28 
// CHECK: encoding: [0x62,0x22,0x7d,0x01,0xa2,0xa4,0xb9,0x00,0x04,0x00,0x00]
          vscatterdps	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm28 

// CHECK: vscatterdps	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm25 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa2,0x8c,0xfe,0x85,0xff,0xff,0xff]
          vscatterdps	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm25 

// CHECK: vscatterdps	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm25 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa2,0x8c,0xfe,0x85,0xff,0xff,0xff]
          vscatterdps	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm25 

// CHECK: vscatterdps	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm25 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa2,0x4c,0x39,0x40]
          vscatterdps	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm25 

// CHECK: vscatterdps	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm25 
// CHECK: encoding: [0x62,0x22,0x7d,0x21,0xa2,0x8c,0xb9,0x00,0x04,0x00,0x00]
          vscatterdps	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm25 

// CHECK: vscatterqpd	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm21 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa3,0xac,0xfe,0x7b,0x00,0x00,0x00]
          vscatterqpd	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm21 

// CHECK: vscatterqpd	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm21 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa3,0xac,0xfe,0x7b,0x00,0x00,0x00]
          vscatterqpd	xmmword ptr [r14 + 8*xmm31 + 123] {k1}, xmm21 

// CHECK: vscatterqpd	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm21 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa3,0x6c,0x39,0x20]
          vscatterqpd	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm21 

// CHECK: vscatterqpd	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm21 
// CHECK: encoding: [0x62,0xa2,0xfd,0x01,0xa3,0xac,0xb9,0x00,0x04,0x00,0x00]
          vscatterqpd	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm21 

// CHECK: vscatterqpd	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm23 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0xa3,0xbc,0xfe,0x7b,0x00,0x00,0x00]
          vscatterqpd	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm23 

// CHECK: vscatterqpd	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm23 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0xa3,0xbc,0xfe,0x7b,0x00,0x00,0x00]
          vscatterqpd	ymmword ptr [r14 + 8*ymm31 + 123] {k1}, ymm23 

// CHECK: vscatterqpd	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm23 
// CHECK: encoding: [0x62,0x82,0xfd,0x21,0xa3,0x7c,0x39,0x20]
          vscatterqpd	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm23 

// CHECK: vscatterqpd	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm23 
// CHECK: encoding: [0x62,0xa2,0xfd,0x21,0xa3,0xbc,0xb9,0x00,0x04,0x00,0x00]
          vscatterqpd	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm23 

// CHECK: vscatterqpd	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm19 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa3,0x9c,0xfe,0x85,0xff,0xff,0xff]
          vscatterqpd	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm19 

// CHECK: vscatterqpd	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm19 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa3,0x9c,0xfe,0x85,0xff,0xff,0xff]
          vscatterqpd	xmmword ptr [r14 + 8*xmm31 - 123] {k1}, xmm19 

// CHECK: vscatterqpd	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm19 
// CHECK: encoding: [0x62,0x82,0xfd,0x01,0xa3,0x5c,0x39,0x20]
          vscatterqpd	xmmword ptr [r9 + xmm31 + 256] {k1}, xmm19 

// CHECK: vscatterqpd	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm19 
// CHECK: encoding: [0x62,0xa2,0xfd,0x01,0xa3,0x9c,0xb9,0x00,0x04,0x00,0x00]
          vscatterqpd	xmmword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm19 

// CHECK: vscatterqpd	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm29 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0xa3,0xac,0xfe,0x85,0xff,0xff,0xff]
          vscatterqpd	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm29 

// CHECK: vscatterqpd	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm29 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0xa3,0xac,0xfe,0x85,0xff,0xff,0xff]
          vscatterqpd	ymmword ptr [r14 + 8*ymm31 - 123] {k1}, ymm29 

// CHECK: vscatterqpd	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm29 
// CHECK: encoding: [0x62,0x02,0xfd,0x21,0xa3,0x6c,0x39,0x20]
          vscatterqpd	ymmword ptr [r9 + ymm31 + 256] {k1}, ymm29 

// CHECK: vscatterqpd	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm29 
// CHECK: encoding: [0x62,0x22,0xfd,0x21,0xa3,0xac,0xb9,0x00,0x04,0x00,0x00]
          vscatterqpd	ymmword ptr [rcx + 4*ymm31 + 1024] {k1}, ymm29 

// CHECK: vscatterqps	qword ptr [r14 + 8*xmm31 + 123] {k1}, xmm28 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0xa3,0xa4,0xfe,0x7b,0x00,0x00,0x00]
          vscatterqps	qword ptr [r14 + 8*xmm31 + 123] {k1}, xmm28 

// CHECK: vscatterqps	qword ptr [r14 + 8*xmm31 + 123] {k1}, xmm28 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0xa3,0xa4,0xfe,0x7b,0x00,0x00,0x00]
          vscatterqps	qword ptr [r14 + 8*xmm31 + 123] {k1}, xmm28 

// CHECK: vscatterqps	qword ptr [r9 + xmm31 + 256] {k1}, xmm28 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0xa3,0x64,0x39,0x40]
          vscatterqps	qword ptr [r9 + xmm31 + 256] {k1}, xmm28 

// CHECK: vscatterqps	qword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm28 
// CHECK: encoding: [0x62,0x22,0x7d,0x01,0xa3,0xa4,0xb9,0x00,0x04,0x00,0x00]
          vscatterqps	qword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm28 

// CHECK: vscatterqps	xmmword ptr [r14 + 8*ymm31 + 123] {k1}, xmm25 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa3,0x8c,0xfe,0x7b,0x00,0x00,0x00]
          vscatterqps	xmmword ptr [r14 + 8*ymm31 + 123] {k1}, xmm25 

// CHECK: vscatterqps	xmmword ptr [r14 + 8*ymm31 + 123] {k1}, xmm25 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa3,0x8c,0xfe,0x7b,0x00,0x00,0x00]
          vscatterqps	xmmword ptr [r14 + 8*ymm31 + 123] {k1}, xmm25 

// CHECK: vscatterqps	xmmword ptr [r9 + ymm31 + 256] {k1}, xmm25 
// CHECK: encoding: [0x62,0x02,0x7d,0x21,0xa3,0x4c,0x39,0x40]
          vscatterqps	xmmword ptr [r9 + ymm31 + 256] {k1}, xmm25 

// CHECK: vscatterqps	xmmword ptr [rcx + 4*ymm31 + 1024] {k1}, xmm25 
// CHECK: encoding: [0x62,0x22,0x7d,0x21,0xa3,0x8c,0xb9,0x00,0x04,0x00,0x00]
          vscatterqps	xmmword ptr [rcx + 4*ymm31 + 1024] {k1}, xmm25 

// CHECK: vscatterqps	qword ptr [r14 + 8*xmm31 - 123] {k1}, xmm27 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0xa3,0x9c,0xfe,0x85,0xff,0xff,0xff]
          vscatterqps	qword ptr [r14 + 8*xmm31 - 123] {k1}, xmm27 

// CHECK: vscatterqps	qword ptr [r14 + 8*xmm31 - 123] {k1}, xmm27 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0xa3,0x9c,0xfe,0x85,0xff,0xff,0xff]
          vscatterqps	qword ptr [r14 + 8*xmm31 - 123] {k1}, xmm27 

// CHECK: vscatterqps	qword ptr [r9 + xmm31 + 256] {k1}, xmm27 
// CHECK: encoding: [0x62,0x02,0x7d,0x01,0xa3,0x5c,0x39,0x40]
          vscatterqps	qword ptr [r9 + xmm31 + 256] {k1}, xmm27 

// CHECK: vscatterqps	qword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm27 
// CHECK: encoding: [0x62,0x22,0x7d,0x01,0xa3,0x9c,0xb9,0x00,0x04,0x00,0x00]
          vscatterqps	qword ptr [rcx + 4*xmm31 + 1024] {k1}, xmm27 

// CHECK: vscatterqps	xmmword ptr [r14 + 8*ymm31 - 123] {k1}, xmm23 
// CHECK: encoding: [0x62,0x82,0x7d,0x21,0xa3,0xbc,0xfe,0x85,0xff,0xff,0xff]
          vscatterqps	xmmword ptr [r14 + 8*ymm31 - 123] {k1}, xmm23 

// CHECK: vscatterqps	xmmword ptr [r14 + 8*ymm31 - 123] {k1}, xmm23 
// CHECK: encoding: [0x62,0x82,0x7d,0x21,0xa3,0xbc,0xfe,0x85,0xff,0xff,0xff]
          vscatterqps	xmmword ptr [r14 + 8*ymm31 - 123] {k1}, xmm23 

// CHECK: vscatterqps	xmmword ptr [r9 + ymm31 + 256] {k1}, xmm23 
// CHECK: encoding: [0x62,0x82,0x7d,0x21,0xa3,0x7c,0x39,0x40]
          vscatterqps	xmmword ptr [r9 + ymm31 + 256] {k1}, xmm23 

// CHECK: vscatterqps	xmmword ptr [rcx + 4*ymm31 + 1024] {k1}, xmm23 
// CHECK: encoding: [0x62,0xa2,0x7d,0x21,0xa3,0xbc,0xb9,0x00,0x04,0x00,0x00]
          vscatterqps	xmmword ptr [rcx + 4*ymm31 + 1024] {k1}, xmm23 

// CHECK: vcvtpd2ps xmm0, xmm23 
// CHECK: encoding: [0x62,0xb1,0xfd,0x08,0x5a,0xc7]
          vcvtpd2ps xmm0, xmm23

// CHECK: vcvtpd2ps xmm16, xmmword ptr [rax]
// CHECK: encoding: [0x62,0xe1,0xfd,0x08,0x5a,0x00]
          vcvtpd2ps xmm16, xmmword ptr [rax]

// CHECK: vcvtpd2ps xmm0, ymm23 
// CHECK: encoding: [0x62,0xb1,0xfd,0x28,0x5a,0xc7]
          vcvtpd2ps xmm0, ymm23

// CHECK: vcvtpd2ps xmm16, ymmword ptr [rax]
// CHECK: encoding: [0x62,0xe1,0xfd,0x28,0x5a,0x00]
          vcvtpd2ps xmm16, ymmword ptr [rax]

// CHECK: vcvtpd2dq xmm0, xmm23 
// CHECK: encoding: [0x62,0xb1,0xff,0x08,0xe6,0xc7]
          vcvtpd2dq xmm0, xmm23

// CHECK: vcvtpd2dq xmm16, xmmword ptr [rax]
// CHECK: encoding: [0x62,0xe1,0xff,0x08,0xe6,0x00]
          vcvtpd2dq xmm16, xmmword ptr [rax]

// CHECK: vcvtpd2dq xmm0, ymm23 
// CHECK: encoding: [0x62,0xb1,0xff,0x28,0xe6,0xc7]
          vcvtpd2dq xmm0, ymm23

// CHECK: vcvtpd2dq xmm16, ymmword ptr [rax]
// CHECK: encoding: [0x62,0xe1,0xff,0x28,0xe6,0x00]
          vcvtpd2dq xmm16, ymmword ptr [rax]

// CHECK: vcvtpd2udq xmm0, xmm23 
// CHECK: encoding: [0x62,0xb1,0xfc,0x08,0x79,0xc7]
          vcvtpd2udq xmm0, xmm23

// CHECK: vcvtpd2udq xmm16, xmmword ptr [rax]
// CHECK: encoding: [0x62,0xe1,0xfc,0x08,0x79,0x00]
          vcvtpd2udq xmm16, xmmword ptr [rax]

// CHECK: vcvtpd2udq xmm0, ymm23 
// CHECK: encoding: [0x62,0xb1,0xfc,0x28,0x79,0xc7]
          vcvtpd2udq xmm0, ymm23

// CHECK: vcvtpd2udq xmm16, ymmword ptr [rax]
// CHECK: encoding: [0x62,0xe1,0xfc,0x28,0x79,0x00]
          vcvtpd2udq xmm16, ymmword ptr [rax]

// CHECK: vcvttpd2dq xmm0, xmm23 
// CHECK: encoding: [0x62,0xb1,0xfd,0x08,0xe6,0xc7]
          vcvttpd2dq xmm0, xmm23

// CHECK: vcvttpd2dq xmm16, xmmword ptr [rax]
// CHECK: encoding: [0x62,0xe1,0xfd,0x08,0xe6,0x00]
          vcvttpd2dq xmm16, xmmword ptr [rax]

// CHECK: vcvttpd2dq xmm0, ymm23 
// CHECK: encoding: [0x62,0xb1,0xfd,0x28,0xe6,0xc7]
          vcvttpd2dq xmm0, ymm23

// CHECK: vcvttpd2dq xmm16, ymmword ptr [rax]
// CHECK: encoding: [0x62,0xe1,0xfd,0x28,0xe6,0x00]
          vcvttpd2dq xmm16, ymmword ptr [rax]

// CHECK: vcvttpd2udq xmm0, xmm23 
// CHECK: encoding: [0x62,0xb1,0xfc,0x08,0x78,0xc7]
          vcvttpd2udq xmm0, xmm23

// CHECK: vcvttpd2udq xmm16, xmmword ptr [rax]
// CHECK: encoding: [0x62,0xe1,0xfc,0x08,0x78,0x00]
          vcvttpd2udq xmm16, xmmword ptr [rax]

// CHECK: vcvttpd2udq xmm0, ymm23 
// CHECK: encoding: [0x62,0xb1,0xfc,0x28,0x78,0xc7]
          vcvttpd2udq xmm0, ymm23

// CHECK: vcvttpd2udq xmm16, ymmword ptr [rax]
// CHECK: encoding: [0x62,0xe1,0xfc,0x28,0x78,0x00]
          vcvttpd2udq xmm16, ymmword ptr [rax]

// CHECK: vcvtqq2ps xmm0, xmm23 
// CHECK: encoding: [0x62,0xb1,0xfc,0x08,0x5b,0xc7]
          vcvtqq2ps xmm0, xmm23

// CHECK: vcvtqq2ps xmm16, xmmword ptr [rax]
// CHECK: encoding: [0x62,0xe1,0xfc,0x08,0x5b,0x00]
          vcvtqq2ps xmm16, xmmword ptr [rax]

// CHECK: vcvtqq2ps xmm0, ymm23 
// CHECK: encoding: [0x62,0xb1,0xfc,0x28,0x5b,0xc7]
          vcvtqq2ps xmm0, ymm23

// CHECK: vcvtqq2ps xmm16, ymmword ptr [rax]
// CHECK: encoding: [0x62,0xe1,0xfc,0x28,0x5b,0x00]
          vcvtqq2ps xmm16, ymmword ptr [rax]

// CHECK: vcvtuqq2ps xmm0, xmm23 
// CHECK: encoding: [0x62,0xb1,0xff,0x08,0x7a,0xc7]
          vcvtuqq2ps xmm0, xmm23

// CHECK: vcvtuqq2ps xmm16, xmmword ptr [rax]
// CHECK: encoding: [0x62,0xe1,0xff,0x08,0x7a,0x00]
          vcvtuqq2ps xmm16, xmmword ptr [rax]

// CHECK: vcvtuqq2ps xmm0, ymm23 
// CHECK: encoding: [0x62,0xb1,0xff,0x28,0x7a,0xc7]
          vcvtuqq2ps xmm0, ymm23

// CHECK: vcvtuqq2ps xmm16, ymmword ptr [rax]
// CHECK: encoding: [0x62,0xe1,0xff,0x28,0x7a,0x00]
          vcvtuqq2ps xmm16, ymmword ptr [rax]

// CHECK: vcvtps2pd xmm1 {k2} {z}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7c,0x8a,0x5a,0x49,0x10]
          vcvtps2pd xmm1 {k2} {z}, qword ptr [rcx+0x80]

// CHECK: vcvtps2pd xmm1 {k2}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7c,0x0a,0x5a,0x49,0x10]
          vcvtps2pd xmm1 {k2}, qword ptr [rcx+0x80]

// CHECK: vcvtudq2pd xmm2 {k2} {z}, qword ptr [rcx + 128]
// CHECK: encoding: [0x62,0xf1,0x7e,0x8a,0x7a,0x51,0x10]
          vcvtudq2pd xmm2 {k2} {z}, qword ptr [rcx+0x80]

// CHECK: vcvtudq2pd xmm2 {k2}, qword ptr [rcx + 128]
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x7a,0x51,0x10]
          vcvtudq2pd xmm2 {k2}, qword ptr [rcx+0x80]

// CHECK: vcvtudq2pd xmm2, qword ptr [rcx + 128]
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x7a,0x51,0x10]
          vcvtudq2pd xmm2, qword ptr [rcx+0x80]

// CHECK: vcvtdq2pd xmm2 {k1}, qword ptr [rcx]
// CHECK: encoding: [0x62,0xf1,0x7e,0x09,0xe6,0x11]
          vcvtdq2pd xmm2 {k1}, qword ptr [rcx]

// CHECK: vcvtdq2pd xmm2 {k1} {z}, qword ptr [rcx]
// CHECK: encoding: [0x62,0xf1,0x7e,0x89,0xe6,0x11]
          vcvtdq2pd xmm2 {k1} {z}, qword ptr [rcx]

// CHECK: vextractps ecx, xmm17, 1
// CHECK: encoding: [0x62,0xe3,0x7d,0x08,0x17,0xc9,0x01]
          vextractps rcx, xmm17, 1
