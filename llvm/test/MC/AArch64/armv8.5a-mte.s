// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+mte   < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+v8.5a < %s 2>&1 | FileCheck %s --check-prefix=NOMTE
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=-mte   < %s 2>&1 | FileCheck %s --check-prefix=NOMTE

dc igvac, x0
dc igsw, x1
dc cgsw, x2
dc cigsw, x3
dc cgvac, x4
dc cgvap, x5
dc cgvadp, x6
dc cigvac, x7
dc gva, x8
dc igdvac, x9
dc igdsw, x10
dc cgdsw, x11
dc cigdsw, x12
dc cgdvac, x13
dc cgdvap, x14
dc cgdvadp, x15
dc cigdvac, x16
dc gzva, x17

// CHECK: dc igvac, x0          // encoding: [0x60,0x76,0x08,0xd5]
// CHECK: dc igsw, x1           // encoding: [0x81,0x76,0x08,0xd5]
// CHECK: dc cgsw, x2           // encoding: [0x82,0x7a,0x08,0xd5]
// CHECK: dc cigsw, x3          // encoding: [0x83,0x7e,0x08,0xd5]
// CHECK: dc cgvac, x4          // encoding: [0x64,0x7a,0x0b,0xd5]
// CHECK: dc cgvap, x5          // encoding: [0x65,0x7c,0x0b,0xd5]
// CHECK: dc cgvadp, x6         // encoding: [0x66,0x7d,0x0b,0xd5]
// CHECK: dc cigvac, x7         // encoding: [0x67,0x7e,0x0b,0xd5]
// CHECK: dc gva, x8            // encoding: [0x68,0x74,0x0b,0xd5]
// CHECK: dc igdvac, x9         // encoding: [0xa9,0x76,0x08,0xd5]
// CHECK: dc igdsw, x10         // encoding: [0xca,0x76,0x08,0xd5]
// CHECK: dc cgdsw, x11         // encoding: [0xcb,0x7a,0x08,0xd5]
// CHECK: dc cigdsw, x12        // encoding: [0xcc,0x7e,0x08,0xd5]
// CHECK: dc cgdvac, x13        // encoding: [0xad,0x7a,0x0b,0xd5]
// CHECK: dc cgdvap, x14        // encoding: [0xae,0x7c,0x0b,0xd5]
// CHECK: dc cgdvadp, x15       // encoding: [0xaf,0x7d,0x0b,0xd5]
// CHECK: dc cigdvac, x16       // encoding: [0xb0,0x7e,0x0b,0xd5]
// CHECK: dc gzva, x17          // encoding: [0x91,0x74,0x0b,0xd5]

// NOMTE: DC IGVAC requires mte
// NOMTE: DC IGSW requires mte
// NOMTE: DC CGSW requires mte
// NOMTE: DC CIGSW requires mte
// NOMTE: DC CGVAC requires mte
// NOMTE: DC CGVAP requires mte
// NOMTE: DC CGVADP requires mte
// NOMTE: DC CIGVAC requires mte
// NOMTE: DC GVA requires mte
// NOMTE: DC IGDVAC requires mte
// NOMTE: DC IGDSW requires mte
// NOMTE: DC CGDSW requires mte
// NOMTE: DC CIGDSW requires mte
// NOMTE: DC CGDVAC requires mte
// NOMTE: DC CGDVAP requires mte
// NOMTE: DC CGDVADP requires mte
// NOMTE: DC CIGDVAC requires mte
// NOMTE: DC GZVA requires mte
