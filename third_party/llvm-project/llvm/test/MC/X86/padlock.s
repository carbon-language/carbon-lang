// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

/// Check xstore does not get an implicit REP prefix but all other PadLock
/// instructions do.

	xstore
// CHECK: xstore
// CHECK: encoding: [0x0f,0xa7,0xc0]
	xcryptecb
// CHECK: xcryptecb
// CHECK: encoding: [0xf3,0x0f,0xa7,0xc8]
	xcryptcbc
// CHECK: xcryptcbc
// CHECK: encoding: [0xf3,0x0f,0xa7,0xd0]
	xcryptctr
// CHECK: xcryptctr
// CHECK: encoding: [0xf3,0x0f,0xa7,0xd8]
	xcryptcfb
// CHECK: xcryptcfb
// CHECK: encoding: [0xf3,0x0f,0xa7,0xe0]
	xcryptofb
// CHECK: xcryptofb
// CHECK: encoding: [0xf3,0x0f,0xa7,0xe8]
	xsha1
// CHECK: xsha1
// CHECK: encoding: [0xf3,0x0f,0xa6,0xc8]
	xsha256
// CHECK: xsha256
// CHECK: encoding: [0xf3,0x0f,0xa6,0xd0]
	montmul
// CHECK: montmul
// CHECK: encoding: [0xf3,0x0f,0xa6,0xc0]
