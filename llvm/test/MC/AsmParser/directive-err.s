// RUN: not llvm-mc -triple i386 %s 2>&1 | FileCheck %s

	.err
// CHECK: error: .err encountered
// CHECK-NEXT: 	.err
// CHECK-NEXT:  ^

	.ifc a,a
		.err
	.endif
// CHECK: error: .err encountered
// CHECK-NEXT:		.err
// CHECK-NEXT:          ^

	.ifnc a,a
		.err
	.endif
// CHECK-NOT: error: .err encountered

