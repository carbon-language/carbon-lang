//RUN: llvm-mc  -triple=aarch64-linux-gnu %s | FileCheck %s

//
// Check that large constants are converted to ldr from constant pool
//
// simple test
.section a, "ax", @progbits
// CHECK-LABEL: f1:
f1:
  ldr x0, =0x1234
// CHECK: movz    x0, #0x1234
  ldr w1, =0x4567
// CHECK:  movz    w1, #0x4567
  ldr x0, =0x12340000
// CHECK:  movz    x0, #0x1234, lsl #16
  ldr w1, =0x45670000
// CHECK: movz    w1, #0x4567, lsl #16
  ldr x0, =0xabc00000000
// CHECK: movz    x0, #0xabc, lsl #32
  ldr x0, =0xbeef000000000000
// CHECK: movz    x0, #0xbeef, lsl #48

.section b,"ax",@progbits
// CHECK-LABEL: f3:
f3:
  ldr x0, =0x10001
// CHECK: ldr x0, .Ltmp[[TMP0:[0-9]+]]

// loading multiple constants
.section c,"ax",@progbits
// CHECK-LABEL: f4:
f4:
  ldr x0, =0x10002
// CHECK: ldr x0, .Ltmp[[TMP1:[0-9]+]]
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  ldr x0, =0x10003
// CHECK: ldr x0, .Ltmp[[TMP2:[0-9]+]]
  adds x0, x0, #1
  adds x0, x0, #1

// TODO: the same constants should have the same constant pool location
.section d,"ax",@progbits
// CHECK-LABEL: f5:
f5:
  ldr x0, =0x10004
// CHECK: ldr x0, .Ltmp[[TMP3:[0-9]+]]
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  ldr x0, =0x10004
// CHECK: ldr x0, .Ltmp[[TMP4:[0-9]+]]
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1

// a section defined in multiple pieces should be merged and use a single constant pool
.section e,"ax",@progbits
// CHECK-LABEL: f6:
f6:
  ldr x0, =0x10006
// CHECK: ldr x0, .Ltmp[[TMP5:[0-9]+]]
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1

.section f, "ax", @progbits
// CHECK-LABEL: f7:
f7:
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1

.section e, "ax", @progbits
// CHECK-LABEL: f8:
f8:
  adds x0, x0, #1
  ldr x0, =0x10007
// CHECK: ldr x0, .Ltmp[[TMP6:[0-9]+]]
  adds x0, x0, #1
  adds x0, x0, #1

//
// Check that symbols can be loaded using ldr pseudo
//

// load an undefined symbol
.section g,"ax",@progbits
// CHECK-LABEL: f9:
f9:
  ldr x0, =foo
// CHECK: ldr x0, .Ltmp[[TMP7:[0-9]+]]

// load a symbol from another section
.section h,"ax",@progbits
// CHECK-LABEL: f10:
f10:
  ldr x0, =f5
// CHECK: ldr x0, .Ltmp[[TMP8:[0-9]+]]

// load a symbol from the same section
.section i,"ax",@progbits
// CHECK-LABEL: f11:
f11:
  ldr x0, =f12
// CHECK: ldr x0, .Ltmp[[TMP9:[0-9]+]]
  ldr w0,=0x3C000
// CHECK: ldr     w0, .Ltmp[[TMP10:[0-9]+]]

// CHECK-LABEL: f12:
f12:
  adds x0, x0, #1
  adds x0, x0, #1

.section j,"ax",@progbits
// mix of symbols and constants
// CHECK-LABEL: f13:
f13:
  adds x0, x0, #1
  adds x0, x0, #1
  ldr x0, =0x101
// CHECK: movz x0, #0x101
  adds x0, x0, #1
  adds x0, x0, #1
  ldr x0, =bar
// CHECK: ldr x0, .Ltmp[[TMP11:[0-9]+]]
  adds x0, x0, #1
  adds x0, x0, #1
//
// Check for correct usage in other contexts
//

// usage in macro
.macro useit_in_a_macro
  ldr x0, =0x10008
  ldr x0, =baz
.endm
.section k,"ax",@progbits
// CHECK-LABEL: f14:
f14:
  useit_in_a_macro
// CHECK: ldr x0, .Ltmp[[TMP12:[0-9]+]]
// CHECK: ldr x0, .Ltmp[[TMP13:[0-9]+]]

// usage with expressions
.section l, "ax", @progbits
// CHECK-LABEL: f15:
f15:
  ldr x0, =0x10001+8
// CHECK: ldr x0, .Ltmp[[TMP14:[0-9]+]]
  adds x0, x0, #1
  ldr x0, =bar+4
// CHECK: ldr x0, .Ltmp[[TMP15:[0-9]+]]
  adds x0, x0, #1

//
// Constant Pools
//
// CHECK: .section b,"ax",@progbits
// CHECK: .align 2
// CHECK: .Ltmp[[TMP0]]
// CHECK: .word 65537

// CHECK: .section c,"ax",@progbits
// CHECK: .align 2
// CHECK: .Ltmp[[TMP1]]
// CHECK: .word 65538
// CHECK: .Ltmp[[TMP2]]
// CHECK: .word 65539

// CHECK: .section d,"ax",@progbits
// CHECK: .align 2
// CHECK: .Ltmp[[TMP3]]
// CHECK: .word 65540
// CHECK: .Ltmp[[TMP4]]
// CHECK: .word 65540

// CHECK: .section e,"ax",@progbits
// CHECK: .align 2
// CHECK: .Ltmp[[TMP5]]
// CHECK: .word 65542
// CHECK: .Ltmp[[TMP6]]
// CHECK: .word 65543

// Should not switch to section because it has no constant pool
// CHECK-NOT: .section f,"ax",@progbits

// CHECK: .section g,"ax",@progbits
// CHECK: .align 2
// CHECK: .Ltmp[[TMP7]]
// CHECK: .word foo

// CHECK: .section h,"ax",@progbits
// CHECK: .align 2
// CHECK: .Ltmp[[TMP8]]
// CHECK: .word f5

// CHECK: .section i,"ax",@progbits
// CHECK: .align 2
// CHECK: .Ltmp[[TMP9]]
// CHECK: .word f12
// CHECK: .Ltmp[[TMP10]]
// CHECK: .word 245760

// CHECK: .section j,"ax",@progbits
// CHECK: .align 2
// CHECK: .Ltmp[[TMP11]]
// CHECK: .word bar

// CHECK: .section k,"ax",@progbits
// CHECK: .align 2
// CHECK: .Ltmp[[TMP12]]
// CHECK: .word 65544
// CHECK: .Ltmp[[TMP13]]
// CHECK: .word baz

// CHECK: .section l,"ax",@progbits
// CHECK: .align 2
// CHECK: .Ltmp[[TMP14]]
// CHECK: .word 65545
// CHECK: .Ltmp[[TMP15]]
// CHECK: .word bar+4
