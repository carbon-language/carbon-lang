// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: ld.lld2 %t -o %tout
// RUN: llvm-objdump -d %tout | FileCheck -check-prefix=DISASM %s
// RUN: llvm-readobj -symbols %tout | FileCheck -check-prefix=SYMBOL %s

// DISASM: _start:
// DISASM:    11000:       e8 0a 00 00 00  callq   10
// DISASM:    11005:       e8 08 00 00 00  callq   8
// DISASM:    1100a:       e8 03 00 00 00  callq   3
// DISASM: Disassembly of section foo:
// DISASM: __start_foo:
// DISASM:    1100f:       90      nop
// DISASM:    11010:       90      nop
// DISASM:    11011:       90      nop
// DISASM: Disassembly of section bar:
// DISASM: __start_bar:
// DISASM:    11012:       90      nop
// DISASM:    11013:       90      nop
// DISASM:    11014:       90      nop

// SYMBOL: Symbol {
// SYMBOL:   Name: __start_bar
// SYMBOL:   Value: 0x11012
// SYMBOL:   Section: bar
// SYMBOL: }
// SYMBOL-NOT:   Section: __stop_bar
// SYMBOL: Symbol {
// SYMBOL:   Name: __start_foo
// SYMBOL:   Value: 0x1100F
// SYMBOL:   Section: foo
// SYMBOL: }
// SYMBOL: Symbol {
// SYMBOL:   Name: __stop_foo
// SYMBOL:   Value: 0x11012
// SYMBOL:   Section: foo (0x2)
// SYMBOL: }

.global _start
.text
_start:
	call __start_foo
	call __stop_foo
	call __start_bar

.section foo,"ax"
	nop
	nop
	nop

.section bar,"ax"
	nop
	nop
	nop
