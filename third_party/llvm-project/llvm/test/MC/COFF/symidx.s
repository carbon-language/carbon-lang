// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-objdump -s -t - | FileCheck %s
.text
foo:
  ret
bar:
  ret
.data
.symidx	bar
.symidx	foo

// CHECK:      SYMBOL TABLE:
// CHECK:      [ [[FOO:[1-9]]]](sec  1)(fl 0x00)(ty   0)(scl   3) (nx 0) 0x00000000 foo
// CHECK-NEXT: [ [[BAR:[1-9]]]](sec  1)(fl 0x00)(ty   0)(scl   3) (nx 0) 0x00000001 bar
// CHECK:      Contents of section .data:
// CHECK-NEXT:  0000 0[[BAR]]000000 0[[FOO]]000000
