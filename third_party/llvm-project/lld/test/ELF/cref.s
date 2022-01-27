// REQUIRES: x86

// RUN: echo '.global foo; foo:' | llvm-mc -filetype=obj -triple=x86_64-pc-linux - -o %t1.o
// RUN: echo '.global foo, bar; bar:' | llvm-mc -filetype=obj -triple=x86_64-pc-linux - -o %t2.o
// RUN: echo '.global zed; zed:' | llvm-mc -filetype=obj -triple=x86_64-pc-linux - -o %ta.o
// RUN: rm -f %t.a
// RUN: llvm-ar rcs %t.a %ta.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t3.o
// RUN: ld.lld -shared -o %t1.so %t1.o
// RUN: ld.lld -o /dev/null %t1.so %t2.o %t3.o %t.a --gc-sections --cref | FileCheck -strict-whitespace %s

/// If -Map is specified, print to the map file.
// RUN: ld.lld -o /dev/null %t1.so %t2.o %t3.o %t.a --gc-sections -Map=%t.map --cref
// RUN: FileCheck --input-file=%t.map %s --check-prefix=CHECK2

// CHECK:      {{^$}}
// CHECK-NEXT: Cross Reference Table
// CHECK-EMPTY:
// CHECK-NEXT: Symbol                                            File
// CHECK-NEXT: foo                                               {{.*}}1.so
// CHECK-NEXT:                                                   {{.*}}2.o
// CHECK-NEXT:                                                   {{.*}}3.o
// CHECK-NEXT: bar                                               {{.*}}2.o
// CHECK-NEXT:                                                   {{.*}}3.o
// CHECK-NEXT: _start                                            {{.*}}3.o
// CHECK-NEXT: baz                                               {{.*}}3.o
// CHECK-NEXT: zed                                               {{.*}}.a({{.*}}a.o)
// CHECK-NEXT:                                                   {{.*}}3.o
// CHECK-NOT:  discarded

// CHECK2:         VMA              LMA     Size Align Out     In      Symbol
// CHECK2:      .strtab
// CHECK2-NEXT:         <internal>:(.strtab)

/// There is a blank line before the "Cross Reference Table" header.
// CHECK2-EMPTY:
// CHECK2-NEXT: Cross Reference Table

.global _start, foo, bar, baz, discarded
_start:
  call foo
  call bar
  call zed
localsym:
baz:

.section .text.a,"ax",@progbits
discarded:
