// This tests that default-null weak symbols (a GNU extension) are created
// properly via the .weak directive.

// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | llvm-readobj -t | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s | llvm-readobj -t | FileCheck %s

    .def    _main;
    .scl    2;
    .type   32;
    .endef
    .text
    .globl  _main
    .align  16, 0x90
_main:                                  # @main
# BB#0:                                 # %entry
    subl    $4, %esp
    movl    $_test_weak, %eax
    testl   %eax, %eax
    je      LBB0_2
# BB#1:                                 # %if.then
    call    _test_weak
    movl    $1, %eax
    addl    $4, %esp
    ret
LBB0_2:                                 # %return
    xorl    %eax, %eax
    addl    $4, %esp
    ret

    .weak   _test_weak

// CHECK: Symbols [

// CHECK:      Symbol {
// CHECK:        Name:           _test_weak
// CHECK-NEXT:   Value:          0
// CHECK-NEXT:   Section:        (0)
// CHECK-NEXT:   BaseType:       Null
// CHECK-NEXT:   ComplexType:    Null
// CHECK-NEXT:   StorageClass:   WeakExternal
// CHECK-NEXT:   AuxSymbolCount: 1
// CHECK-NEXT:   AuxWeakExternal {
// CHECK-NEXT:     Linked: .weak._test_weak.default
// CHECK-NEXT:      Search: Library
// CHECK-NEXT:      Unused: (00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK:      Symbol {
// CHECK:        Name:                .weak._test_weak.default
// CHECK-NEXT:   Value:               0
// CHECK-NEXT:   Section:             (-1)
// CHECK-NEXT:   BaseType:            Null
// CHECK-NEXT:   ComplexType:         Null
// CHECK-NEXT:   StorageClass:        External
// CHECK-NEXT:   AuxSymbolCount:      0
// CHECK-NEXT: }
