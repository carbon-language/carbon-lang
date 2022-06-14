// This tests that default-null weak symbols (a GNU extension) are created
// properly via the .weak directive.

// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | llvm-readobj --symbols - | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s | llvm-readobj --symbols - | FileCheck %s

    .def    _main;
    .scl    2;
    .type   32;
    .endef
    .text
    .globl  _main
    .align  16, 0x90
_main:                                  # @main
# %bb.0:                                # %entry
    subl    $4, %esp
    movl    $_test_weak, %eax
    testl   %eax, %eax
    je      LBB0_2
# %bb.1:                                # %if.then
    call    _test_weak
    movl    $1, %eax
    addl    $4, %esp
    ret
LBB0_2:                                 # %return
    xorl    %eax, %eax
    addl    $4, %esp
    ret

    .weak   _test_weak

    .weak   _test_weak_alias
    _test_weak_alias=_main

    .weak weakfunc
    .globl weakfunc
weakfunc:
    ret

// CHECK: Symbols [

// CHECK:      Symbol {
// CHECK:        Name: _main
// CHECK-NEXT:   Value: 0
// CHECK-NEXT:   Section: .text
// CHECK-NEXT:   BaseType: Null
// CHECK-NEXT:   ComplexType: Function
// CHECK-NEXT:   StorageClass: External
// CHECK-NEXT:   AuxSymbolCount: 0
// CHECK-NEXT: }

// CHECK:      Symbol {
// CHECK:        Name:           _test_weak
// CHECK-NEXT:   Value:          0
// CHECK-NEXT:   Section:        IMAGE_SYM_UNDEFINED
// CHECK-NEXT:   BaseType:       Null
// CHECK-NEXT:   ComplexType:    Null
// CHECK-NEXT:   StorageClass:   WeakExternal
// CHECK-NEXT:   AuxSymbolCount: 1
// CHECK-NEXT:   AuxWeakExternal {
// CHECK-NEXT:     Linked: .weak._test_weak.default
// CHECK-NEXT:      Search: Alias
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK:      Symbol {
// CHECK:        Name:                .weak._test_weak.default._main
// CHECK-NEXT:   Value:               0
// CHECK-NEXT:   Section:             IMAGE_SYM_ABSOLUTE
// CHECK-NEXT:   BaseType:            Null
// CHECK-NEXT:   ComplexType:         Null
// CHECK-NEXT:   StorageClass:        External
// CHECK-NEXT:   AuxSymbolCount:      0
// CHECK-NEXT: }

// CHECK:      Symbol {
// CHECK:        Name:           _test_weak_alias
// CHECK-NEXT:   Value:          0
// CHECK-NEXT:   Section:        IMAGE_SYM_UNDEFINED
// CHECK-NEXT:   BaseType:       Null
// CHECK-NEXT:   ComplexType:    Null
// CHECK-NEXT:   StorageClass:   WeakExternal
// CHECK-NEXT:   AuxSymbolCount: 1
// CHECK-NEXT:   AuxWeakExternal {
// CHECK-NEXT:     Linked: _main
// CHECK-NEXT:     Search: Alias
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK:      Symbol {
// CHECK:        Name:           weakfunc
// CHECK-NEXT:   Value:          0
// CHECK-NEXT:   Section:        IMAGE_SYM_UNDEFINED
// CHECK-NEXT:   BaseType:       Null
// CHECK-NEXT:   ComplexType:    Null
// CHECK-NEXT:   StorageClass:   WeakExternal
// CHECK-NEXT:   AuxSymbolCount: 1
// CHECK-NEXT:   AuxWeakExternal {
// CHECK-NEXT:     Linked: .weak.weakfunc.default
// CHECK-NEXT:     Search: Alias
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK:      Symbol {
// CHECK:        Name:           .weak.weakfunc.default
// CHECK-NOT:    Value:          0
// CHECK-NOT:  Symbol {
// CHECK:        Section:        .text
// CHECK-NEXT:   BaseType:       Null
// CHECK-NEXT:   ComplexType:    Null
// CHECK-NEXT:   StorageClass:   External
// CHECK-NEXT:   AuxSymbolCount: 0
// CHECK-NEXT: }
