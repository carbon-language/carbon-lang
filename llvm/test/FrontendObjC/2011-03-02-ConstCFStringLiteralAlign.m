// RUN: %llvmgcc -S -w -m64 -mmacosx-version-min=10.5 %s -o - | \
// RUN:     llc --disable-fp-elim -o - | FileCheck %s
// XFAIL: *
// XTARGET: darwin

@interface Foo
@end
Foo *FooName = @"FooBar";

// CHECK:      .section __TEXT,__cstring,cstring_literals
// CHECK-NEXT: L_.str:
