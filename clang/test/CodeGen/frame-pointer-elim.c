// RUN: %clang -ccc-host-triple i386 -S -o - %s | \
// RUN:   FileCheck --check-prefix=DEFAULT %s
// DEFAULT: f0:
// DEFAULT: pushl %ebp
// DEFAULT: ret
// DEFAULT: f1:
// DEFAULT: pushl %ebp
// DEFAULT: ret

// RUN: %clang -ccc-host-triple i386 -S -o - -fomit-frame-pointer %s | \
// RUN:   FileCheck --check-prefix=OMIT_ALL %s
// OMIT_ALL: f0:
// OMIT_ALL-NOT: pushl %ebp
// OMIT_ALL: ret
// OMIT_ALL: f1:
// OMIT_ALL-NOT: pushl %ebp
// OMIT_ALL: ret

// RUN: %clang -ccc-host-triple i386 -S -o - -momit-leaf-frame-pointer %s | \
// RUN:   FileCheck --check-prefix=OMIT_LEAF %s
// OMIT_LEAF: f0:
// OMIT_LEAF-NOT: pushl %ebp
// OMIT_LEAF: ret
// OMIT_LEAF: f1:
// OMIT_LEAF: pushl %ebp
// OMIT_LEAF: ret

void f0() {}
void f1() { f0(); }
