// For these next two tests when optimized we should omit the leaf frame
// pointer, for unoptimized we should have a leaf frame pointer.
// RUN: %clang -### -target i386-pc-linux-gnu -S -O1 %s 2>&1 | \
// RUN:   FileCheck --check-prefix=LINUX-OPT %s
// LINUX-OPT: "-momit-leaf-frame-pointer"

// RUN: %clang -### -target i386-pc-linux-gnu -S %s 2>&1 | \
// RUN:   FileCheck --check-prefix=LINUX %s
// LINUX-NOT: "-momit-leaf-frame-pointer"

// Darwin disables omitting the leaf frame pointer even under optimization
// unless the command lines are given.
// RUN: %clang -### -target i386-apple-darwin -S %s 2>&1 | \
// RUN:   FileCheck --check-prefix=DARWIN %s
// DARWIN: "-mdisable-fp-elim"

// RUN: %clang -### -target i386-apple-darwin -S -O1 %s 2>&1 | \
// RUN:   FileCheck --check-prefix=DARWIN-OPT %s
// DARWIN-OPT-NOT: "-momit-leaf-frame-pointer"

// RUN: %clang -### -target i386-darwin -S -fomit-frame-pointer %s 2>&1 | \
// RUN:   FileCheck --check-prefix=OMIT_ALL %s
// OMIT_ALL-NOT: "-mdisable-fp-elim"

// RUN: %clang -### -target i386-darwin -S -momit-leaf-frame-pointer %s 2>&1 | \
// RUN:   FileCheck --check-prefix=OMIT_LEAF %s
// OMIT_LEAF: "-momit-leaf-frame-pointer"

void f0() {}
void f1() { f0(); }
