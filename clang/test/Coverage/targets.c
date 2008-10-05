// RUN: clang -g -triple i686-unknown-unknown -emit-llvm -o %t %s &&
// RUN: clang -g -triple i686-pc-linux-gnu -emit-llvm -o %t %s &&
// RUN: clang -g -triple i686-unknown-dragonfly -emit-llvm -o %t %s &&
// RUN: clang -g -triple i686-unknown-win32 -emit-llvm -o %t %s &&
// RUN: clang -g -triple i686-apple-darwin9 -emit-llvm -o %t %s &&
// RUN: clang -g -triple x86_64-unknown-unknown -emit-llvm -o %t %s &&
// RUN: clang -g -triple x86_64-pc-linux-gnu -emit-llvm -o %t %s &&
// RUN: clang -g -triple x86_64-apple-darwin9 -emit-llvm -o %t %s &&
// RUN: clang -g -triple ppc-unknown-unknown -emit-llvm -o %t %s &&
// RUN: clang -g -triple ppc-apple-darwin9 -emit-llvm -o %t %s &&
// RUN: clang -g -triple ppc64-unknown-unknown -emit-llvm -o %t %s &&
// RUN: clang -g -triple ppc64-apple-darwin9 -emit-llvm -o %t %s &&
// RUN: clang -g -triple armv6-unknown-unknown -emit-llvm -o %t %s &&
// RUN: clang -g -triple armv6-apple-darwin9 -emit-llvm -o %t %s &&
// RUN: clang -g -triple sparc-unknown-unknown -emit-llvm -o %t %s &&
// RUN: clang -g -triple sparc-unknown-solaris -emit-llvm -o %t %s &&
// RUN: clang -g -triple pic16-unknown-unknown -emit-llvm -o %t %s &&
// RUN: true
