// RUN: %clang -target arm-unknown-linux-gnueabi \
// RUN:   -isystem %S/Inputs/include -ffreestanding -fsyntax-only %s
// RUN: %clang -target mips-unknown-linux \
// RUN:   -isystem %S/Inputs/include -ffreestanding -fsyntax-only %s
// RUN: %clang -target i686-unknown-linux \
// RUN:   -isystem %S/Inputs/include -ffreestanding -fsyntax-only %s
// RUN: %clang -target x86_64-unknown-linux \
// RUN:   -isystem %S/Inputs/include -ffreestanding -fsyntax-only %s

// RUN: %clang -target arm-unknown-linux-gnueabi \
// RUN:   -isystem %S/Inputs/include -ffreestanding -fsyntax-only -x c++ %s
// RUN: %clang -target mips-unknown-linux \
// RUN:   -isystem %S/Inputs/include -ffreestanding -fsyntax-only -x c++ %s
// RUN: %clang -target i686-unknown-linux \
// RUN:   -isystem %S/Inputs/include -ffreestanding -fsyntax-only -x c++ %s
// RUN: %clang -target x86_64-unknown-linux \
// RUN:   -isystem %S/Inputs/include -ffreestanding -fsyntax-only -x c++ %s

#include "unwind.h"
