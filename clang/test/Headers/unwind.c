// RUN: %clang -target arm-unknown-linux-gnueabi -ffreestanding -fsyntax-only %s
// RUN: %clang -target i686-unknown-linux -ffreestanding -fsyntax-only %s
// RUN: %clang -ffreestanding -fsyntax-only -x c++ %s

#include "unwind.h"
