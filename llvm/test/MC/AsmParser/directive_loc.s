# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s -check-prefix=FILE
# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s -check-prefix=BASIC-LOC-1
# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s -check-prefix=BASIC-LOC-2
# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s -check-prefix=DISCRIMINATOR
# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s -check-prefix=ISA

        .file 1 "hello"
# FILE: .file 1 "hello"
        .loc 1
        .loc 1 2
# BASIC-LOC-1: .loc 1 2 0
        .loc 1 2 3
# BASIC-LOC-2: .loc 1 2 3
        .loc 1 2 discriminator 1
# DISCRIMINATOR: 1 2 0 discriminator 1
        .loc 1 2 0 isa 3
# ISA: 1 2 0 isa 3
        .loc 1 0
