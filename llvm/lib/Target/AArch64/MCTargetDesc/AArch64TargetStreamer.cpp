//===- AArch64TargetStreamer.cpp - AArch64TargetStreamer class --*- C++ -*---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64TargetStreamer class.
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/MapVector.h"
#include "llvm/MC/ConstantPools.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

//
// AArch64TargetStreamer Implemenation
//
AArch64TargetStreamer::AArch64TargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S), ConstantPools(new AssemblerConstantPools()) {}

AArch64TargetStreamer::~AArch64TargetStreamer() {}

// The constant pool handling is shared by all AArch64TargetStreamer
// implementations.
const MCExpr *AArch64TargetStreamer::addConstantPoolEntry(const MCExpr *Expr) {
  return ConstantPools->addEntry(Streamer, Expr);
}

void AArch64TargetStreamer::emitCurrentConstantPool() {
  ConstantPools->emitForCurrentSection(Streamer);
}

// finish() - write out any non-empty assembler constant pools.
void AArch64TargetStreamer::finish() { ConstantPools->emitAll(Streamer); }
