//===- ARMTargetStreamer.cpp - ARMTargetStreamer class --*- C++ -*---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARMTargetStreamer class.
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/MapVector.h"
#include "llvm/MC/ConstantPools.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;
//
// ARMTargetStreamer Implemenation
//
ARMTargetStreamer::ARMTargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S), ConstantPools(new AssemblerConstantPools()) {}

ARMTargetStreamer::~ARMTargetStreamer() {}

// The constant pool handling is shared by all ARMTargetStreamer
// implementations.
const MCExpr *ARMTargetStreamer::addConstantPoolEntry(const MCExpr *Expr) {
  return ConstantPools->addEntry(Streamer, Expr);
}

void ARMTargetStreamer::emitCurrentConstantPool() {
  ConstantPools->emitForCurrentSection(Streamer);
}

// finish() - write out any non-empty assembler constant pools.
void ARMTargetStreamer::finish() { ConstantPools->emitAll(Streamer); }

// The remaining callbacks should be handled separately by each
// streamer.
void ARMTargetStreamer::emitFnStart() {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitFnEnd() {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitCantUnwind() {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitPersonality(const MCSymbol *Personality) {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitPersonalityIndex(unsigned Index) {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitHandlerData() {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitSetFP(unsigned FpReg, unsigned SpReg,
                                       int64_t Offset) {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitMovSP(unsigned Reg, int64_t Offset) {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitPad(int64_t Offset) {
  llvm_unreachable("unimplemented");
}
void
ARMTargetStreamer::emitRegSave(const SmallVectorImpl<unsigned> &RegList,
                                    bool isVector) {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitUnwindRaw(
    int64_t StackOffset, const SmallVectorImpl<uint8_t> &Opcodes) {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::switchVendor(StringRef Vendor) {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitAttribute(unsigned Attribute, unsigned Value) {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitTextAttribute(unsigned Attribute,
                                               StringRef String) {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitIntTextAttribute(unsigned Attribute,
                                                  unsigned IntValue,
                                                  StringRef StringValue) {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitArch(unsigned Arch) {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitObjectArch(unsigned Arch) {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitFPU(unsigned FPU) {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::finishAttributeSection() {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::emitInst(uint32_t Inst, char Suffix) {
  llvm_unreachable("unimplemented");
}
void ARMTargetStreamer::AnnotateTLSDescriptorSequence(
    const MCSymbolRefExpr *SRE) {
  llvm_unreachable("unimplemented");
}

void ARMTargetStreamer::emitThumbSet(MCSymbol *Symbol, const MCExpr *Value) {
  llvm_unreachable("unimplemented");
}
