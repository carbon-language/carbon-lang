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
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

namespace {
// A class to keep track of assembler-generated constant pools that are use to
// implement the ldr-pseudo.
class ConstantPool {
  typedef SmallVector<std::pair<MCSymbol *, const MCExpr *>, 4> EntryVecTy;
  EntryVecTy Entries;

public:
  // Initialize a new empty constant pool
  ConstantPool() {}

  // Add a new entry to the constant pool in the next slot.
  // \param Value is the new entry to put in the constant pool.
  //
  // \returns a MCExpr that references the newly inserted value
  const MCExpr *addEntry(const MCExpr *Value, MCContext &Context);

  // Emit the contents of the constant pool using the provided streamer.
  void emitEntries(MCStreamer &Streamer);

  // Return true if the constant pool is empty
  bool empty();
};
}

namespace llvm {
class AssemblerConstantPools {
  // Map type used to keep track of per-Section constant pools used by the
  // ldr-pseudo opcode. The map associates a section to its constant pool. The
  // constant pool is a vector of (label, value) pairs. When the ldr
  // pseudo is parsed we insert a new (label, value) pair into the constant pool
  // for the current section and add MCSymbolRefExpr to the new label as
  // an opcode to the ldr. After we have parsed all the user input we
  // output the (label, value) pairs in each constant pool at the end of the
  // section.
  //
  // We use the MapVector for the map type to ensure stable iteration of
  // the sections at the end of the parse. We need to iterate over the
  // sections in a stable order to ensure that we have print the
  // constant pools in a deterministic order when printing an assembly
  // file.
  typedef MapVector<const MCSection *, ConstantPool> ConstantPoolMapTy;
  ConstantPoolMapTy ConstantPools;

public:
  AssemblerConstantPools() {}
  ~AssemblerConstantPools() {}

  void emitAll(MCStreamer &Streamer);
  void emitForCurrentSection(MCStreamer &Streamer);
  const MCExpr *addEntry(MCStreamer &Streamer, const MCExpr *Expr);

private:
  ConstantPool *getConstantPool(const MCSection *Section);
  ConstantPool &getOrCreateConstantPool(const MCSection *Section);
};
}

//
// ConstantPool implementation
//
// Emit the contents of the constant pool using the provided streamer.
void ConstantPool::emitEntries(MCStreamer &Streamer) {
  if (Entries.empty())
    return;
  Streamer.EmitCodeAlignment(4); // align to 4-byte address
  Streamer.EmitDataRegion(MCDR_DataRegion);
  for (EntryVecTy::const_iterator I = Entries.begin(), E = Entries.end();
       I != E; ++I) {
    Streamer.EmitLabel(I->first);
    Streamer.EmitValue(I->second, 4);
  }
  Streamer.EmitDataRegion(MCDR_DataRegionEnd);
  Entries.clear();
}

const MCExpr *ConstantPool::addEntry(const MCExpr *Value, MCContext &Context) {
  MCSymbol *CPEntryLabel = Context.CreateTempSymbol();

  Entries.push_back(std::make_pair(CPEntryLabel, Value));
  return MCSymbolRefExpr::Create(CPEntryLabel, Context);
}

bool ConstantPool::empty() { return Entries.empty(); }

//
// AssemblerConstantPools implementation
//
ConstantPool *
AssemblerConstantPools::getConstantPool(const MCSection *Section) {
  ConstantPoolMapTy::iterator CP = ConstantPools.find(Section);
  if (CP == ConstantPools.end())
    return 0;

  return &CP->second;
}

ConstantPool &
AssemblerConstantPools::getOrCreateConstantPool(const MCSection *Section) {
  return ConstantPools[Section];
}

static void emitConstantPool(MCStreamer &Streamer, const MCSection *Section,
                             ConstantPool &CP) {
  if (!CP.empty()) {
    Streamer.SwitchSection(Section);
    CP.emitEntries(Streamer);
  }
}

void AssemblerConstantPools::emitAll(MCStreamer &Streamer) {
  // Dump contents of assembler constant pools.
  for (ConstantPoolMapTy::iterator CPI = ConstantPools.begin(),
                                   CPE = ConstantPools.end();
       CPI != CPE; ++CPI) {
    const MCSection *Section = CPI->first;
    ConstantPool &CP = CPI->second;

    emitConstantPool(Streamer, Section, CP);
  }
}

void AssemblerConstantPools::emitForCurrentSection(MCStreamer &Streamer) {
  const MCSection *Section = Streamer.getCurrentSection().first;
  if (ConstantPool *CP = getConstantPool(Section)) {
    emitConstantPool(Streamer, Section, *CP);
  }
}

const MCExpr *AssemblerConstantPools::addEntry(MCStreamer &Streamer,
                                               const MCExpr *Expr) {
  const MCSection *Section = Streamer.getCurrentSection().first;
  return getOrCreateConstantPool(Section).addEntry(Expr, Streamer.getContext());
}

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
