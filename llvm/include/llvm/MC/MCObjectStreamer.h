//===- MCObjectStreamer.h - MCStreamer Object File Interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCOBJECTSTREAMER_H
#define LLVM_MC_MCOBJECTSTREAMER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {
class MCAssembler;
class MCCodeEmitter;
class MCSectionData;
class MCSubtargetInfo;
class MCExpr;
class MCFragment;
class MCDataFragment;
class MCAsmBackend;
class raw_ostream;

/// \brief Streaming object file generation interface.
///
/// This class provides an implementation of the MCStreamer interface which is
/// suitable for use with the assembler backend. Specific object file formats
/// are expected to subclass this interface to implement directives specific
/// to that file format or custom semantics expected by the object writer
/// implementation.
class MCObjectStreamer : public MCStreamer {
  MCAssembler *Assembler;
  MCSectionData *CurSectionData;
  MCSectionData::iterator CurInsertionPoint;
  bool EmitEHFrame;
  bool EmitDebugFrame;
  SmallVector<MCSymbolData *, 2> PendingLabels;

  virtual void EmitInstToData(const MCInst &Inst, const MCSubtargetInfo&) = 0;
  void EmitCFIStartProcImpl(MCDwarfFrameInfo &Frame) override;
  void EmitCFIEndProcImpl(MCDwarfFrameInfo &Frame) override;

  // If any labels have been emitted but not assigned fragments, ensure that
  // they get assigned, either to F if possible or to a new data fragment.
  void flushPendingLabels(MCFragment *F);

protected:
  MCObjectStreamer(MCContext &Context, MCAsmBackend &TAB, raw_ostream &_OS,
                   MCCodeEmitter *_Emitter);
  MCObjectStreamer(MCContext &Context, MCAsmBackend &TAB, raw_ostream &_OS,
                   MCCodeEmitter *_Emitter, MCAssembler *_Assembler);
  ~MCObjectStreamer();

public:
  /// state management
  void reset() override;

  /// Object streamers require the integrated assembler.
  bool isIntegratedAssemblerRequired() const override { return true; }

  MCSymbolData &getOrCreateSymbolData(const MCSymbol *Symbol) {
    return getAssembler().getOrCreateSymbolData(*Symbol);
  }
  void EmitFrames(MCAsmBackend *MAB);
  void EmitCFISections(bool EH, bool Debug) override;

protected:
  MCSectionData *getCurrentSectionData() const {
    return CurSectionData;
  }

  MCFragment *getCurrentFragment() const;

  void insert(MCFragment *F) {
    flushPendingLabels(F);
    CurSectionData->getFragmentList().insert(CurInsertionPoint, F);
    F->setParent(CurSectionData);
  }

  /// Get a data fragment to write into, creating a new one if the current
  /// fragment is not a data fragment.
  MCDataFragment *getOrCreateDataFragment();

public:
  void visitUsedSymbol(const MCSymbol &Sym) override;

  MCAssembler &getAssembler() { return *Assembler; }

  /// @name MCStreamer Interface
  /// @{

  void EmitLabel(MCSymbol *Symbol) override;
  void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) override;
  void EmitValueImpl(const MCExpr *Value, unsigned Size,
                     const SMLoc &Loc = SMLoc()) override;
  void EmitULEB128Value(const MCExpr *Value) override;
  void EmitSLEB128Value(const MCExpr *Value) override;
  void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) override;
  void ChangeSection(const MCSection *Section,
                     const MCExpr *Subsection) override;
  void EmitInstruction(const MCInst &Inst, const MCSubtargetInfo& STI) override;

  /// \brief Emit an instruction to a special fragment, because this instruction
  /// can change its size during relaxation.
  virtual void EmitInstToFragment(const MCInst &Inst, const MCSubtargetInfo &);

  void EmitBundleAlignMode(unsigned AlignPow2) override;
  void EmitBundleLock(bool AlignToEnd) override;
  void EmitBundleUnlock() override;
  void EmitBytes(StringRef Data) override;
  void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                            unsigned ValueSize = 1,
                            unsigned MaxBytesToEmit = 0) override;
  void EmitCodeAlignment(unsigned ByteAlignment,
                         unsigned MaxBytesToEmit = 0) override;
  bool EmitValueToOffset(const MCExpr *Offset, unsigned char Value) override;
  void EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                             unsigned Column, unsigned Flags,
                             unsigned Isa, unsigned Discriminator,
                             StringRef FileName) override;
  void EmitDwarfAdvanceLineAddr(int64_t LineDelta, const MCSymbol *LastLabel,
                                const MCSymbol *Label,
                                unsigned PointerSize);
  void EmitDwarfAdvanceFrameAddr(const MCSymbol *LastLabel,
                                 const MCSymbol *Label);
  void EmitGPRel32Value(const MCExpr *Value) override;
  void EmitGPRel64Value(const MCExpr *Value) override;
  void EmitFill(uint64_t NumBytes, uint8_t FillValue) override;
  void EmitZeros(uint64_t NumBytes) override;
  void FinishImpl() override;

  bool mayHaveInstructions() const override {
    return getCurrentSectionData()->hasInstructions();
  }
};

} // end namespace llvm

#endif
