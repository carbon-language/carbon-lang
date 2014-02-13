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

  virtual void EmitInstToData(const MCInst &Inst, const MCSubtargetInfo&) = 0;
  virtual void EmitCFIStartProcImpl(MCDwarfFrameInfo &Frame);
  virtual void EmitCFIEndProcImpl(MCDwarfFrameInfo &Frame);

protected:
  MCObjectStreamer(MCContext &Context, MCAsmBackend &TAB, raw_ostream &_OS,
                   MCCodeEmitter *_Emitter);
  MCObjectStreamer(MCContext &Context, MCAsmBackend &TAB, raw_ostream &_OS,
                   MCCodeEmitter *_Emitter, MCAssembler *_Assembler);
  ~MCObjectStreamer();

public:
  /// state management
  virtual void reset();

  /// Object streamers require the integrated assembler.
  virtual bool isIntegratedAssemblerRequired() const { return true; }

protected:
  MCSectionData *getCurrentSectionData() const {
    return CurSectionData;
  }

  MCFragment *getCurrentFragment() const;

  void insert(MCFragment *F) const {
    CurSectionData->getFragmentList().insert(CurInsertionPoint, F);
    F->setParent(CurSectionData);
  }

  /// Get a data fragment to write into, creating a new one if the current
  /// fragment is not a data fragment.
  MCDataFragment *getOrCreateDataFragment() const;

  const MCExpr *AddValueSymbols(const MCExpr *Value);

public:
  MCAssembler &getAssembler() { return *Assembler; }

  /// @name MCStreamer Interface
  /// @{

  virtual void EmitLabel(MCSymbol *Symbol);
  virtual void EmitDebugLabel(MCSymbol *Symbol);
  virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value);
  virtual void EmitValueImpl(const MCExpr *Value, unsigned Size);
  virtual void EmitULEB128Value(const MCExpr *Value);
  virtual void EmitSLEB128Value(const MCExpr *Value);
  virtual void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol);
  virtual void ChangeSection(const MCSection *Section,
                             const MCExpr *Subsection);
  virtual void EmitInstruction(const MCInst &Inst, const MCSubtargetInfo& STI);

  /// \brief Emit an instruction to a special fragment, because this instruction
  /// can change its size during relaxation.
  virtual void EmitInstToFragment(const MCInst &Inst, const MCSubtargetInfo &);

  virtual void EmitBundleAlignMode(unsigned AlignPow2);
  virtual void EmitBundleLock(bool AlignToEnd);
  virtual void EmitBundleUnlock();
  virtual void EmitBytes(StringRef Data);
  virtual void EmitValueToAlignment(unsigned ByteAlignment,
                                    int64_t Value = 0,
                                    unsigned ValueSize = 1,
                                    unsigned MaxBytesToEmit = 0);
  virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                 unsigned MaxBytesToEmit = 0);
  virtual bool EmitValueToOffset(const MCExpr *Offset, unsigned char Value);
  virtual void EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                                     unsigned Column, unsigned Flags,
                                     unsigned Isa, unsigned Discriminator,
                                     StringRef FileName);
  virtual void EmitDwarfAdvanceLineAddr(int64_t LineDelta,
                                        const MCSymbol *LastLabel,
                                        const MCSymbol *Label,
                                        unsigned PointerSize);
  virtual void EmitDwarfAdvanceFrameAddr(const MCSymbol *LastLabel,
                                         const MCSymbol *Label);
  virtual void EmitGPRel32Value(const MCExpr *Value);
  virtual void EmitGPRel64Value(const MCExpr *Value);
  virtual void EmitFill(uint64_t NumBytes, uint8_t FillValue);
  virtual void EmitZeros(uint64_t NumBytes);
  virtual void FinishImpl();
};

} // end namespace llvm

#endif
