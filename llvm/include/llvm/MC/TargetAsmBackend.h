//===-- llvm/MC/TargetAsmBackend.h - Target Asm Backend ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETASMBACKEND_H
#define LLVM_TARGET_TARGETASMBACKEND_H

#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCELFObjectTargetWriter;
class MCFixup;
class MCInst;
class MCObjectWriter;
class MCSection;
template<typename T>
class SmallVectorImpl;
class raw_ostream;

/// TargetAsmBackend - Generic interface to target specific assembler backends.
class TargetAsmBackend {
  TargetAsmBackend(const TargetAsmBackend &);   // DO NOT IMPLEMENT
  void operator=(const TargetAsmBackend &);  // DO NOT IMPLEMENT
protected: // Can only create subclasses.
  TargetAsmBackend();

  unsigned HasReliableSymbolDifference : 1;

public:
  virtual ~TargetAsmBackend();

  /// createObjectWriter - Create a new MCObjectWriter instance for use by the
  /// assembler backend to emit the final object file.
  virtual MCObjectWriter *createObjectWriter(raw_ostream &OS) const = 0;

  /// createELFObjectTargetWriter - Create a new ELFObjectTargetWriter to enable
  /// non-standard ELFObjectWriters.
  virtual  MCELFObjectTargetWriter *createELFObjectTargetWriter() const {
    assert(0 && "createELFObjectTargetWriter is not supported by asm backend");
    return 0;
  }

  /// hasReliableSymbolDifference - Check whether this target implements
  /// accurate relocations for differences between symbols. If not, differences
  /// between symbols will always be relocatable expressions and any references
  /// to temporary symbols will be assumed to be in the same atom, unless they
  /// reside in a different section.
  ///
  /// This should always be true (since it results in fewer relocations with no
  /// loss of functionality), but is currently supported as a way to maintain
  /// exact object compatibility with Darwin 'as' (on non-x86_64). It should
  /// eventually should be eliminated.
  bool hasReliableSymbolDifference() const {
    return HasReliableSymbolDifference;
  }

  /// doesSectionRequireSymbols - Check whether the given section requires that
  /// all symbols (even temporaries) have symbol table entries.
  virtual bool doesSectionRequireSymbols(const MCSection &Section) const {
    return false;
  }

  /// isSectionAtomizable - Check whether the given section can be split into
  /// atoms.
  ///
  /// \see MCAssembler::isSymbolLinkerVisible().
  virtual bool isSectionAtomizable(const MCSection &Section) const {
    return true;
  }

  /// @name Target Fixup Interfaces
  /// @{

  /// getNumFixupKinds - Get the number of target specific fixup kinds.
  virtual unsigned getNumFixupKinds() const = 0;

  /// getFixupKindInfo - Get information on a fixup kind.
  virtual const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const;

  /// @}

  /// ApplyFixup - Apply the \arg Value for given \arg Fixup into the provided
  /// data fragment, at the offset specified by the fixup and following the
  /// fixup kind as appropriate.
  virtual void ApplyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                          uint64_t Value) const = 0;

  /// @}

  /// @name Target Relaxation Interfaces
  /// @{

  /// MayNeedRelaxation - Check whether the given instruction may need
  /// relaxation.
  ///
  /// \param Inst - The instruction to test.
  virtual bool MayNeedRelaxation(const MCInst &Inst) const = 0;

  /// RelaxInstruction - Relax the instruction in the given fragment to the next
  /// wider instruction.
  ///
  /// \param Inst - The instruction to relax, which may be the same as the
  /// output.
  /// \parm Res [output] - On return, the relaxed instruction.
  virtual void RelaxInstruction(const MCInst &Inst, MCInst &Res) const = 0;

  /// @}

  /// WriteNopData - Write an (optimal) nop sequence of Count bytes to the given
  /// output. If the target cannot generate such a sequence, it should return an
  /// error.
  ///
  /// \return - True on success.
  virtual bool WriteNopData(uint64_t Count, MCObjectWriter *OW) const = 0;

  /// HandleAssemblerFlag - Handle any target-specific assembler flags.
  /// By default, do nothing.
  virtual void HandleAssemblerFlag(MCAssemblerFlag Flag) {}
};

} // End llvm namespace

#endif
