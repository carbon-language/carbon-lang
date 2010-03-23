//===-- llvm/Target/TargetAsmBackend.h - Target Asm Backend -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETASMBACKEND_H
#define LLVM_TARGET_TARGETASMBACKEND_H

#include "llvm/System/DataTypes.h"

namespace llvm {
class MCAsmFixup;
class MCDataFragment;
class MCInst;
class MCInstFragment;
class MCObjectWriter;
class MCSection;
template<typename T>
class SmallVectorImpl;
class Target;
class raw_ostream;

/// TargetAsmBackend - Generic interface to target specific assembler backends.
class TargetAsmBackend {
  TargetAsmBackend(const TargetAsmBackend &);   // DO NOT IMPLEMENT
  void operator=(const TargetAsmBackend &);  // DO NOT IMPLEMENT
protected: // Can only create subclasses.
  TargetAsmBackend(const Target &);

  /// TheTarget - The Target that this machine was created for.
  const Target &TheTarget;

  unsigned HasAbsolutizedSet : 1;
  unsigned HasReliableSymbolDifference : 1;
  unsigned HasScatteredSymbols : 1;

public:
  virtual ~TargetAsmBackend();

  const Target &getTarget() const { return TheTarget; }

  /// createObjectWriter - Create a new MCObjectWriter instance for use by the
  /// assembler backend to emit the final object file.
  virtual MCObjectWriter *createObjectWriter(raw_ostream &OS) const = 0;

  /// hasAbsolutizedSet - Check whether this target "absolutizes"
  /// assignments. That is, given code like:
  ///   a:
  ///   ...
  ///   b:
  ///   tmp = a - b
  ///       .long tmp
  /// will the value of 'tmp' be a relocatable expression, or the assembly time
  /// value of L0 - L1. This distinction is only relevant for platforms that
  /// support scattered symbols, since in the absence of scattered symbols (a -
  /// b) cannot change after assembly.
  bool hasAbsolutizedSet() const { return HasAbsolutizedSet; }

  /// hasReliableSymbolDifference - Check whether this target implements
  /// accurate relocations for differences between symbols. If not, differences
  /// between symbols will always be relocatable expressions and any references
  /// to temporary symbols will be assumed to be in the same atom, unless they
  /// reside in a different section.
  ///
  /// This should always be true (since it results in fewer relocations with no
  /// loss of functionality), but is currently supported as a way to maintain
  /// exact object compatibility with Darwin 'as' (on non-x86_64). It should
  /// eventually should be eliminated. See also \see hasAbsolutizedSet.
  bool hasReliableSymbolDifference() const {
    return HasReliableSymbolDifference;
  }

  /// hasScatteredSymbols - Check whether this target supports scattered
  /// symbols. If so, the assembler should assume that atoms can be scattered by
  /// the linker. In particular, this means that the offsets between symbols
  /// which are in distinct atoms is not known at link time, and the assembler
  /// must generate fixups and relocations appropriately.
  ///
  /// Note that the assembler currently does not reason about atoms, instead it
  /// assumes all temporary symbols reside in the "current atom".
  bool hasScatteredSymbols() const { return HasScatteredSymbols; }

  /// doesSectionRequireSymbols - Check whether the given section requires that
  /// all symbols (even temporaries) have symbol table entries.
  virtual bool doesSectionRequireSymbols(const MCSection &Section) const {
    return false;
  }

  /// isVirtualSection - Check whether the given section is "virtual", that is
  /// has no actual object file contents.
  virtual bool isVirtualSection(const MCSection &Section) const = 0;

  /// ApplyFixup - Apply the \arg Value for given \arg Fixup into the provided
  /// data fragment, at the offset specified by the fixup and following the
  /// fixup kind as appropriate.
  virtual void ApplyFixup(const MCAsmFixup &Fixup, MCDataFragment &Fragment,
                          uint64_t Value) const = 0;

  /// MayNeedRelaxation - Check whether the given instruction may need
  /// relaxation.
  ///
  /// \arg Inst - The instruction to test.
  /// \arg Fixups - The actual fixups this instruction encoded to, for potential
  /// use by the target backend.
  virtual bool MayNeedRelaxation(const MCInst &Inst,
                           const SmallVectorImpl<MCAsmFixup> &Fixups) const = 0;

  /// RelaxInstruction - Relax the instruction in the given fragment to the next
  /// wider instruction.
  virtual void RelaxInstruction(const MCInstFragment *IF,
                                MCInst &Res) const = 0;

  /// WriteNopData - Write an (optimal) nop sequence of Count bytes to the given
  /// output. If the target cannot generate such a sequence, it should return an
  /// error.
  ///
  /// \return - True on success.
  virtual bool WriteNopData(uint64_t Count, MCObjectWriter *OW) const = 0;
};

} // End llvm namespace

#endif
