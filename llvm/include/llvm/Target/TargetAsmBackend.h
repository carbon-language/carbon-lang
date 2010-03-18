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

namespace llvm {
class MCSection;
class Target;

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
};

} // End llvm namespace

#endif
