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
  virtual bool hasAbsolutizedSet() const { return false; }

  /// hasScatteredSymbols - Check whether this target supports scattered
  /// symbols. If so, the assembler should assume that atoms can be scattered by
  /// the linker. In particular, this means that the offsets between symbols
  /// which are in distinct atoms is not known at link time, and the assembler
  /// must generate fixups and relocations appropriately.
  ///
  /// Note that the assembler currently does not reason about atoms, instead it
  /// assumes all temporary symbols reside in the "current atom".
  virtual bool hasScatteredSymbols() const { return false; }

  /// doesSectionRequireSymbols - Check whether the given section requires that
  /// all symbols (even temporaries) have symbol table entries.
  virtual bool doesSectionRequireSymbols(const MCSection &Section) const {
    return false;
  }
};

} // End llvm namespace

#endif
