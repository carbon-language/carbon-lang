//===- MCAsmLayout.h - Assembly Layout Object -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMLAYOUT_H
#define LLVM_MC_MCASMLAYOUT_H

namespace llvm {
class MCAssembler;

/// Encapsulates the layout of an assembly file at a particular point in time.
///
/// Assembly may requiring compute multiple layouts for a particular assembly
/// file as part of the relaxation process. This class encapsulates the layout
/// at a single point in time in such a way that it is always possible to
/// efficiently compute the exact addresses of any symbol in the assembly file,
/// even during the relaxation process.
class MCAsmLayout {
private:
  MCAssembler &Assembler;

public:
  MCAsmLayout(MCAssembler &_Assembler) : Assembler(_Assembler) {}

  /// Get the assembler object this is a layout for.
  MCAssembler &getAssembler() const { return Assembler; }
};

} // end namespace llvm

#endif
