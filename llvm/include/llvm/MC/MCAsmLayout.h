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
  uint64_t CurrentLocation;

  MCAssembler &Assembler;

public:
  MCAsmLayout(MCAssembler &_Assembler)
    : CurrentLocation(0), Assembler(_Assember) {}

  /// Get the assembler object this is a layout for.
  MCAssembler &getAssembler() { return Assembler; }

  /// Get the current location value, i.e. that value of the '.' expression.
  uin64_t getCurrentLocation() {
    return CurrentLocation;
  }

  /// Set the current location.
  void setCurrentLocation(uint64_t Value) {
    CurrentLocation = Value;
  }
};

} // end namespace llvm

#endif
