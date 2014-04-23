//===- MCTargetOptions.h - MC Target Options -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCTARGETOPTIONS_H
#define LLVM_MC_MCTARGETOPTIONS_H

namespace llvm {

class MCTargetOptions {
public:
  enum AsmInstrumentation {
    AsmInstrumentationNone,
    AsmInstrumentationAddress
  };

  /// Enables AddressSanitizer instrumentation at machine level.
  bool SanitizeAddress : 1;

  MCTargetOptions();
};

inline bool operator==(const MCTargetOptions &LHS, const MCTargetOptions &RHS) {
#define ARE_EQUAL(X) LHS.X == RHS.X
  return ARE_EQUAL(SanitizeAddress);
#undef ARE_EQUAL
}

inline bool operator!=(const MCTargetOptions &LHS, const MCTargetOptions &RHS) {
  return !(LHS == RHS);
}

} // end namespace llvm

#endif
