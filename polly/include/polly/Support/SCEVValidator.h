//===--- SCEVValidator.h - Detect Scops -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Checks if a SCEV expression represents a valid affine expression.
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCEV_VALIDATOR_H
#define POLLY_SCEV_VALIDATOR_H

namespace llvm {
  class Region;
  class SCEV;
  class ScalarEvolution;
  class Value;
}

namespace polly {
  bool isAffineExpr(const llvm::Region *R, const llvm::SCEV *Expression,
                    llvm::ScalarEvolution &SE, llvm::Value **BaseAddress = 0);
}

#endif
