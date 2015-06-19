//===------------------- FaultMaps.h - StackMaps ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_FAULTMAPS_H
#define LLVM_CODEGEN_FAULTMAPS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCSymbol.h"

#include <vector>
#include <map>

namespace llvm {

class AsmPrinter;
class MCExpr;
class MCSymbol;
class MCStreamer;

class FaultMaps {
public:
  enum FaultKind { FaultingLoad = 1, FaultKindMax };

  static const char *faultTypeToString(FaultKind);

  explicit FaultMaps(AsmPrinter &AP);

  void recordFaultingOp(FaultKind FaultTy, const MCSymbol *HandlerLabel);
  void serializeToFaultMapSection();

private:
  static const char *WFMP;

  struct FaultInfo {
    FaultKind Kind;
    const MCExpr *FaultingOffsetExpr;
    const MCExpr *HandlerOffsetExpr;

    FaultInfo()
        : Kind(FaultKindMax), FaultingOffsetExpr(nullptr),
          HandlerOffsetExpr(nullptr) {}

    explicit FaultInfo(FaultMaps::FaultKind Kind, const MCExpr *FaultingOffset,
                       const MCExpr *HandlerOffset)
        : Kind(Kind), FaultingOffsetExpr(FaultingOffset),
          HandlerOffsetExpr(HandlerOffset) {}
  };

  typedef std::vector<FaultInfo> FunctionFaultInfos;

  // We'd like to keep a stable iteration order for FunctionInfos to help
  // FileCheck based testing.
  struct MCSymbolComparator {
    bool operator()(const MCSymbol *LHS, const MCSymbol *RHS) const {
      return LHS->getName() < RHS->getName();
    }
  };

  std::map<const MCSymbol *, FunctionFaultInfos, MCSymbolComparator>
      FunctionInfos;
  AsmPrinter &AP;

  void emitFunctionInfo(const MCSymbol *FnLabel, const FunctionFaultInfos &FFI);
};
} // namespace llvm

#endif
