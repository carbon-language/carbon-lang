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
  enum FaultType { FaultingLoad = 1, FaultTypeMax };

  static const char *faultTypeToString(FaultType);

  explicit FaultMaps(AsmPrinter &AP);

  void recordFaultingOp(FaultType FaultTy, const MCSymbol *HandlerLabel);
  void serializeToFaultMapSection();

private:
  static const char *WFMP;

  struct FaultInfo {
    FaultType FaultType;
    const MCExpr *FaultingOffsetExpr;
    const MCExpr *HandlerOffsetExpr;

    FaultInfo()
        : FaultType(FaultTypeMax), FaultingOffsetExpr(nullptr),
          HandlerOffsetExpr(nullptr) {}

    explicit FaultInfo(FaultMaps::FaultType FType, const MCExpr *FaultingOffset,
                       const MCExpr *HandlerOffset)
        : FaultType(FType), FaultingOffsetExpr(FaultingOffset),
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
}

#endif
