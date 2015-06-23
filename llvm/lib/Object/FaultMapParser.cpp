//===----------------------- FaultMapParser.cpp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/FaultMapParser.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"

using namespace llvm;

const char *faultKindToString(FaultMapParser::FaultKind FT) {
  switch (FT) {
  default:
    llvm_unreachable("unhandled fault type!");

  case FaultMapParser::FaultingLoad:
    return "FaultingLoad";
  }
}

raw_ostream &llvm::
operator<<(raw_ostream &OS,
           const FaultMapParser::FunctionFaultInfoAccessor &FFI) {
  OS << "Fault kind: "
     << faultKindToString((FaultMapParser::FaultKind)FFI.getFaultKind())
     << ", faulting PC offset: " << FFI.getFaultingPCOffset()
     << ", handling PC offset: " << FFI.getHandlerPCOffset();
  return OS;
}

raw_ostream &llvm::
operator<<(raw_ostream &OS, const FaultMapParser::FunctionInfoAccessor &FI) {
  OS << "FunctionAddress: " << format_hex(FI.getFunctionAddr(), 8)
     << ", NumFaultingPCs: " << FI.getNumFaultingPCs() << "\n";
  for (unsigned i = 0, e = FI.getNumFaultingPCs(); i != e; ++i)
    OS << FI.getFunctionFaultInfoAt(i) << "\n";
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const FaultMapParser &FMP) {
  OS << "Version: " << format_hex(FMP.getFaultMapVersion(), 2) << "\n";
  OS << "NumFunctions: " << FMP.getNumFunctions() << "\n";

  if (FMP.getNumFunctions() == 0)
    return OS;

  FaultMapParser::FunctionInfoAccessor FI;

  for (unsigned i = 0, e = FMP.getNumFunctions(); i != e; ++i) {
    FI = (i == 0) ? FMP.getFirstFunctionInfo() : FI.getNextFunctionInfo();
    OS << FI;
  }

  return OS;
}
