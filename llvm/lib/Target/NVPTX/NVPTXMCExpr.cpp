//===-- NVPTXMCExpr.cpp - NVPTX specific MC expression classes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NVPTXMCExpr.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
using namespace llvm;

#define DEBUG_TYPE "nvptx-mcexpr"

const NVPTXFloatMCExpr*
NVPTXFloatMCExpr::Create(VariantKind Kind, APFloat Flt, MCContext &Ctx) {
  return new (Ctx) NVPTXFloatMCExpr(Kind, Flt);
}

void NVPTXFloatMCExpr::PrintImpl(raw_ostream &OS) const {
  bool Ignored;
  unsigned NumHex;
  APFloat APF = getAPFloat();

  switch (Kind) {
  default: llvm_unreachable("Invalid kind!");
  case VK_NVPTX_SINGLE_PREC_FLOAT:
    OS << "0f";
    NumHex = 8;
    APF.convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven, &Ignored);
    break;
  case VK_NVPTX_DOUBLE_PREC_FLOAT:
    OS << "0d";
    NumHex = 16;
    APF.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, &Ignored);
    break;
  }

  APInt API = APF.bitcastToAPInt();
  std::string HexStr(utohexstr(API.getZExtValue()));
  if (HexStr.length() < NumHex)
    OS << std::string(NumHex - HexStr.length(), '0');
  OS << utohexstr(API.getZExtValue());
}

const NVPTXGenericMCSymbolRefExpr*
NVPTXGenericMCSymbolRefExpr::Create(const MCSymbolRefExpr *SymExpr,
                                    MCContext &Ctx) {
  return new (Ctx) NVPTXGenericMCSymbolRefExpr(SymExpr);
}

void NVPTXGenericMCSymbolRefExpr::PrintImpl(raw_ostream &OS) const {
  OS << "generic(" << *SymExpr << ")";
}
