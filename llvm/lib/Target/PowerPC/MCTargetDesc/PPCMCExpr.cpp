//===-- PPCMCExpr.cpp - PPC specific MC expression classes ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ppcmcexpr"
#include "PPCMCExpr.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCAsmInfo.h"

using namespace llvm;

const PPCMCExpr*
PPCMCExpr::Create(VariantKind Kind, const MCExpr *Expr,
                       MCContext &Ctx) {
  int AssemblerDialect = Ctx.getAsmInfo()->getAssemblerDialect();
  return new (Ctx) PPCMCExpr(Kind, Expr, AssemblerDialect);
}

void PPCMCExpr::PrintImpl(raw_ostream &OS) const {
  if (isDarwinSyntax()) {
    switch (Kind) {
    default: llvm_unreachable("Invalid kind!");
    case VK_PPC_LO: OS << "lo16"; break;
    case VK_PPC_HI: OS << "hi16"; break;
    case VK_PPC_HA: OS << "ha16"; break;
    }

    OS << '(';
    getSubExpr()->print(OS);
    OS << ')';
  } else {
    getSubExpr()->print(OS);

    switch (Kind) {
    default: llvm_unreachable("Invalid kind!");
    case VK_PPC_LO: OS << "@l"; break;
    case VK_PPC_HI: OS << "@h"; break;
    case VK_PPC_HA: OS << "@ha"; break;
    }
  }
}

bool
PPCMCExpr::EvaluateAsRelocatableImpl(MCValue &Res,
                                     const MCAsmLayout *Layout) const {
  MCValue Value;

  if (!getSubExpr()->EvaluateAsRelocatable(Value, *Layout))
    return false;

  if (Value.isAbsolute()) {
    int64_t Result = Value.getConstant();
    switch (Kind) {
      default:
        llvm_unreachable("Invalid kind!");
      case VK_PPC_LO:
        Result = Result & 0xffff;
        break;
      case VK_PPC_HI:
        Result = (Result >> 16) & 0xffff;
        break;
      case VK_PPC_HA:
        Result = ((Result >> 16) + ((Result & 0x8000) ? 1 : 0)) & 0xffff;
        break;
    }
    Res = MCValue::get(Result);
  } else {
    MCContext &Context = Layout->getAssembler().getContext();
    const MCSymbolRefExpr *Sym = Value.getSymA();
    MCSymbolRefExpr::VariantKind Modifier = Sym->getKind();
    if (Modifier != MCSymbolRefExpr::VK_None)
      return false;
    switch (Kind) {
      default:
        llvm_unreachable("Invalid kind!");
      case VK_PPC_LO:
        Modifier = MCSymbolRefExpr::VK_PPC_LO;
        break;
      case VK_PPC_HI:
        Modifier = MCSymbolRefExpr::VK_PPC_HI;
        break;
      case VK_PPC_HA:
        Modifier = MCSymbolRefExpr::VK_PPC_HA;
        break;
    }
    Sym = MCSymbolRefExpr::Create(&Sym->getSymbol(), Modifier, Context);
    Res = MCValue::get(Sym, Value.getSymB(), Value.getConstant());
  }

  return true;
}

// FIXME: This basically copies MCObjectStreamer::AddValueSymbols. Perhaps
// that method should be made public?
static void AddValueSymbols_(const MCExpr *Value, MCAssembler *Asm) {
  switch (Value->getKind()) {
  case MCExpr::Target:
    llvm_unreachable("Can't handle nested target expr!");

  case MCExpr::Constant:
    break;

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(Value);
    AddValueSymbols_(BE->getLHS(), Asm);
    AddValueSymbols_(BE->getRHS(), Asm);
    break;
  }

  case MCExpr::SymbolRef:
    Asm->getOrCreateSymbolData(cast<MCSymbolRefExpr>(Value)->getSymbol());
    break;

  case MCExpr::Unary:
    AddValueSymbols_(cast<MCUnaryExpr>(Value)->getSubExpr(), Asm);
    break;
  }
}

void PPCMCExpr::AddValueSymbols(MCAssembler *Asm) const {
  AddValueSymbols_(getSubExpr(), Asm);
}
