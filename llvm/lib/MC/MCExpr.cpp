//===- MCExpr.cpp - Assembly Level Expression Implementation --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mcexpr"
#include "llvm/MC/MCExpr.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFormat.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetAsmBackend.h"
using namespace llvm;

namespace {
namespace stats {
STATISTIC(MCExprEvaluate, "Number of MCExpr evaluations");
}
}

void MCExpr::print(raw_ostream &OS) const {
  switch (getKind()) {
  case MCExpr::Target:
    return cast<MCTargetExpr>(this)->PrintImpl(OS);
  case MCExpr::Constant:
    OS << cast<MCConstantExpr>(*this).getValue();
    return;

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr &SRE = cast<MCSymbolRefExpr>(*this);
    const MCSymbol &Sym = SRE.getSymbol();

    if (SRE.getKind() == MCSymbolRefExpr::VK_ARM_HI16 ||
        SRE.getKind() == MCSymbolRefExpr::VK_ARM_LO16)
      OS << MCSymbolRefExpr::getVariantKindName(SRE.getKind());

    // Parenthesize names that start with $ so that they don't look like
    // absolute names.
    if (Sym.getName()[0] == '$')
      OS << '(' << Sym << ')';
    else
      OS << Sym;

    if (SRE.getKind() == MCSymbolRefExpr::VK_ARM_PLT)
      OS << MCSymbolRefExpr::getVariantKindName(SRE.getKind());
    else if (SRE.getKind() != MCSymbolRefExpr::VK_None &&
             SRE.getKind() != MCSymbolRefExpr::VK_ARM_HI16 &&
             SRE.getKind() != MCSymbolRefExpr::VK_ARM_LO16)
      OS << '@' << MCSymbolRefExpr::getVariantKindName(SRE.getKind());

    return;
  }

  case MCExpr::Unary: {
    const MCUnaryExpr &UE = cast<MCUnaryExpr>(*this);
    switch (UE.getOpcode()) {
    default: assert(0 && "Invalid opcode!");
    case MCUnaryExpr::LNot:  OS << '!'; break;
    case MCUnaryExpr::Minus: OS << '-'; break;
    case MCUnaryExpr::Not:   OS << '~'; break;
    case MCUnaryExpr::Plus:  OS << '+'; break;
    }
    OS << *UE.getSubExpr();
    return;
  }

  case MCExpr::Binary: {
    const MCBinaryExpr &BE = cast<MCBinaryExpr>(*this);

    // Only print parens around the LHS if it is non-trivial.
    if (isa<MCConstantExpr>(BE.getLHS()) || isa<MCSymbolRefExpr>(BE.getLHS())) {
      OS << *BE.getLHS();
    } else {
      OS << '(' << *BE.getLHS() << ')';
    }

    switch (BE.getOpcode()) {
    default: assert(0 && "Invalid opcode!");
    case MCBinaryExpr::Add:
      // Print "X-42" instead of "X+-42".
      if (const MCConstantExpr *RHSC = dyn_cast<MCConstantExpr>(BE.getRHS())) {
        if (RHSC->getValue() < 0) {
          OS << RHSC->getValue();
          return;
        }
      }

      OS <<  '+';
      break;
    case MCBinaryExpr::And:  OS <<  '&'; break;
    case MCBinaryExpr::Div:  OS <<  '/'; break;
    case MCBinaryExpr::EQ:   OS << "=="; break;
    case MCBinaryExpr::GT:   OS <<  '>'; break;
    case MCBinaryExpr::GTE:  OS << ">="; break;
    case MCBinaryExpr::LAnd: OS << "&&"; break;
    case MCBinaryExpr::LOr:  OS << "||"; break;
    case MCBinaryExpr::LT:   OS <<  '<'; break;
    case MCBinaryExpr::LTE:  OS << "<="; break;
    case MCBinaryExpr::Mod:  OS <<  '%'; break;
    case MCBinaryExpr::Mul:  OS <<  '*'; break;
    case MCBinaryExpr::NE:   OS << "!="; break;
    case MCBinaryExpr::Or:   OS <<  '|'; break;
    case MCBinaryExpr::Shl:  OS << "<<"; break;
    case MCBinaryExpr::Shr:  OS << ">>"; break;
    case MCBinaryExpr::Sub:  OS <<  '-'; break;
    case MCBinaryExpr::Xor:  OS <<  '^'; break;
    }

    // Only print parens around the LHS if it is non-trivial.
    if (isa<MCConstantExpr>(BE.getRHS()) || isa<MCSymbolRefExpr>(BE.getRHS())) {
      OS << *BE.getRHS();
    } else {
      OS << '(' << *BE.getRHS() << ')';
    }
    return;
  }
  }

  assert(0 && "Invalid expression kind!");
}

void MCExpr::dump() const {
  print(dbgs());
  dbgs() << '\n';
}

/* *** */

const MCBinaryExpr *MCBinaryExpr::Create(Opcode Opc, const MCExpr *LHS,
                                         const MCExpr *RHS, MCContext &Ctx) {
  return new (Ctx) MCBinaryExpr(Opc, LHS, RHS);
}

const MCUnaryExpr *MCUnaryExpr::Create(Opcode Opc, const MCExpr *Expr,
                                       MCContext &Ctx) {
  return new (Ctx) MCUnaryExpr(Opc, Expr);
}

const MCConstantExpr *MCConstantExpr::Create(int64_t Value, MCContext &Ctx) {
  return new (Ctx) MCConstantExpr(Value);
}

/* *** */

const MCSymbolRefExpr *MCSymbolRefExpr::Create(const MCSymbol *Sym,
                                               VariantKind Kind,
                                               MCContext &Ctx) {
  return new (Ctx) MCSymbolRefExpr(Sym, Kind);
}

const MCSymbolRefExpr *MCSymbolRefExpr::Create(StringRef Name, VariantKind Kind,
                                               MCContext &Ctx) {
  return Create(Ctx.GetOrCreateSymbol(Name), Kind, Ctx);
}

StringRef MCSymbolRefExpr::getVariantKindName(VariantKind Kind) {
  switch (Kind) {
  default:
  case VK_Invalid: return "<<invalid>>";
  case VK_None: return "<<none>>";

  case VK_GOT: return "GOT";
  case VK_GOTOFF: return "GOTOFF";
  case VK_GOTPCREL: return "GOTPCREL";
  case VK_GOTTPOFF: return "GOTTPOFF";
  case VK_INDNTPOFF: return "INDNTPOFF";
  case VK_NTPOFF: return "NTPOFF";
  case VK_GOTNTPOFF: return "GOTNTPOFF";
  case VK_PLT: return "PLT";
  case VK_TLSGD: return "TLSGD";
  case VK_TLSLD: return "TLSLD";
  case VK_TLSLDM: return "TLSLDM";
  case VK_TPOFF: return "TPOFF";
  case VK_DTPOFF: return "DTPOFF";
  case VK_ARM_HI16: return ":upper16:";
  case VK_ARM_LO16: return ":lower16:";
  case VK_ARM_PLT: return "(PLT)";
  case VK_TLVP: return "TLVP";
  }
}

MCSymbolRefExpr::VariantKind
MCSymbolRefExpr::getVariantKindForName(StringRef Name) {
  return StringSwitch<VariantKind>(Name)
    .Case("GOT", VK_GOT)
    .Case("GOTOFF", VK_GOTOFF)
    .Case("GOTPCREL", VK_GOTPCREL)
    .Case("GOTTPOFF", VK_GOTTPOFF)
    .Case("INDNTPOFF", VK_INDNTPOFF)
    .Case("NTPOFF", VK_NTPOFF)
    .Case("GOTNTPOFF", VK_GOTNTPOFF)
    .Case("PLT", VK_PLT)
    .Case("TLSGD", VK_TLSGD)
    .Case("TLSLD", VK_TLSLD)
    .Case("TLSLDM", VK_TLSLDM)
    .Case("TPOFF", VK_TPOFF)
    .Case("DTPOFF", VK_DTPOFF)
    .Case("TLVP", VK_TLVP)
    .Default(VK_Invalid);
}

/* *** */

void MCTargetExpr::Anchor() {}

/* *** */

bool MCExpr::EvaluateAsAbsolute(int64_t &Res, const MCAsmLayout *Layout) const {
  MCValue Value;

  // Fast path constants.
  if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(this)) {
    Res = CE->getValue();
    return true;
  }

  if (!EvaluateAsRelocatable(Value, Layout) || !Value.isAbsolute()) {
    // EvaluateAsAbsolute is defined to return the "current value" of
    // the expression if we are given a Layout object, even in cases
    // when the value is not fixed.
    if (Layout) {
      Res = Value.getConstant();
      if (Value.getSymA()) {
	Res += Layout->getSymbolAddress(
          &Layout->getAssembler().getSymbolData(Value.getSymA()->getSymbol()));
      }
      if (Value.getSymB()) {
	Res -= Layout->getSymbolAddress(
          &Layout->getAssembler().getSymbolData(Value.getSymB()->getSymbol()));
      }
    }
    return false;
  }

  Res = Value.getConstant();
  return true;
}

static bool EvaluateSymbolicAdd(const MCAsmLayout *Layout, bool InSet,
                                const MCValue &LHS,const MCSymbolRefExpr *RHS_A,
                                const MCSymbolRefExpr *RHS_B, int64_t RHS_Cst,
                                MCValue &Res) {
  // We can't add or subtract two symbols.
  if ((LHS.getSymA() && RHS_A) ||
      (LHS.getSymB() && RHS_B))
    return false;

  const MCSymbolRefExpr *A = LHS.getSymA() ? LHS.getSymA() : RHS_A;
  const MCSymbolRefExpr *B = LHS.getSymB() ? LHS.getSymB() : RHS_B;
  if (B) {
    // If we have a negated symbol, then we must have also have a non-negated
    // symbol in order to encode the expression. We can do this check later to
    // permit expressions which eventually fold to a representable form -- such
    // as (a + (0 - b)) -- if necessary.
    if (!A)
      return false;
  }

  // Absolutize symbol differences between defined symbols when we have a
  // layout object and the target requests it.

  if (Layout && A && B) {
    const MCSymbol &SA = A->getSymbol();
    const MCSymbol &SB = B->getSymbol();
    const MCObjectFormat &F =
      Layout->getAssembler().getBackend().getObjectFormat();
    if (SA.isDefined() && SB.isDefined() && F.isAbsolute(InSet, SA, SB)) {
      const MCAssembler &Asm = Layout->getAssembler();
      MCSymbolData &AD = Asm.getSymbolData(A->getSymbol());
      MCSymbolData &BD = Asm.getSymbolData(B->getSymbol());
      Res = MCValue::get(+ Layout->getSymbolAddress(&AD)
                         - Layout->getSymbolAddress(&BD)
                         + LHS.getConstant()
                         + RHS_Cst);
      return true;
    }
  }


  Res = MCValue::get(A, B, LHS.getConstant() + RHS_Cst);
  return true;
}

bool MCExpr::EvaluateAsRelocatable(MCValue &Res,
                                   const MCAsmLayout *Layout) const {
  return EvaluateAsRelocatableImpl(Res, Layout, false);
}

bool MCExpr::EvaluateAsRelocatableImpl(MCValue &Res,
                                       const MCAsmLayout *Layout,
                                       bool InSet) const {
  ++stats::MCExprEvaluate;

  switch (getKind()) {
  case Target:
    return cast<MCTargetExpr>(this)->EvaluateAsRelocatableImpl(Res, Layout);

  case Constant:
    Res = MCValue::get(cast<MCConstantExpr>(this)->getValue());
    return true;

  case SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(this);
    const MCSymbol &Sym = SRE->getSymbol();

    // Evaluate recursively if this is a variable.
    if (Sym.isVariable() && SRE->getKind() == MCSymbolRefExpr::VK_None)
      return Sym.getVariableValue()->EvaluateAsRelocatableImpl(Res, Layout,
                                                               true);

    Res = MCValue::get(SRE, 0, 0);
    return true;
  }

  case Unary: {
    const MCUnaryExpr *AUE = cast<MCUnaryExpr>(this);
    MCValue Value;

    if (!AUE->getSubExpr()->EvaluateAsRelocatableImpl(Value, Layout, InSet))
      return false;

    switch (AUE->getOpcode()) {
    case MCUnaryExpr::LNot:
      if (!Value.isAbsolute())
        return false;
      Res = MCValue::get(!Value.getConstant());
      break;
    case MCUnaryExpr::Minus:
      /// -(a - b + const) ==> (b - a - const)
      if (Value.getSymA() && !Value.getSymB())
        return false;
      Res = MCValue::get(Value.getSymB(), Value.getSymA(),
                         -Value.getConstant());
      break;
    case MCUnaryExpr::Not:
      if (!Value.isAbsolute())
        return false;
      Res = MCValue::get(~Value.getConstant());
      break;
    case MCUnaryExpr::Plus:
      Res = Value;
      break;
    }

    return true;
  }

  case Binary: {
    const MCBinaryExpr *ABE = cast<MCBinaryExpr>(this);
    MCValue LHSValue, RHSValue;

    if (!ABE->getLHS()->EvaluateAsRelocatableImpl(LHSValue, Layout, InSet) ||
        !ABE->getRHS()->EvaluateAsRelocatableImpl(RHSValue, Layout, InSet))
      return false;

    // We only support a few operations on non-constant expressions, handle
    // those first.
    if (!LHSValue.isAbsolute() || !RHSValue.isAbsolute()) {
      switch (ABE->getOpcode()) {
      default:
        return false;
      case MCBinaryExpr::Sub:
        // Negate RHS and add.
        return EvaluateSymbolicAdd(Layout, InSet, LHSValue,
                                   RHSValue.getSymB(), RHSValue.getSymA(),
                                   -RHSValue.getConstant(),
                                   Res);

      case MCBinaryExpr::Add:
        return EvaluateSymbolicAdd(Layout, InSet, LHSValue,
                                   RHSValue.getSymA(), RHSValue.getSymB(),
                                   RHSValue.getConstant(),
                                   Res);
      }
    }

    // FIXME: We need target hooks for the evaluation. It may be limited in
    // width, and gas defines the result of comparisons and right shifts
    // differently from Apple as.
    int64_t LHS = LHSValue.getConstant(), RHS = RHSValue.getConstant();
    int64_t Result = 0;
    switch (ABE->getOpcode()) {
    case MCBinaryExpr::Add:  Result = LHS + RHS; break;
    case MCBinaryExpr::And:  Result = LHS & RHS; break;
    case MCBinaryExpr::Div:  Result = LHS / RHS; break;
    case MCBinaryExpr::EQ:   Result = LHS == RHS; break;
    case MCBinaryExpr::GT:   Result = LHS > RHS; break;
    case MCBinaryExpr::GTE:  Result = LHS >= RHS; break;
    case MCBinaryExpr::LAnd: Result = LHS && RHS; break;
    case MCBinaryExpr::LOr:  Result = LHS || RHS; break;
    case MCBinaryExpr::LT:   Result = LHS < RHS; break;
    case MCBinaryExpr::LTE:  Result = LHS <= RHS; break;
    case MCBinaryExpr::Mod:  Result = LHS % RHS; break;
    case MCBinaryExpr::Mul:  Result = LHS * RHS; break;
    case MCBinaryExpr::NE:   Result = LHS != RHS; break;
    case MCBinaryExpr::Or:   Result = LHS | RHS; break;
    case MCBinaryExpr::Shl:  Result = LHS << RHS; break;
    case MCBinaryExpr::Shr:  Result = LHS >> RHS; break;
    case MCBinaryExpr::Sub:  Result = LHS - RHS; break;
    case MCBinaryExpr::Xor:  Result = LHS ^ RHS; break;
    }

    Res = MCValue::get(Result);
    return true;
  }
  }

  assert(0 && "Invalid assembly expression kind!");
  return false;
}
