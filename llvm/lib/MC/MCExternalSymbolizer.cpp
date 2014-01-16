//===-- lib/MC/MCExternalSymbolizer.cpp - External symbolizer ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCExternalSymbolizer.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>

using namespace llvm;

// This function tries to add a symbolic operand in place of the immediate
// Value in the MCInst. The immediate Value has had any PC adjustment made by
// the caller. If the instruction is a branch instruction then IsBranch is true,
// else false. If the getOpInfo() function was set as part of the
// setupForSymbolicDisassembly() call then that function is called to get any
// symbolic information at the Address for this instruction. If that returns
// non-zero then the symbolic information it returns is used to create an MCExpr
// and that is added as an operand to the MCInst. If getOpInfo() returns zero
// and IsBranch is true then a symbol look up for Value is done and if a symbol
// is found an MCExpr is created with that, else an MCExpr with Value is
// created. This function returns true if it adds an operand to the MCInst and
// false otherwise.
bool MCExternalSymbolizer::tryAddingSymbolicOperand(MCInst &MI,
                                                    raw_ostream &cStream,
                                                    int64_t Value,
                                                    uint64_t Address,
                                                    bool IsBranch,
                                                    uint64_t Offset,
                                                    uint64_t InstSize) {
  struct LLVMOpInfo1 SymbolicOp;
  std::memset(&SymbolicOp, '\0', sizeof(struct LLVMOpInfo1));
  SymbolicOp.Value = Value;

  if (!GetOpInfo ||
      !GetOpInfo(DisInfo, Address, Offset, InstSize, 1, &SymbolicOp)) {
    // Clear SymbolicOp.Value from above and also all other fields.
    std::memset(&SymbolicOp, '\0', sizeof(struct LLVMOpInfo1));
    if (!SymbolLookUp)
      return false;
    uint64_t ReferenceType;
    if (IsBranch)
       ReferenceType = LLVMDisassembler_ReferenceType_In_Branch;
    else
       ReferenceType = LLVMDisassembler_ReferenceType_InOut_None;
    const char *ReferenceName;
    const char *Name = SymbolLookUp(DisInfo, Value, &ReferenceType, Address,
                                    &ReferenceName);
    if (Name) {
      SymbolicOp.AddSymbol.Name = Name;
      SymbolicOp.AddSymbol.Present = true;
      // If Name is a C++ symbol name put the human readable name in a comment.
      if(ReferenceType == LLVMDisassembler_ReferenceType_DeMangled_Name)
        cStream << ReferenceName;
    }
    // For branches always create an MCExpr so it gets printed as hex address.
    else if (IsBranch) {
      SymbolicOp.Value = Value;
    }
    if(ReferenceType == LLVMDisassembler_ReferenceType_Out_SymbolStub)
      cStream << "symbol stub for: " << ReferenceName;
    else if(ReferenceType == LLVMDisassembler_ReferenceType_Out_Objc_Message)
      cStream << "Objc message: " << ReferenceName;
    if (!Name && !IsBranch)
      return false;
  }

  const MCExpr *Add = NULL;
  if (SymbolicOp.AddSymbol.Present) {
    if (SymbolicOp.AddSymbol.Name) {
      StringRef Name(SymbolicOp.AddSymbol.Name);
      MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name);
      Add = MCSymbolRefExpr::Create(Sym, Ctx);
    } else {
      Add = MCConstantExpr::Create((int)SymbolicOp.AddSymbol.Value, Ctx);
    }
  }

  const MCExpr *Sub = NULL;
  if (SymbolicOp.SubtractSymbol.Present) {
      if (SymbolicOp.SubtractSymbol.Name) {
      StringRef Name(SymbolicOp.SubtractSymbol.Name);
      MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name);
      Sub = MCSymbolRefExpr::Create(Sym, Ctx);
    } else {
      Sub = MCConstantExpr::Create((int)SymbolicOp.SubtractSymbol.Value, Ctx);
    }
  }

  const MCExpr *Off = NULL;
  if (SymbolicOp.Value != 0)
    Off = MCConstantExpr::Create(SymbolicOp.Value, Ctx);

  const MCExpr *Expr;
  if (Sub) {
    const MCExpr *LHS;
    if (Add)
      LHS = MCBinaryExpr::CreateSub(Add, Sub, Ctx);
    else
      LHS = MCUnaryExpr::CreateMinus(Sub, Ctx);
    if (Off != 0)
      Expr = MCBinaryExpr::CreateAdd(LHS, Off, Ctx);
    else
      Expr = LHS;
  } else if (Add) {
    if (Off != 0)
      Expr = MCBinaryExpr::CreateAdd(Add, Off, Ctx);
    else
      Expr = Add;
  } else {
    if (Off != 0)
      Expr = Off;
    else
      Expr = MCConstantExpr::Create(0, Ctx);
  }

  Expr = RelInfo->createExprForCAPIVariantKind(Expr, SymbolicOp.VariantKind);
  if (!Expr)
    return false;

  MI.addOperand(MCOperand::CreateExpr(Expr));
  return true;
}

// This function tries to add a comment as to what is being referenced by a load
// instruction with the base register that is the Pc.  These can often be values
// in a literal pool near the Address of the instruction. The Address of the
// instruction and its immediate Value are used as a possible literal pool entry.
// The SymbolLookUp call back will return the name of a symbol referenced by the
// literal pool's entry if the referenced address is that of a symbol. Or it
// will return a pointer to a literal 'C' string if the referenced address of
// the literal pool's entry is an address into a section with C string literals.
// Or if the reference is to an Objective-C data structure it will return a
// specific reference type for it and a string.
void MCExternalSymbolizer::tryAddingPcLoadReferenceComment(raw_ostream &cStream,
                                                           int64_t Value,
                                                           uint64_t Address) {
  if (SymbolLookUp) {
    uint64_t ReferenceType = LLVMDisassembler_ReferenceType_In_PCrel_Load;
    const char *ReferenceName;
    (void)SymbolLookUp(DisInfo, Value, &ReferenceType, Address, &ReferenceName);
    if(ReferenceType == LLVMDisassembler_ReferenceType_Out_LitPool_SymAddr)
      cStream << "literal pool symbol address: " << ReferenceName;
    else if(ReferenceType ==
            LLVMDisassembler_ReferenceType_Out_LitPool_CstrAddr) {
      cStream << "literal pool for: \"";
      cStream.write_escaped(ReferenceName);
      cStream << "\"";
    }
    else if(ReferenceType ==
            LLVMDisassembler_ReferenceType_Out_Objc_CFString_Ref)
      cStream << "Objc cfstring ref: @\"" << ReferenceName << "\"";
    else if(ReferenceType ==
            LLVMDisassembler_ReferenceType_Out_Objc_Message)
      cStream << "Objc message: " << ReferenceName;
    else if(ReferenceType ==
            LLVMDisassembler_ReferenceType_Out_Objc_Message_Ref)
      cStream << "Objc message ref: " << ReferenceName;
    else if(ReferenceType ==
            LLVMDisassembler_ReferenceType_Out_Objc_Selector_Ref)
      cStream << "Objc selector ref: " << ReferenceName;
    else if(ReferenceType ==
            LLVMDisassembler_ReferenceType_Out_Objc_Class_Ref)
      cStream << "Objc class ref: " << ReferenceName;
  }
}

namespace llvm {
MCSymbolizer *createMCSymbolizer(StringRef TT, LLVMOpInfoCallback GetOpInfo,
                                 LLVMSymbolLookupCallback SymbolLookUp,
                                 void *DisInfo,
                                 MCContext *Ctx,
                                 MCRelocationInfo *RelInfo) {
  assert(Ctx != 0 && "No MCContext given for symbolic disassembly");

  OwningPtr<MCRelocationInfo> RelInfoOwingPtr(RelInfo);
  return new MCExternalSymbolizer(*Ctx, RelInfoOwingPtr, GetOpInfo,
                                  SymbolLookUp, DisInfo);
}
}
