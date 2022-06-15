//===- bolt/Target/X86/X86MCSymbolizer.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "X86MCSymbolizer.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Core/Relocation.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"

#define DEBUG_TYPE "bolt-symbolizer"

namespace llvm {
namespace bolt {

X86MCSymbolizer::~X86MCSymbolizer() {}

bool X86MCSymbolizer::tryAddingSymbolicOperand(
    MCInst &Inst, raw_ostream &CStream, int64_t Value, uint64_t InstAddress,
    bool IsBranch, uint64_t ImmOffset, uint64_t ImmSize, uint64_t InstSize) {
  if (IsBranch)
    return false;

  // Ignore implicit operands.
  if (ImmSize == 0)
    return false;

  BinaryContext &BC = Function.getBinaryContext();
  MCContext *Ctx = BC.Ctx.get();

  if (BC.MIB->isBranch(Inst) || BC.MIB->isCall(Inst))
    return false;

  /// Add symbolic operand to the instruction with an optional addend.
  auto addOperand = [&](const MCSymbol *Symbol, uint64_t Addend) {
    const MCExpr *Expr = MCSymbolRefExpr::create(Symbol, *Ctx);
    if (Addend)
      Expr = MCBinaryExpr::createAdd(Expr, MCConstantExpr::create(Addend, *Ctx),
                                     *Ctx);
    Inst.addOperand(MCOperand::createExpr(Expr));
  };

  // Check if the operand being added is a displacement part of a compound
  // memory operand that uses PC-relative addressing. If it is, try to symbolize
  // it without relocations. Return true on success, false otherwise.
  auto processPCRelOperandNoRel = [&]() {
    const int MemOp = BC.MIB->getMemoryOperandNo(Inst);
    if (MemOp == -1)
      return false;

    const unsigned DispOp = MemOp + X86::AddrDisp;
    if (Inst.getNumOperands() != DispOp)
      return false;

    const MCOperand &Base = Inst.getOperand(MemOp + X86::AddrBaseReg);
    if (Base.getReg() != BC.MRI->getProgramCounter())
      return false;

    const MCOperand &Scale = Inst.getOperand(MemOp + X86::AddrScaleAmt);
    const MCOperand &Index = Inst.getOperand(MemOp + X86::AddrIndexReg);
    if (Scale.getImm() != 0 && Index.getReg() != MCRegister::NoRegister)
      return false;

    const MCSymbol *TargetSymbol;
    uint64_t TargetOffset;
    std::tie(TargetSymbol, TargetOffset) =
        BC.handleAddressRef(Value, Function, /*IsPCRel=*/true);

    addOperand(TargetSymbol, TargetOffset);

    return true;
  };

  // Check for GOTPCRELX relocations first. Because these relocations allow the
  // linker to modify the instruction, we have to check the offset range
  // corresponding to the instruction, not the offset of the operand.
  // Note that if there is GOTPCRELX relocation against the instruction, there
  // will be no other relocation in this range, since GOTPCRELX applies only to
  // certain instruction types.
  const uint64_t InstOffset = InstAddress - Function.getAddress();
  const Relocation *Relocation =
      Function.getRelocationInRange(InstOffset, InstOffset + InstSize);
  if (Relocation && Relocation::isX86GOTPCRELX(Relocation->Type)) {
    // If the operand is PC-relative, convert it without using the relocation
    // information. For GOTPCRELX, it is safe to use the absolute address
    // instead of extracting the addend from the relocation, as non-standard
    // forms will be rejected by linker conversion process and the operand
    // will always reference GOT which we don't rewrite.
    if (processPCRelOperandNoRel())
      return true;

    // The linker converted the PC-relative address to an absolute one.
    // Symbolize this address.
    BC.handleAddressRef(Value, Function, /*IsPCRel=*/false);
    const BinaryData *Target = BC.getBinaryDataAtAddress(Value);
    assert(Target &&
           "BinaryData should exist at converted GOTPCRELX destination");

    addOperand(Target->getSymbol(), /*Addend=*/0);

    return true;
  }

  // Check for relocations against the operand.
  if (!Relocation || Relocation->Offset != InstOffset + ImmOffset)
    Relocation = Function.getRelocationAt(InstOffset + ImmOffset);

  if (!Relocation)
    return processPCRelOperandNoRel();

  uint64_t SymbolValue = Relocation->Value - Relocation->Addend;
  if (Relocation->isPCRelative())
    SymbolValue += InstAddress + ImmOffset;

  // Process reference to the symbol.
  BC.handleAddressRef(SymbolValue, Function, Relocation->isPCRelative());

  uint64_t Addend = Relocation->Addend;
  // Real addend for pc-relative targets is adjusted with a delta from
  // the relocation placement to the next instruction.
  if (Relocation->isPCRelative())
    Addend += InstOffset + InstSize - Relocation->Offset;

  addOperand(Relocation->Symbol, Addend);

  return true;
}

void X86MCSymbolizer::tryAddingPcLoadReferenceComment(raw_ostream &CStream,
                                                      int64_t Value,
                                                      uint64_t Address) {}

} // namespace bolt
} // namespace llvm
