//===- AMDGPUMCInstLower.cpp - Lower AMDGPU MachineInstr to an MCInst -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Code to lower AMDGPU MachineInstrs to their corresponding MCInst.
//
//===----------------------------------------------------------------------===//
//

#include "AMDGPUMCInstLower.h"
#include "AMDGPUAsmPrinter.h"
#include "InstPrinter/AMDGPUInstPrinter.h"
#include "R600InstrInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/Constants.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include <algorithm>

using namespace llvm;

AMDGPUMCInstLower::AMDGPUMCInstLower(MCContext &ctx):
  Ctx(ctx)
{ }

void AMDGPUMCInstLower::lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumExplicitOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);

    MCOperand MCOp;
    switch (MO.getType()) {
    default:
      llvm_unreachable("unknown operand type");
    case MachineOperand::MO_FPImmediate: {
      const APFloat &FloatValue = MO.getFPImm()->getValueAPF();
      assert(&FloatValue.getSemantics() == &APFloat::IEEEsingle &&
             "Only floating point immediates are supported at the moment.");
      MCOp = MCOperand::CreateFPImm(FloatValue.convertToFloat());
      break;
    }
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::CreateImm(MO.getImm());
      break;
    case MachineOperand::MO_Register:
      MCOp = MCOperand::CreateReg(MO.getReg());
      break;
    case MachineOperand::MO_MachineBasicBlock:
      MCOp = MCOperand::CreateExpr(MCSymbolRefExpr::Create(
                                   MO.getMBB()->getSymbol(), Ctx));
    }
    OutMI.addOperand(MCOp);
  }
}

void AMDGPUAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  AMDGPUMCInstLower MCInstLowering(OutContext);

  if (MI->isBundle()) {
    const MachineBasicBlock *MBB = MI->getParent();
    MachineBasicBlock::const_instr_iterator I = MI;
    ++I;
    while (I != MBB->end() && I->isInsideBundle()) {
      EmitInstruction(I);
      ++I;
    }
  } else {
    MCInst TmpInst;
    MCInstLowering.lower(MI, TmpInst);
    OutStreamer.EmitInstruction(TmpInst);

    if (DisasmEnabled) {
      // Disassemble instruction/operands to text.
      DisasmLines.resize(DisasmLines.size() + 1);
      std::string &DisasmLine = DisasmLines.back();
      raw_string_ostream DisasmStream(DisasmLine);

      AMDGPUInstPrinter InstPrinter(*TM.getMCAsmInfo(), *TM.getInstrInfo(),
                                    *TM.getRegisterInfo());
      InstPrinter.printInst(&TmpInst, DisasmStream, StringRef());

      // Disassemble instruction/operands to hex representation.
      SmallVector<MCFixup, 4> Fixups;
      SmallVector<char, 16> CodeBytes;
      raw_svector_ostream CodeStream(CodeBytes);

      MCObjectStreamer &ObjStreamer = (MCObjectStreamer &)OutStreamer;
      MCCodeEmitter &InstEmitter = ObjStreamer.getAssembler().getEmitter();
      InstEmitter.EncodeInstruction(TmpInst, CodeStream, Fixups);
      CodeStream.flush();

      HexLines.resize(HexLines.size() + 1);
      std::string &HexLine = HexLines.back();
      raw_string_ostream HexStream(HexLine);

      for (size_t i = 0; i < CodeBytes.size(); i += 4) {
        unsigned int CodeDWord = *(unsigned int *)&CodeBytes[i];
        HexStream << format("%s%08X", (i > 0 ? " " : ""), CodeDWord);
      }

      DisasmStream.flush();
      DisasmLineMaxLen = std::max(DisasmLineMaxLen, DisasmLine.size());
    }
  }
}
