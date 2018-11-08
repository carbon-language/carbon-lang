//===-- Uops.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ROBSize.h"

#include "Assembler.h"
#include "BenchmarkRunner.h"
#include "MCInstrDescView.h"
#include "Target.h"

namespace llvm {
namespace exegesis {

ROBSizeSnippetGenerator::~ROBSizeSnippetGenerator() = default;

llvm::Expected<std::vector<CodeTemplate>>
ROBSizeSnippetGenerator::generateCodeTemplates(const Instruction &Instr) const {
  CodeTemplate CT;
  // const llvm::BitVector *ScratchSpaceAliasedRegs = nullptr;
  const auto &ET = State.getExegesisTarget();
  const auto &TM = State.getTargetMachine();

  CT.ScratchSpacePointerInReg =
      ET.getScratchMemoryRegister(TM.getTargetTriple());
  if (CT.ScratchSpacePointerInReg == 0)
    return llvm::make_error<BenchmarkFailure>(
        "Infeasible : target does not support memory instructions");
  // ScratchSpaceAliasedRegs =
  //      &State.getRATC().getRegister(CT.ScratchSpacePointerInReg).aliasedBits();

  const unsigned ECX = 50u; // FIXME: pick any available register.
  const unsigned EDX = 52u; // FIXME: pick any available register.
  CT.ScratchRegisterCopies.push_back(ECX);
  CT.ScratchRegisterCopies.push_back(EDX);

  /*
    const llvm::TargetInstrInfo *const TII =
    State.getSubtargetInfo().getInstrInfo(); MCInst NopInst;
    TII->getNoop(NopInst);
    */
  Instruction ChaseRegInst(State.getInstrInfo(), State.getRATC(), ET.getChaseRegOpcode());
  //errs() << ChaseRegInst.Variables.size() << "\n";
  assert(ChaseRegInst.Variables.size() >= 2 && "'mov reg, [reg]'' should have at least two variables");
  InstructionTemplate IT1(ChaseRegInst);
  IT1.getValueFor(ChaseRegInst.Variables[0]) = MCOperand::createReg(ECX);
  ET.fillMemoryOperands(IT1, ECX, 0);
  CT.Instructions.push_back(std::move(IT1));
  InstructionTemplate IT2(ChaseRegInst);
  IT2.getValueFor(ChaseRegInst.Variables[0]) = MCOperand::createReg(EDX);
  ET.fillMemoryOperands(IT2, EDX, 0);
  CT.Instructions.push_back(std::move(IT2));

  // const auto &ReservedRegisters = State.getRATC().reservedRegisters();
  // No tied variables, we pick random values for defs.
  llvm::BitVector Defs(State.getRegInfo().getNumRegs());
  CT.Info =
      "instruction has no tied variables picking Uses different from defs";
  // CT.Instructions.push_back(std::move(IT));
  return getSingleton(std::move(CT));
}

} // namespace exegesis
} // namespace llvm
