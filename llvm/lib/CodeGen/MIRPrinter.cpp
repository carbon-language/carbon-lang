//===- MIRPrinter.cpp - MIR serialization format printer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the class that prints out the LLVM IR and machine
// functions using the MIR serialization format.
//
//===----------------------------------------------------------------------===//

#include "MIRPrinter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

namespace {

/// This class prints out the machine functions using the MIR serialization
/// format.
class MIRPrinter {
  raw_ostream &OS;
  DenseMap<const uint32_t *, unsigned> RegisterMaskIds;

public:
  MIRPrinter(raw_ostream &OS) : OS(OS) {}

  void print(const MachineFunction &MF);

  void convert(yaml::MachineFunction &MF, const MachineRegisterInfo &RegInfo);
  void convert(const Module &M, yaml::MachineBasicBlock &YamlMBB,
               const MachineBasicBlock &MBB);

private:
  void initRegisterMaskIds(const MachineFunction &MF);
};

/// This class prints out the machine instructions using the MIR serialization
/// format.
class MIPrinter {
  const Module &M;
  raw_ostream &OS;
  const DenseMap<const uint32_t *, unsigned> &RegisterMaskIds;

public:
  MIPrinter(const Module &M, raw_ostream &OS,
            const DenseMap<const uint32_t *, unsigned> &RegisterMaskIds)
      : M(M), OS(OS), RegisterMaskIds(RegisterMaskIds) {}

  void print(const MachineInstr &MI);
  void printMBBReference(const MachineBasicBlock &MBB);
  void print(const MachineOperand &Op, const TargetRegisterInfo *TRI);
};

} // end anonymous namespace

namespace llvm {
namespace yaml {

/// This struct serializes the LLVM IR module.
template <> struct BlockScalarTraits<Module> {
  static void output(const Module &Mod, void *Ctxt, raw_ostream &OS) {
    Mod.print(OS, nullptr);
  }
  static StringRef input(StringRef Str, void *Ctxt, Module &Mod) {
    llvm_unreachable("LLVM Module is supposed to be parsed separately");
    return "";
  }
};

} // end namespace yaml
} // end namespace llvm

void MIRPrinter::print(const MachineFunction &MF) {
  initRegisterMaskIds(MF);

  yaml::MachineFunction YamlMF;
  YamlMF.Name = MF.getName();
  YamlMF.Alignment = MF.getAlignment();
  YamlMF.ExposesReturnsTwice = MF.exposesReturnsTwice();
  YamlMF.HasInlineAsm = MF.hasInlineAsm();
  convert(YamlMF, MF.getRegInfo());

  int I = 0;
  const auto &M = *MF.getFunction()->getParent();
  for (const auto &MBB : MF) {
    // TODO: Allow printing of non sequentially numbered MBBs.
    // This is currently needed as the basic block references get their index
    // from MBB.getNumber(), thus it should be sequential so that the parser can
    // map back to the correct MBBs when parsing the output.
    assert(MBB.getNumber() == I++ &&
           "Can't print MBBs that aren't sequentially numbered");
    (void)I;
    yaml::MachineBasicBlock YamlMBB;
    convert(M, YamlMBB, MBB);
    YamlMF.BasicBlocks.push_back(YamlMBB);
  }
  yaml::Output Out(OS);
  Out << YamlMF;
}

void MIRPrinter::convert(yaml::MachineFunction &MF,
                         const MachineRegisterInfo &RegInfo) {
  MF.IsSSA = RegInfo.isSSA();
  MF.TracksRegLiveness = RegInfo.tracksLiveness();
  MF.TracksSubRegLiveness = RegInfo.subRegLivenessEnabled();
}

void MIRPrinter::convert(const Module &M, yaml::MachineBasicBlock &YamlMBB,
                         const MachineBasicBlock &MBB) {
  assert(MBB.getNumber() >= 0 && "Invalid MBB number");
  YamlMBB.ID = (unsigned)MBB.getNumber();
  // TODO: Serialize unnamed BB references.
  if (const auto *BB = MBB.getBasicBlock())
    YamlMBB.Name = BB->hasName() ? BB->getName() : "<unnamed bb>";
  else
    YamlMBB.Name = "";
  YamlMBB.Alignment = MBB.getAlignment();
  YamlMBB.AddressTaken = MBB.hasAddressTaken();
  YamlMBB.IsLandingPad = MBB.isLandingPad();
  for (const auto *SuccMBB : MBB.successors()) {
    std::string Str;
    raw_string_ostream StrOS(Str);
    MIPrinter(M, StrOS, RegisterMaskIds).printMBBReference(*SuccMBB);
    YamlMBB.Successors.push_back(StrOS.str());
  }

  // Print the machine instructions.
  YamlMBB.Instructions.reserve(MBB.size());
  std::string Str;
  for (const auto &MI : MBB) {
    raw_string_ostream StrOS(Str);
    MIPrinter(M, StrOS, RegisterMaskIds).print(MI);
    YamlMBB.Instructions.push_back(StrOS.str());
    Str.clear();
  }
}

void MIRPrinter::initRegisterMaskIds(const MachineFunction &MF) {
  const auto *TRI = MF.getSubtarget().getRegisterInfo();
  unsigned I = 0;
  for (const uint32_t *Mask : TRI->getRegMasks())
    RegisterMaskIds.insert(std::make_pair(Mask, I++));
}

void MIPrinter::print(const MachineInstr &MI) {
  const auto &SubTarget = MI.getParent()->getParent()->getSubtarget();
  const auto *TRI = SubTarget.getRegisterInfo();
  assert(TRI && "Expected target register info");
  const auto *TII = SubTarget.getInstrInfo();
  assert(TII && "Expected target instruction info");

  unsigned I = 0, E = MI.getNumOperands();
  for (; I < E && MI.getOperand(I).isReg() && MI.getOperand(I).isDef() &&
         !MI.getOperand(I).isImplicit();
       ++I) {
    if (I)
      OS << ", ";
    print(MI.getOperand(I), TRI);
  }

  if (I)
    OS << " = ";
  OS << TII->getName(MI.getOpcode());
  // TODO: Print the instruction flags, machine mem operands.
  if (I < E)
    OS << ' ';

  bool NeedComma = false;
  for (; I < E; ++I) {
    if (NeedComma)
      OS << ", ";
    print(MI.getOperand(I), TRI);
    NeedComma = true;
  }
}

static void printReg(unsigned Reg, raw_ostream &OS,
                     const TargetRegisterInfo *TRI) {
  // TODO: Print Stack Slots.
  // TODO: Print virtual registers.
  if (!Reg)
    OS << '_';
  else if (Reg < TRI->getNumRegs())
    OS << '%' << StringRef(TRI->getName(Reg)).lower();
  else
    llvm_unreachable("Can't print this kind of register yet");
}

void MIPrinter::printMBBReference(const MachineBasicBlock &MBB) {
  OS << "%bb." << MBB.getNumber();
  if (const auto *BB = MBB.getBasicBlock()) {
    if (BB->hasName())
      OS << '.' << BB->getName();
  }
}

void MIPrinter::print(const MachineOperand &Op, const TargetRegisterInfo *TRI) {
  switch (Op.getType()) {
  case MachineOperand::MO_Register:
    // TODO: Print register flags.
    printReg(Op.getReg(), OS, TRI);
    // TODO: Print sub register.
    break;
  case MachineOperand::MO_Immediate:
    OS << Op.getImm();
    break;
  case MachineOperand::MO_MachineBasicBlock:
    printMBBReference(*Op.getMBB());
    break;
  case MachineOperand::MO_GlobalAddress:
    // FIXME: Make this faster - print as operand will create a slot tracker to
    // print unnamed values for the whole module every time it's called, which
    // is inefficient.
    Op.getGlobal()->printAsOperand(OS, /*PrintType=*/false, &M);
    // TODO: Print offset and target flags.
    break;
  case MachineOperand::MO_RegisterMask: {
    auto RegMaskInfo = RegisterMaskIds.find(Op.getRegMask());
    if (RegMaskInfo != RegisterMaskIds.end())
      OS << StringRef(TRI->getRegMaskNames()[RegMaskInfo->second]).lower();
    else
      llvm_unreachable("Can't print this machine register mask yet.");
    break;
  }
  default:
    // TODO: Print the other machine operands.
    llvm_unreachable("Can't print this machine operand at the moment");
  }
}

void llvm::printMIR(raw_ostream &OS, const Module &M) {
  yaml::Output Out(OS);
  Out << const_cast<Module &>(M);
}

void llvm::printMIR(raw_ostream &OS, const MachineFunction &MF) {
  MIRPrinter Printer(OS);
  Printer.print(MF);
}
