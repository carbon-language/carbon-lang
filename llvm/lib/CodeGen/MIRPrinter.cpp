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

public:
  MIRPrinter(raw_ostream &OS) : OS(OS) {}

  void print(const MachineFunction &MF);

  void convert(yaml::MachineBasicBlock &YamlMBB, const MachineBasicBlock &MBB);
};

/// This class prints out the machine instructions using the MIR serialization
/// format.
class MIPrinter {
  raw_ostream &OS;

public:
  MIPrinter(raw_ostream &OS) : OS(OS) {}

  void print(const MachineInstr &MI);
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
  yaml::MachineFunction YamlMF;
  YamlMF.Name = MF.getName();
  YamlMF.Alignment = MF.getAlignment();
  YamlMF.ExposesReturnsTwice = MF.exposesReturnsTwice();
  YamlMF.HasInlineAsm = MF.hasInlineAsm();
  for (const auto &MBB : MF) {
    yaml::MachineBasicBlock YamlMBB;
    convert(YamlMBB, MBB);
    YamlMF.BasicBlocks.push_back(YamlMBB);
  }
  yaml::Output Out(OS);
  Out << YamlMF;
}

void MIRPrinter::convert(yaml::MachineBasicBlock &YamlMBB,
                         const MachineBasicBlock &MBB) {
  // TODO: Serialize unnamed BB references.
  if (const auto *BB = MBB.getBasicBlock())
    YamlMBB.Name = BB->hasName() ? BB->getName() : "<unnamed bb>";
  else
    YamlMBB.Name = "";
  YamlMBB.Alignment = MBB.getAlignment();
  YamlMBB.AddressTaken = MBB.hasAddressTaken();
  YamlMBB.IsLandingPad = MBB.isLandingPad();

  // Print the machine instructions.
  YamlMBB.Instructions.reserve(MBB.size());
  std::string Str;
  for (const auto &MI : MBB) {
    raw_string_ostream StrOS(Str);
    MIPrinter(StrOS).print(MI);
    YamlMBB.Instructions.push_back(StrOS.str());
    Str.clear();
  }
}

void MIPrinter::print(const MachineInstr &MI) {
  const auto &SubTarget = MI.getParent()->getParent()->getSubtarget();
  const auto *TII = SubTarget.getInstrInfo();
  assert(TII && "Expected target instruction info");

  OS << TII->getName(MI.getOpcode());
  // TODO: Print the instruction flags, machine operands, machine mem operands.
}

void llvm::printMIR(raw_ostream &OS, const Module &M) {
  yaml::Output Out(OS);
  Out << const_cast<Module &>(M);
}

void llvm::printMIR(raw_ostream &OS, const MachineFunction &MF) {
  MIRPrinter Printer(OS);
  Printer.print(MF);
}
