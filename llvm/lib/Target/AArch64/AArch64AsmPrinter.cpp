//===-- AArch64AsmPrinter.cpp - Print machine code to an AArch64 .s file --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format AArch64 assembly language.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "AArch64AsmPrinter.h"
#include "InstPrinter/AArch64InstPrinter.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

/// Try to print a floating-point register as if it belonged to a specified
/// register-class. For example the inline asm operand modifier "b" requires its
/// argument to be printed as "bN".
static bool printModifiedFPRAsmOperand(const MachineOperand &MO,
                                       const TargetRegisterInfo *TRI,
                                       char RegType, raw_ostream &O) {
  if (!MO.isReg())
    return true;

  for (MCRegAliasIterator AR(MO.getReg(), TRI, true); AR.isValid(); ++AR) {
    if (AArch64::FPR8RegClass.contains(*AR)) {
      O << RegType << TRI->getEncodingValue(MO.getReg());
      return false;
    }
  }

  // The register doesn't correspond to anything floating-point like.
  return true;
}

/// Implements the 'w' and 'x' inline asm operand modifiers, which print a GPR
/// with the obvious type and an immediate 0 as either wzr or xzr.
static bool printModifiedGPRAsmOperand(const MachineOperand &MO,
                                       const TargetRegisterInfo *TRI,
                                       const TargetRegisterClass &RegClass,
                                       raw_ostream &O) {
  char Prefix = &RegClass == &AArch64::GPR32RegClass ? 'w' : 'x';

  if (MO.isImm() && MO.getImm() == 0) {
    O << Prefix << "zr";
    return false;
  } else if (MO.isReg()) {
    if (MO.getReg() == AArch64::XSP || MO.getReg() == AArch64::WSP) {
      O << (Prefix == 'x' ? "sp" : "wsp");
      return false;
    }

    for (MCRegAliasIterator AR(MO.getReg(), TRI, true); AR.isValid(); ++AR) {
      if (RegClass.contains(*AR)) {
        O << AArch64InstPrinter::getRegisterName(*AR);
        return false;
      }
    }
  }

  return true;
}

bool AArch64AsmPrinter::printSymbolicAddress(const MachineOperand &MO,
                                             bool PrintImmediatePrefix,
                                             StringRef Suffix, raw_ostream &O) {
  StringRef Name;
  StringRef Modifier;
  switch (MO.getType()) {
  default:
    return true;
  case MachineOperand::MO_GlobalAddress:
    Name = getSymbol(MO.getGlobal())->getName();

    // Global variables may be accessed either via a GOT or in various fun and
    // interesting TLS-model specific ways. Set the prefix modifier as
    // appropriate here.
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(MO.getGlobal())) {
      Reloc::Model RelocM = TM.getRelocationModel();
      if (GV->isThreadLocal()) {
        switch (TM.getTLSModel(GV)) {
        case TLSModel::GeneralDynamic:
          Modifier = "tlsdesc";
          break;
        case TLSModel::LocalDynamic:
          Modifier = "dtprel";
          break;
        case TLSModel::InitialExec:
          Modifier = "gottprel";
          break;
        case TLSModel::LocalExec:
          Modifier = "tprel";
          break;
        }
      } else if (Subtarget->GVIsIndirectSymbol(GV, RelocM)) {
        Modifier = "got";
      }
    }
    break;
  case MachineOperand::MO_BlockAddress:
    Name = GetBlockAddressSymbol(MO.getBlockAddress())->getName();
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    Name = GetCPISymbol(MO.getIndex())->getName();
    break;
  }

  // Some instructions (notably ADRP) don't take the # prefix for
  // immediates. Only print it if asked to.
  if (PrintImmediatePrefix)
    O << '#';

  // Only need the joining "_" if both the prefix and the suffix are
  // non-null. This little block simply takes care of the four possibly
  // combinations involved there.
  if (Modifier == "" && Suffix == "")
    O << Name;
  else if (Modifier == "" && Suffix != "")
    O << ":" << Suffix << ':' << Name;
  else if (Modifier != "" && Suffix == "")
    O << ":" << Modifier << ':' << Name;
  else
    O << ":" << Modifier << '_' << Suffix << ':' << Name;

  return false;
}

bool AArch64AsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                                        unsigned AsmVariant,
                                        const char *ExtraCode, raw_ostream &O) {
  const TargetRegisterInfo *TRI = MF->getTarget().getRegisterInfo();

  if (!ExtraCode)
    ExtraCode = "";

  switch(ExtraCode[0]) {
  default:
    if (!AsmPrinter::PrintAsmOperand(MI, OpNum, AsmVariant, ExtraCode, O))
        return false;
    break;
  case 'w':
    // Output 32-bit general register operand, constant zero as wzr, or stack
    // pointer as wsp. Ignored when used with other operand types.
    if (!printModifiedGPRAsmOperand(MI->getOperand(OpNum), TRI,
                                    AArch64::GPR32RegClass, O))
      return false;
    break;
  case 'x':
    // Output 64-bit general register operand, constant zero as xzr, or stack
    // pointer as sp. Ignored when used with other operand types.
    if (!printModifiedGPRAsmOperand(MI->getOperand(OpNum), TRI,
                                    AArch64::GPR64RegClass, O))
      return false;
    break;
  case 'H':
    // Output higher numbered of a 64-bit general register pair
  case 'Q':
    // Output least significant register of a 64-bit general register pair
  case 'R':
    // Output most significant register of a 64-bit general register pair

    // FIXME note: these three operand modifiers will require, to some extent,
    // adding a paired GPR64 register class. Initial investigation suggests that
    // assertions are hit unless it has a type and is made legal for that type
    // in ISelLowering. After that step is made, the number of modifications
    // needed explodes (operation legality, calling conventions, stores, reg
    // copies ...).
    llvm_unreachable("FIXME: Unimplemented register pairs");
  case 'b':
  case 'h':
  case 's':
  case 'd':
  case 'q':
    if (!printModifiedFPRAsmOperand(MI->getOperand(OpNum), TRI,
                                    ExtraCode[0], O))
      return false;
    break;
  case 'A':
    // Output symbolic address with appropriate relocation modifier (also
    // suitable for ADRP).
    if (!printSymbolicAddress(MI->getOperand(OpNum), false, "", O))
      return false;
    break;
  case 'L':
    // Output bits 11:0 of symbolic address with appropriate :lo12: relocation
    // modifier.
    if (!printSymbolicAddress(MI->getOperand(OpNum), true, "lo12", O))
      return false;
    break;
  case 'G':
    // Output bits 23:12 of symbolic address with appropriate :hi12: relocation
    // modifier (currently only for TLS local exec).
    if (!printSymbolicAddress(MI->getOperand(OpNum), true, "hi12", O))
      return false;
    break;
  case 'a':
    return PrintAsmMemoryOperand(MI, OpNum, AsmVariant, ExtraCode, O);
  }

  // There's actually no operand modifier, which leads to a slightly eclectic
  // set of behaviour which we have to handle here.
  const MachineOperand &MO = MI->getOperand(OpNum);
  switch (MO.getType()) {
  default:
    llvm_unreachable("Unexpected operand for inline assembly");
  case MachineOperand::MO_Register:
    // GCC prints the unmodified operand of a 'w' constraint as the vector
    // register. Technically, we could allocate the argument as a VPR128, but
    // that leads to extremely dodgy copies being generated to get the data
    // there.
    if (printModifiedFPRAsmOperand(MO, TRI, 'v', O))
      O << AArch64InstPrinter::getRegisterName(MO.getReg());
    break;
  case MachineOperand::MO_Immediate:
    O << '#' << MO.getImm();
    break;
  case MachineOperand::MO_FPImmediate:
    assert(MO.getFPImm()->isExactlyValue(0.0) && "Only FP 0.0 expected");
    O << "#0.0";
    break;
  case MachineOperand::MO_BlockAddress:
  case MachineOperand::MO_ConstantPoolIndex:
  case MachineOperand::MO_GlobalAddress:
    return printSymbolicAddress(MO, false, "", O);
  }

  return false;
}

bool AArch64AsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                              unsigned OpNum,
                                              unsigned AsmVariant,
                                              const char *ExtraCode,
                                              raw_ostream &O) {
  // Currently both the memory constraints (m and Q) behave the same and amount
  // to the address as a single register. In future, we may allow "m" to provide
  // both a base and an offset.
  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isReg() && "unexpected inline assembly memory operand");
  O << '[' << AArch64InstPrinter::getRegisterName(MO.getReg()) << ']';
  return false;
}

#include "AArch64GenMCPseudoLowering.inc"

void AArch64AsmPrinter::EmitInstruction(const MachineInstr *MI) {
  // Do any auto-generated pseudo lowerings.
  if (emitPseudoExpansionLowering(OutStreamer, MI))
    return;

  MCInst TmpInst;
  LowerAArch64MachineInstrToMCInst(MI, TmpInst, *this);
  EmitToStreamer(OutStreamer, TmpInst);
}

void AArch64AsmPrinter::EmitEndOfAsmFile(Module &M) {
  if (Subtarget->isTargetELF()) {
    const TargetLoweringObjectFileELF &TLOFELF =
      static_cast<const TargetLoweringObjectFileELF &>(getObjFileLowering());

    MachineModuleInfoELF &MMIELF = MMI->getObjFileInfo<MachineModuleInfoELF>();

    // Output stubs for external and common global variables.
    MachineModuleInfoELF::SymbolListTy Stubs = MMIELF.GetGVStubList();
    if (!Stubs.empty()) {
      OutStreamer.SwitchSection(TLOFELF.getDataRelSection());
      const DataLayout *TD = TM.getDataLayout();

      for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
        OutStreamer.EmitLabel(Stubs[i].first);
        OutStreamer.EmitSymbolValue(Stubs[i].second.getPointer(),
                                    TD->getPointerSize(0));
      }
      Stubs.clear();
    }
  }
}

bool AArch64AsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  return AsmPrinter::runOnMachineFunction(MF);
}

// Force static initialization.
extern "C" void LLVMInitializeAArch64AsmPrinter() {
    RegisterAsmPrinter<AArch64AsmPrinter> X(TheAArch64leTarget);
    RegisterAsmPrinter<AArch64AsmPrinter> Y(TheAArch64beTarget);
}

