//===-- SystemZAsmPrinter.cpp - SystemZ LLVM assembly printer -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Streams SystemZ assembly language and associated data, in the form of
// MCInsts and MCExprs respectively.
//
//===----------------------------------------------------------------------===//

#include "SystemZAsmPrinter.h"
#include "InstPrinter/SystemZInstPrinter.h"
#include "SystemZConstantPoolValue.h"
#include "SystemZMCInstLower.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/Mangler.h"

using namespace llvm;

void SystemZAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  SystemZMCInstLower Lower(Mang, MF->getContext(), *this);
  MCInst LoweredMI;
  Lower.lower(MI, LoweredMI);
  OutStreamer.EmitInstruction(LoweredMI);
}

// Convert a SystemZ-specific constant pool modifier into the associated
// MCSymbolRefExpr variant kind.
static MCSymbolRefExpr::VariantKind
getModifierVariantKind(SystemZCP::SystemZCPModifier Modifier) {
  switch (Modifier) {
  case SystemZCP::NTPOFF: return MCSymbolRefExpr::VK_NTPOFF;
  }
  llvm_unreachable("Invalid SystemCPModifier!");
}

void SystemZAsmPrinter::
EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) {
  SystemZConstantPoolValue *ZCPV =
    static_cast<SystemZConstantPoolValue*>(MCPV);

  const MCExpr *Expr =
    MCSymbolRefExpr::Create(Mang->getSymbol(ZCPV->getGlobalValue()),
                            getModifierVariantKind(ZCPV->getModifier()),
                            OutContext);
  uint64_t Size = TM.getDataLayout()->getTypeAllocSize(ZCPV->getType());

  OutStreamer.EmitValue(Expr, Size);
}

bool SystemZAsmPrinter::PrintAsmOperand(const MachineInstr *MI,
                                        unsigned OpNo,
                                        unsigned AsmVariant,
                                        const char *ExtraCode,
                                        raw_ostream &OS) {
  if (ExtraCode && *ExtraCode == 'n') {
    if (!MI->getOperand(OpNo).isImm())
      return true;
    OS << -int64_t(MI->getOperand(OpNo).getImm());
  } else {
    SystemZMCInstLower Lower(Mang, MF->getContext(), *this);
    MCOperand MO(Lower.lowerOperand(MI->getOperand(OpNo)));
    SystemZInstPrinter::printOperand(MO, OS);
  }
  return false;
}

bool SystemZAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                              unsigned OpNo,
                                              unsigned AsmVariant,
                                              const char *ExtraCode,
                                              raw_ostream &OS) {
  SystemZInstPrinter::printAddress(MI->getOperand(OpNo).getReg(),
                                   MI->getOperand(OpNo + 1).getImm(),
                                   MI->getOperand(OpNo + 2).getReg(), OS);
  return false;
}

void SystemZAsmPrinter::EmitEndOfAsmFile(Module &M) {
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

// Force static initialization.
extern "C" void LLVMInitializeSystemZAsmPrinter() {
  RegisterAsmPrinter<SystemZAsmPrinter> X(TheSystemZTarget);
}
