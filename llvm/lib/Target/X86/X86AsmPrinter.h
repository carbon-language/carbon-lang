//===-- X86AsmPrinter.h - Convert X86 LLVM code to Intel assembly ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file the shared super class printer that converts from our internal
// representation of machine-dependent LLVM code to Intel and AT&T format
// assembly language.  This printer is the output mechanism used by `llc'.
//
//===----------------------------------------------------------------------===//

#ifndef X86ASMPRINTER_H
#define X86ASMPRINTER_H

#include "X86.h"
#include "X86MachineFunctionInfo.h"
#include "X86TargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineDebugInfo.h"
#include "llvm/ADT/Statistic.h"
#include <set>


namespace llvm {

extern Statistic<> EmittedInsts;

// FIXME: Move this to CodeGen/AsmPrinter.h
namespace PICStyle {
  enum X86AsmPICStyle {
    Stub, GOT
  };
}

struct VISIBILITY_HIDDEN X86SharedAsmPrinter : public AsmPrinter {
  DwarfWriter DW;

  X86SharedAsmPrinter(std::ostream &O, X86TargetMachine &TM,
                      const TargetAsmInfo *T)
    : AsmPrinter(O, TM, T), DW(O, this, T), X86PICStyle(PICStyle::GOT) {
    Subtarget = &TM.getSubtarget<X86Subtarget>();
  }

  // We have to propagate some information about MachineFunction to
  // AsmPrinter. It's ok, when we're printing the function, since we have
  // access to MachineFunction and can get the appropriate MachineFunctionInfo.
  // Unfortunately, this is not possible when we're printing reference to
  // Function (e.g. calling it and so on). Even more, there is no way to get the
  // corresponding MachineFunctions: it can even be not created at all. That's
  // why we should use additional structure, when we're collecting all necessary
  // information.
  //
  // This structure is using e.g. for name decoration for stdcall & fastcall'ed
  // function, since we have to use arguments' size for decoration.
  typedef std::map<const Function*, X86FunctionInfo> FMFInfoMap;
  FMFInfoMap FunctionInfoMap;

  void decorateName(std::string& Name, const GlobalValue* GV);

  bool doInitialization(Module &M);
  bool doFinalization(Module &M);

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    if (Subtarget->isTargetDarwin() || Subtarget->isTargetELF()) {
      AU.addRequired<MachineDebugInfo>();
    }
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  PICStyle::X86AsmPICStyle X86PICStyle;
  
  const X86Subtarget *Subtarget;

  // Necessary for Darwin to print out the apprioriate types of linker stubs
  std::set<std::string> FnStubs, GVStubs, LinkOnceStubs;

  // Necessary for dllexport support
  std::set<std::string> DLLExportedFns, DLLExportedGVs;
  
  inline static bool isScale(const MachineOperand &MO) {
    return MO.isImmediate() &&
          (MO.getImmedValue() == 1 || MO.getImmedValue() == 2 ||
          MO.getImmedValue() == 4 || MO.getImmedValue() == 8);
  }

  inline static bool isMem(const MachineInstr *MI, unsigned Op) {
    if (MI->getOperand(Op).isFrameIndex()) return true;
    return Op+4 <= MI->getNumOperands() &&
      MI->getOperand(Op  ).isRegister() && isScale(MI->getOperand(Op+1)) &&
      MI->getOperand(Op+2).isRegister() &&
      (MI->getOperand(Op+3).isImmediate() ||
       MI->getOperand(Op+3).isGlobalAddress() ||
       MI->getOperand(Op+3).isConstantPoolIndex() ||
       MI->getOperand(Op+3).isJumpTableIndex());
  }
};

} // end namespace llvm

#endif
