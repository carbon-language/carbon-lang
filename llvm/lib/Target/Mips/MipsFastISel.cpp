//===-- MipsastISel.cpp - Mips FastISel implementation
//---------------------===//

#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/FastISel.h"

#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "MipsISelLowering.h"
#include "MipsMachineFunction.h"
#include "MipsSubtarget.h"

using namespace llvm;

namespace {

class MipsFastISel final : public FastISel {

  /// Subtarget - Keep a pointer to the MipsSubtarget around so that we can
  /// make the right decision when generating code for different targets.
  const MipsSubtarget *Subtarget;
  Module &M;
  const TargetMachine &TM;
  const TargetInstrInfo &TII;
  const TargetLowering &TLI;
  MipsFunctionInfo *MFI;

  // Convenience variables to avoid some queries.
  LLVMContext *Context;

  bool TargetSupported;

public:
  explicit MipsFastISel(FunctionLoweringInfo &funcInfo,
                        const TargetLibraryInfo *libInfo)
      : FastISel(funcInfo, libInfo),
        M(const_cast<Module &>(*funcInfo.Fn->getParent())),
        TM(funcInfo.MF->getTarget()), TII(*TM.getInstrInfo()),
        TLI(*TM.getTargetLowering()) {
    Subtarget = &TM.getSubtarget<MipsSubtarget>();
    MFI = funcInfo.MF->getInfo<MipsFunctionInfo>();
    Context = &funcInfo.Fn->getContext();
    TargetSupported = ((Subtarget->getRelocationModel() == Reloc::PIC_) &&
                       (Subtarget->hasMips32r2() && (Subtarget->isABI_O32())));
  }

  bool TargetSelectInstruction(const Instruction *I) override;

  bool SelectRet(const Instruction *I);
};

bool MipsFastISel::SelectRet(const Instruction *I) {
  const ReturnInst *Ret = cast<ReturnInst>(I);

  if (!FuncInfo.CanLowerReturn)
    return false;
  if (Ret->getNumOperands() > 0) {
    return false;
  }
  unsigned RetOpc = Mips::RetRA;
  BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DbgLoc, TII.get(RetOpc));
  return true;
}

bool MipsFastISel::TargetSelectInstruction(const Instruction *I) {
  if (!TargetSupported)
    return false;
  switch (I->getOpcode()) {
  default:
    break;
  case Instruction::Ret:
    return SelectRet(I);
  }
  return false;
}
}

namespace llvm {
FastISel *Mips::createFastISel(FunctionLoweringInfo &funcInfo,
                               const TargetLibraryInfo *libInfo) {
  return new MipsFastISel(funcInfo, libInfo);
}
}
