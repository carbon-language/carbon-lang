//===-- NVPTXReplaceImageHandles.cpp - Replace image handles for Fermi ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// On Fermi, image handles are not supported. To work around this, we traverse
// the machine code and replace image handles with concrete symbols. For this
// to work reliably, inlining of all function call must be performed.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"

using namespace llvm;

namespace {
class NVPTXReplaceImageHandles : public MachineFunctionPass {
private:
  static char ID;
  DenseSet<MachineInstr *> InstrsToRemove;

public:
  NVPTXReplaceImageHandles();

  bool runOnMachineFunction(MachineFunction &MF);
private:
  bool processInstr(MachineInstr &MI);
  void replaceImageHandle(MachineOperand &Op, MachineFunction &MF);
};
}

char NVPTXReplaceImageHandles::ID = 0;

NVPTXReplaceImageHandles::NVPTXReplaceImageHandles()
  : MachineFunctionPass(ID) {}

bool NVPTXReplaceImageHandles::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  InstrsToRemove.clear();

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end(); BI != BE;
       ++BI) {
    for (MachineBasicBlock::iterator I = (*BI).begin(), E = (*BI).end();
         I != E; ++I) {
      MachineInstr &MI = *I;
      Changed |= processInstr(MI);
    }
  }

  // Now clean up any handle-access instructions
  // This is needed in debug mode when code cleanup passes are not executed,
  // but we need the handle access to be eliminated because they are not
  // valid instructions when image handles are disabled.
  for (DenseSet<MachineInstr *>::iterator I = InstrsToRemove.begin(),
       E = InstrsToRemove.end(); I != E; ++I) {
    (*I)->eraseFromParent();
  }

  return Changed;
}

bool NVPTXReplaceImageHandles::processInstr(MachineInstr &MI) {
  MachineFunction &MF = *MI.getParent()->getParent();
  // Check if we have a surface/texture instruction
  switch (MI.getOpcode()) {
  default: return false;
  case NVPTX::TEX_1D_F32_I32:
  case NVPTX::TEX_1D_F32_F32:
  case NVPTX::TEX_1D_F32_F32_LEVEL:
  case NVPTX::TEX_1D_F32_F32_GRAD:
  case NVPTX::TEX_1D_I32_I32:
  case NVPTX::TEX_1D_I32_F32:
  case NVPTX::TEX_1D_I32_F32_LEVEL:
  case NVPTX::TEX_1D_I32_F32_GRAD:
  case NVPTX::TEX_1D_ARRAY_F32_I32:
  case NVPTX::TEX_1D_ARRAY_F32_F32:
  case NVPTX::TEX_1D_ARRAY_F32_F32_LEVEL:
  case NVPTX::TEX_1D_ARRAY_F32_F32_GRAD:
  case NVPTX::TEX_1D_ARRAY_I32_I32:
  case NVPTX::TEX_1D_ARRAY_I32_F32:
  case NVPTX::TEX_1D_ARRAY_I32_F32_LEVEL:
  case NVPTX::TEX_1D_ARRAY_I32_F32_GRAD:
  case NVPTX::TEX_2D_F32_I32:
  case NVPTX::TEX_2D_F32_F32:
  case NVPTX::TEX_2D_F32_F32_LEVEL:
  case NVPTX::TEX_2D_F32_F32_GRAD:
  case NVPTX::TEX_2D_I32_I32:
  case NVPTX::TEX_2D_I32_F32:
  case NVPTX::TEX_2D_I32_F32_LEVEL:
  case NVPTX::TEX_2D_I32_F32_GRAD:
  case NVPTX::TEX_2D_ARRAY_F32_I32:
  case NVPTX::TEX_2D_ARRAY_F32_F32:
  case NVPTX::TEX_2D_ARRAY_F32_F32_LEVEL:
  case NVPTX::TEX_2D_ARRAY_F32_F32_GRAD:
  case NVPTX::TEX_2D_ARRAY_I32_I32:
  case NVPTX::TEX_2D_ARRAY_I32_F32:
  case NVPTX::TEX_2D_ARRAY_I32_F32_LEVEL:
  case NVPTX::TEX_2D_ARRAY_I32_F32_GRAD:
  case NVPTX::TEX_3D_F32_I32:
  case NVPTX::TEX_3D_F32_F32:
  case NVPTX::TEX_3D_F32_F32_LEVEL:
  case NVPTX::TEX_3D_F32_F32_GRAD:
  case NVPTX::TEX_3D_I32_I32:
  case NVPTX::TEX_3D_I32_F32:
  case NVPTX::TEX_3D_I32_F32_LEVEL:
  case NVPTX::TEX_3D_I32_F32_GRAD: {
    // This is a texture fetch, so operand 4 is a texref and operand 5 is
    // a samplerref
    MachineOperand &TexHandle = MI.getOperand(4);
    MachineOperand &SampHandle = MI.getOperand(5);

    replaceImageHandle(TexHandle, MF);
    replaceImageHandle(SampHandle, MF);

    return true;
  }
  case NVPTX::SULD_1D_I8_TRAP:
  case NVPTX::SULD_1D_I16_TRAP:
  case NVPTX::SULD_1D_I32_TRAP:
  case NVPTX::SULD_1D_ARRAY_I8_TRAP:
  case NVPTX::SULD_1D_ARRAY_I16_TRAP:
  case NVPTX::SULD_1D_ARRAY_I32_TRAP:
  case NVPTX::SULD_2D_I8_TRAP:
  case NVPTX::SULD_2D_I16_TRAP:
  case NVPTX::SULD_2D_I32_TRAP:
  case NVPTX::SULD_2D_ARRAY_I8_TRAP:
  case NVPTX::SULD_2D_ARRAY_I16_TRAP:
  case NVPTX::SULD_2D_ARRAY_I32_TRAP:
  case NVPTX::SULD_3D_I8_TRAP:
  case NVPTX::SULD_3D_I16_TRAP:
  case NVPTX::SULD_3D_I32_TRAP: {
    // This is a V1 surface load, so operand 1 is a surfref
    MachineOperand &SurfHandle = MI.getOperand(1);

    replaceImageHandle(SurfHandle, MF);

    return true;
  }
  case NVPTX::SULD_1D_V2I8_TRAP:
  case NVPTX::SULD_1D_V2I16_TRAP:
  case NVPTX::SULD_1D_V2I32_TRAP:
  case NVPTX::SULD_1D_ARRAY_V2I8_TRAP:
  case NVPTX::SULD_1D_ARRAY_V2I16_TRAP:
  case NVPTX::SULD_1D_ARRAY_V2I32_TRAP:
  case NVPTX::SULD_2D_V2I8_TRAP:
  case NVPTX::SULD_2D_V2I16_TRAP:
  case NVPTX::SULD_2D_V2I32_TRAP:
  case NVPTX::SULD_2D_ARRAY_V2I8_TRAP:
  case NVPTX::SULD_2D_ARRAY_V2I16_TRAP:
  case NVPTX::SULD_2D_ARRAY_V2I32_TRAP:
  case NVPTX::SULD_3D_V2I8_TRAP:
  case NVPTX::SULD_3D_V2I16_TRAP:
  case NVPTX::SULD_3D_V2I32_TRAP: {
    // This is a V2 surface load, so operand 2 is a surfref
    MachineOperand &SurfHandle = MI.getOperand(2);

    replaceImageHandle(SurfHandle, MF);

    return true;
  }
  case NVPTX::SULD_1D_V4I8_TRAP:
  case NVPTX::SULD_1D_V4I16_TRAP:
  case NVPTX::SULD_1D_V4I32_TRAP:
  case NVPTX::SULD_1D_ARRAY_V4I8_TRAP:
  case NVPTX::SULD_1D_ARRAY_V4I16_TRAP:
  case NVPTX::SULD_1D_ARRAY_V4I32_TRAP:
  case NVPTX::SULD_2D_V4I8_TRAP:
  case NVPTX::SULD_2D_V4I16_TRAP:
  case NVPTX::SULD_2D_V4I32_TRAP:
  case NVPTX::SULD_2D_ARRAY_V4I8_TRAP:
  case NVPTX::SULD_2D_ARRAY_V4I16_TRAP:
  case NVPTX::SULD_2D_ARRAY_V4I32_TRAP:
  case NVPTX::SULD_3D_V4I8_TRAP:
  case NVPTX::SULD_3D_V4I16_TRAP:
  case NVPTX::SULD_3D_V4I32_TRAP: {
    // This is a V4 surface load, so operand 4 is a surfref
    MachineOperand &SurfHandle = MI.getOperand(4);

    replaceImageHandle(SurfHandle, MF);

    return true;
  }
  case NVPTX::SUST_B_1D_B8_TRAP:
  case NVPTX::SUST_B_1D_B16_TRAP:
  case NVPTX::SUST_B_1D_B32_TRAP:
  case NVPTX::SUST_B_1D_V2B8_TRAP:
  case NVPTX::SUST_B_1D_V2B16_TRAP:
  case NVPTX::SUST_B_1D_V2B32_TRAP:
  case NVPTX::SUST_B_1D_V4B8_TRAP:
  case NVPTX::SUST_B_1D_V4B16_TRAP:
  case NVPTX::SUST_B_1D_V4B32_TRAP:
  case NVPTX::SUST_B_1D_ARRAY_B8_TRAP:
  case NVPTX::SUST_B_1D_ARRAY_B16_TRAP:
  case NVPTX::SUST_B_1D_ARRAY_B32_TRAP:
  case NVPTX::SUST_B_1D_ARRAY_V2B8_TRAP:
  case NVPTX::SUST_B_1D_ARRAY_V2B16_TRAP:
  case NVPTX::SUST_B_1D_ARRAY_V2B32_TRAP:
  case NVPTX::SUST_B_1D_ARRAY_V4B8_TRAP:
  case NVPTX::SUST_B_1D_ARRAY_V4B16_TRAP:
  case NVPTX::SUST_B_1D_ARRAY_V4B32_TRAP:
  case NVPTX::SUST_B_2D_B8_TRAP:
  case NVPTX::SUST_B_2D_B16_TRAP:
  case NVPTX::SUST_B_2D_B32_TRAP:
  case NVPTX::SUST_B_2D_V2B8_TRAP:
  case NVPTX::SUST_B_2D_V2B16_TRAP:
  case NVPTX::SUST_B_2D_V2B32_TRAP:
  case NVPTX::SUST_B_2D_V4B8_TRAP:
  case NVPTX::SUST_B_2D_V4B16_TRAP:
  case NVPTX::SUST_B_2D_V4B32_TRAP:
  case NVPTX::SUST_B_2D_ARRAY_B8_TRAP:
  case NVPTX::SUST_B_2D_ARRAY_B16_TRAP:
  case NVPTX::SUST_B_2D_ARRAY_B32_TRAP:
  case NVPTX::SUST_B_2D_ARRAY_V2B8_TRAP:
  case NVPTX::SUST_B_2D_ARRAY_V2B16_TRAP:
  case NVPTX::SUST_B_2D_ARRAY_V2B32_TRAP:
  case NVPTX::SUST_B_2D_ARRAY_V4B8_TRAP:
  case NVPTX::SUST_B_2D_ARRAY_V4B16_TRAP:
  case NVPTX::SUST_B_2D_ARRAY_V4B32_TRAP:
  case NVPTX::SUST_B_3D_B8_TRAP:
  case NVPTX::SUST_B_3D_B16_TRAP:
  case NVPTX::SUST_B_3D_B32_TRAP:
  case NVPTX::SUST_B_3D_V2B8_TRAP:
  case NVPTX::SUST_B_3D_V2B16_TRAP:
  case NVPTX::SUST_B_3D_V2B32_TRAP:
  case NVPTX::SUST_B_3D_V4B8_TRAP:
  case NVPTX::SUST_B_3D_V4B16_TRAP:
  case NVPTX::SUST_B_3D_V4B32_TRAP:
  case NVPTX::SUST_P_1D_B8_TRAP:
  case NVPTX::SUST_P_1D_B16_TRAP:
  case NVPTX::SUST_P_1D_B32_TRAP:
  case NVPTX::SUST_P_1D_V2B8_TRAP:
  case NVPTX::SUST_P_1D_V2B16_TRAP:
  case NVPTX::SUST_P_1D_V2B32_TRAP:
  case NVPTX::SUST_P_1D_V4B8_TRAP:
  case NVPTX::SUST_P_1D_V4B16_TRAP:
  case NVPTX::SUST_P_1D_V4B32_TRAP:
  case NVPTX::SUST_P_1D_ARRAY_B8_TRAP:
  case NVPTX::SUST_P_1D_ARRAY_B16_TRAP:
  case NVPTX::SUST_P_1D_ARRAY_B32_TRAP:
  case NVPTX::SUST_P_1D_ARRAY_V2B8_TRAP:
  case NVPTX::SUST_P_1D_ARRAY_V2B16_TRAP:
  case NVPTX::SUST_P_1D_ARRAY_V2B32_TRAP:
  case NVPTX::SUST_P_1D_ARRAY_V4B8_TRAP:
  case NVPTX::SUST_P_1D_ARRAY_V4B16_TRAP:
  case NVPTX::SUST_P_1D_ARRAY_V4B32_TRAP:
  case NVPTX::SUST_P_2D_B8_TRAP:
  case NVPTX::SUST_P_2D_B16_TRAP:
  case NVPTX::SUST_P_2D_B32_TRAP:
  case NVPTX::SUST_P_2D_V2B8_TRAP:
  case NVPTX::SUST_P_2D_V2B16_TRAP:
  case NVPTX::SUST_P_2D_V2B32_TRAP:
  case NVPTX::SUST_P_2D_V4B8_TRAP:
  case NVPTX::SUST_P_2D_V4B16_TRAP:
  case NVPTX::SUST_P_2D_V4B32_TRAP:
  case NVPTX::SUST_P_2D_ARRAY_B8_TRAP:
  case NVPTX::SUST_P_2D_ARRAY_B16_TRAP:
  case NVPTX::SUST_P_2D_ARRAY_B32_TRAP:
  case NVPTX::SUST_P_2D_ARRAY_V2B8_TRAP:
  case NVPTX::SUST_P_2D_ARRAY_V2B16_TRAP:
  case NVPTX::SUST_P_2D_ARRAY_V2B32_TRAP:
  case NVPTX::SUST_P_2D_ARRAY_V4B8_TRAP:
  case NVPTX::SUST_P_2D_ARRAY_V4B16_TRAP:
  case NVPTX::SUST_P_2D_ARRAY_V4B32_TRAP:
  case NVPTX::SUST_P_3D_B8_TRAP:
  case NVPTX::SUST_P_3D_B16_TRAP:
  case NVPTX::SUST_P_3D_B32_TRAP:
  case NVPTX::SUST_P_3D_V2B8_TRAP:
  case NVPTX::SUST_P_3D_V2B16_TRAP:
  case NVPTX::SUST_P_3D_V2B32_TRAP:
  case NVPTX::SUST_P_3D_V4B8_TRAP:
  case NVPTX::SUST_P_3D_V4B16_TRAP:
  case NVPTX::SUST_P_3D_V4B32_TRAP: {
    // This is a surface store, so operand 0 is a surfref
    MachineOperand &SurfHandle = MI.getOperand(0);

    replaceImageHandle(SurfHandle, MF);

    return true;
  }
  case NVPTX::TXQ_CHANNEL_ORDER:
  case NVPTX::TXQ_CHANNEL_DATA_TYPE:
  case NVPTX::TXQ_WIDTH:
  case NVPTX::TXQ_HEIGHT:
  case NVPTX::TXQ_DEPTH:
  case NVPTX::TXQ_ARRAY_SIZE:
  case NVPTX::TXQ_NUM_SAMPLES:
  case NVPTX::TXQ_NUM_MIPMAP_LEVELS:
  case NVPTX::SUQ_CHANNEL_ORDER:
  case NVPTX::SUQ_CHANNEL_DATA_TYPE:
  case NVPTX::SUQ_WIDTH:
  case NVPTX::SUQ_HEIGHT:
  case NVPTX::SUQ_DEPTH:
  case NVPTX::SUQ_ARRAY_SIZE: {
    // This is a query, so operand 1 is a surfref/texref
    MachineOperand &Handle = MI.getOperand(1);

    replaceImageHandle(Handle, MF);

    return true; 
  }
  }
}

void NVPTXReplaceImageHandles::
replaceImageHandle(MachineOperand &Op, MachineFunction &MF) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  NVPTXMachineFunctionInfo *MFI = MF.getInfo<NVPTXMachineFunctionInfo>();
  // Which instruction defines the handle?
  MachineInstr *MI = MRI.getVRegDef(Op.getReg());
  assert(MI && "No def for image handle vreg?");
  MachineInstr &TexHandleDef = *MI;

  switch (TexHandleDef.getOpcode()) {
  case NVPTX::LD_i64_avar: {
    // The handle is a parameter value being loaded, replace with the
    // parameter symbol
    assert(TexHandleDef.getOperand(6).isSymbol() && "Load is not a symbol!");
    StringRef Sym = TexHandleDef.getOperand(6).getSymbolName();
    std::string ParamBaseName = MF.getName();
    ParamBaseName += "_param_";
    assert(Sym.startswith(ParamBaseName) && "Invalid symbol reference");
    unsigned Param = atoi(Sym.data()+ParamBaseName.size());
    std::string NewSym;
    raw_string_ostream NewSymStr(NewSym);
    NewSymStr << MF.getFunction()->getName() << "_param_" << Param;
    Op.ChangeToImmediate(
      MFI->getImageHandleSymbolIndex(NewSymStr.str().c_str()));
    InstrsToRemove.insert(&TexHandleDef);
    break;
  }
  case NVPTX::texsurf_handles: {
    // The handle is a global variable, replace with the global variable name
    assert(TexHandleDef.getOperand(1).isGlobal() && "Load is not a global!");
    const GlobalValue *GV = TexHandleDef.getOperand(1).getGlobal();
    assert(GV->hasName() && "Global sampler must be named!");
    Op.ChangeToImmediate(MFI->getImageHandleSymbolIndex(GV->getName().data()));
    InstrsToRemove.insert(&TexHandleDef);
    break;
  }
  default:
    llvm_unreachable("Unknown instruction operating on handle");
  }
}

MachineFunctionPass *llvm::createNVPTXReplaceImageHandlesPass() {
  return new NVPTXReplaceImageHandles();
}
