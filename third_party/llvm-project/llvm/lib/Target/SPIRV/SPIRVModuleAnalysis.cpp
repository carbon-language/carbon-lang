//===- SPIRVModuleAnalysis.cpp - analysis of global instrs & regs - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The analysis collects instructions that should be output at the module level
// and performs the global register numbering.
//
// The results of this analysis are used in AsmPrinter to rename registers
// globally and to output required instructions at the module level.
//
//===----------------------------------------------------------------------===//

#include "SPIRVModuleAnalysis.h"
#include "SPIRV.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "TargetInfo/SPIRVTargetInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"

using namespace llvm;

#define DEBUG_TYPE "spirv-module-analysis"

char llvm::SPIRVModuleAnalysis::ID = 0;

namespace llvm {
void initializeSPIRVModuleAnalysisPass(PassRegistry &);
} // namespace llvm

INITIALIZE_PASS(SPIRVModuleAnalysis, DEBUG_TYPE, "SPIRV module analysis", true,
                true)

// Retrieve an unsigned from an MDNode with a list of them as operands.
static unsigned getMetadataUInt(MDNode *MdNode, unsigned OpIndex,
                                unsigned DefaultVal = 0) {
  if (MdNode && OpIndex < MdNode->getNumOperands()) {
    const auto &Op = MdNode->getOperand(OpIndex);
    return mdconst::extract<ConstantInt>(Op)->getZExtValue();
  }
  return DefaultVal;
}

void SPIRVModuleAnalysis::setBaseInfo(const Module &M) {
  MAI.MaxID = 0;
  for (int i = 0; i < SPIRV::NUM_MODULE_SECTIONS; i++)
    MAI.MS[i].clear();
  MAI.RegisterAliasTable.clear();
  MAI.InstrsToDelete.clear();
  MAI.FuncNameMap.clear();
  MAI.GlobalVarList.clear();

  // TODO: determine memory model and source language from the configuratoin.
  MAI.Mem = SPIRV::MemoryModel::OpenCL;
  MAI.SrcLang = SPIRV::SourceLanguage::OpenCL_C;
  unsigned PtrSize = ST->getPointerSize();
  MAI.Addr = PtrSize == 32   ? SPIRV::AddressingModel::Physical32
             : PtrSize == 64 ? SPIRV::AddressingModel::Physical64
                             : SPIRV::AddressingModel::Logical;
  // Get the OpenCL version number from metadata.
  // TODO: support other source languages.
  MAI.SrcLangVersion = 0;
  if (auto VerNode = M.getNamedMetadata("opencl.ocl.version")) {
    // Construct version literal according to OpenCL 2.2 environment spec.
    auto VersionMD = VerNode->getOperand(0);
    unsigned MajorNum = getMetadataUInt(VersionMD, 0, 2);
    unsigned MinorNum = getMetadataUInt(VersionMD, 1);
    unsigned RevNum = getMetadataUInt(VersionMD, 2);
    MAI.SrcLangVersion = 0 | (MajorNum << 16) | (MinorNum << 8) | RevNum;
  }
}

// True if there is an instruction in the MS list with all the same operands as
// the given instruction has (after the given starting index).
// TODO: maybe it needs to check Opcodes too.
static bool findSameInstrInMS(const MachineInstr &A,
                              SPIRV::ModuleSectionType MSType,
                              SPIRV::ModuleAnalysisInfo &MAI,
                              bool UpdateRegAliases,
                              unsigned StartOpIndex = 0) {
  for (const auto *B : MAI.MS[MSType]) {
    const unsigned NumAOps = A.getNumOperands();
    if (NumAOps == B->getNumOperands() && A.getNumDefs() == B->getNumDefs()) {
      bool AllOpsMatch = true;
      for (unsigned i = StartOpIndex; i < NumAOps && AllOpsMatch; ++i) {
        if (A.getOperand(i).isReg() && B->getOperand(i).isReg()) {
          Register RegA = A.getOperand(i).getReg();
          Register RegB = B->getOperand(i).getReg();
          AllOpsMatch = MAI.getRegisterAlias(A.getMF(), RegA) ==
                        MAI.getRegisterAlias(B->getMF(), RegB);
        } else {
          AllOpsMatch = A.getOperand(i).isIdenticalTo(B->getOperand(i));
        }
      }
      if (AllOpsMatch) {
        if (UpdateRegAliases) {
          assert(A.getOperand(0).isReg() && B->getOperand(0).isReg());
          Register LocalReg = A.getOperand(0).getReg();
          Register GlobalReg =
              MAI.getRegisterAlias(B->getMF(), B->getOperand(0).getReg());
          MAI.setRegisterAlias(A.getMF(), LocalReg, GlobalReg);
        }
        return true;
      }
    }
  }
  return false;
}

// Look for IDs declared with Import linkage, and map the imported name string
// to the register defining that variable (which will usually be the result of
// an OpFunction). This lets us call externally imported functions using
// the correct ID registers.
void SPIRVModuleAnalysis::collectFuncNames(MachineInstr &MI,
                                           const Function &F) {
  if (MI.getOpcode() == SPIRV::OpDecorate) {
    // If it's got Import linkage.
    auto Dec = MI.getOperand(1).getImm();
    if (Dec == static_cast<unsigned>(SPIRV::Decoration::LinkageAttributes)) {
      auto Lnk = MI.getOperand(MI.getNumOperands() - 1).getImm();
      if (Lnk == static_cast<unsigned>(SPIRV::LinkageType::Import)) {
        // Map imported function name to function ID register.
        std::string Name = getStringImm(MI, 2);
        Register Target = MI.getOperand(0).getReg();
        // TODO: check defs from different MFs.
        MAI.FuncNameMap[Name] = MAI.getRegisterAlias(MI.getMF(), Target);
      }
    }
  } else if (MI.getOpcode() == SPIRV::OpFunction) {
    // Record all internal OpFunction declarations.
    Register Reg = MI.defs().begin()->getReg();
    Register GlobalReg = MAI.getRegisterAlias(MI.getMF(), Reg);
    assert(GlobalReg.isValid());
    // TODO: check that it does not conflict with existing entries.
    MAI.FuncNameMap[F.getGlobalIdentifier()] = GlobalReg;
  }
}

// Collect the given instruction in the specified MS. We assume global register
// numbering has already occurred by this point. We can directly compare reg
// arguments when detecting duplicates.
static void collectOtherInstr(MachineInstr &MI, SPIRV::ModuleAnalysisInfo &MAI,
                              SPIRV::ModuleSectionType MSType,
                              bool IsConstOrType = false) {
  MAI.setSkipEmission(&MI);
  if (findSameInstrInMS(MI, MSType, MAI, IsConstOrType, IsConstOrType ? 1 : 0))
    return; // Found a duplicate, so don't add it.
  // No duplicates, so add it.
  MAI.MS[MSType].push_back(&MI);
}

// Some global instructions make reference to function-local ID regs, so cannot
// be correctly collected until these registers are globally numbered.
void SPIRVModuleAnalysis::processOtherInstrs(const Module &M) {
  for (auto F = M.begin(), E = M.end(); F != E; ++F) {
    if ((*F).isDeclaration())
      continue;
    MachineFunction *MF = MMI->getMachineFunction(*F);
    assert(MF);
    unsigned FCounter = 0;
    for (MachineBasicBlock &MBB : *MF)
      for (MachineInstr &MI : MBB) {
        if (MI.getOpcode() == SPIRV::OpFunction)
          FCounter++;
        if (MAI.getSkipEmission(&MI))
          continue;
        const unsigned OpCode = MI.getOpcode();
        const bool IsFuncOrParm =
            OpCode == SPIRV::OpFunction || OpCode == SPIRV::OpFunctionParameter;
        const bool IsConstOrType =
            TII->isConstantInstr(MI) || TII->isTypeDeclInstr(MI);
        if (OpCode == SPIRV::OpName || OpCode == SPIRV::OpMemberName) {
          collectOtherInstr(MI, MAI, SPIRV::MB_DebugNames);
        } else if (OpCode == SPIRV::OpEntryPoint) {
          collectOtherInstr(MI, MAI, SPIRV::MB_EntryPoints);
        } else if (TII->isDecorationInstr(MI)) {
          collectOtherInstr(MI, MAI, SPIRV::MB_Annotations);
          collectFuncNames(MI, *F);
        } else if (IsConstOrType || (FCounter > 1 && IsFuncOrParm)) {
          // Now OpSpecConstant*s are not in DT,
          // but they need to be collected anyway.
          enum SPIRV::ModuleSectionType Type =
              IsFuncOrParm ? SPIRV::MB_ExtFuncDecls : SPIRV::MB_TypeConstVars;
          collectOtherInstr(MI, MAI, Type, IsConstOrType);
        } else if (OpCode == SPIRV::OpFunction) {
          collectFuncNames(MI, *F);
        }
      }
  }
}

// Number registers in all functions globally from 0 onwards and store
// the result in global register alias table.
void SPIRVModuleAnalysis::numberRegistersGlobally(const Module &M) {
  for (auto F = M.begin(), E = M.end(); F != E; ++F) {
    if ((*F).isDeclaration())
      continue;
    MachineFunction *MF = MMI->getMachineFunction(*F);
    assert(MF);
    for (MachineBasicBlock &MBB : *MF) {
      for (MachineInstr &MI : MBB) {
        for (MachineOperand &Op : MI.operands()) {
          if (!Op.isReg())
            continue;
          Register Reg = Op.getReg();
          if (MAI.hasRegisterAlias(MF, Reg))
            continue;
          Register NewReg = Register::index2VirtReg(MAI.getNextID());
          MAI.setRegisterAlias(MF, Reg, NewReg);
        }
      }
    }
  }
}

struct SPIRV::ModuleAnalysisInfo SPIRVModuleAnalysis::MAI;

void SPIRVModuleAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.addRequired<MachineModuleInfoWrapperPass>();
}

bool SPIRVModuleAnalysis::runOnModule(Module &M) {
  SPIRVTargetMachine &TM =
      getAnalysis<TargetPassConfig>().getTM<SPIRVTargetMachine>();
  ST = TM.getSubtargetImpl();
  GR = ST->getSPIRVGlobalRegistry();
  TII = ST->getInstrInfo();

  MMI = &getAnalysis<MachineModuleInfoWrapperPass>().getMMI();

  setBaseInfo(M);

  // TODO: Process type/const/global var/func decl instructions, number their
  // destination registers from 0 to N, collect Extensions and Capabilities.

  // Number rest of registers from N+1 onwards.
  numberRegistersGlobally(M);

  // Collect OpName, OpEntryPoint, OpDecorate etc, process other instructions.
  processOtherInstrs(M);

  return false;
}
