//===-- MipsTargetMachine.cpp - Define TargetMachine for Mips -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Mips target spec.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsTargetAsmInfo.h"
#include "MipsTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

namespace {
  // Register the target.
  RegisterTarget<MipsTargetMachine> X("mips", "  Mips");
}

const TargetAsmInfo *MipsTargetMachine::
createTargetAsmInfo() const 
{
  return new MipsTargetAsmInfo(*this);
}

// DataLayout --> Big-endian, 32-bit pointer/ABI/alignment
//
// FrameInfo  --> StackGrowsDown, 8 bytes aligned, 
//                LOA : 0
MipsTargetMachine::
MipsTargetMachine(const Module &M, const std::string &FS): 
  Subtarget(*this, M, FS), DataLayout("E-p:32:32:32"), 
  InstrInfo(*this), FrameInfo(TargetFrameInfo::StackGrowsDown, 8, 0),
  TLInfo(*this) {}

// return 0 and must specify -march to gen MIPS code.
unsigned MipsTargetMachine::
getModuleMatchQuality(const Module &M) 
{
  // We strongly match "mips-*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 5 && std::string(TT.begin(), TT.begin()+5) == "mips-")
    return 20;
  
  return 0;
}

// Install an instruction selector pass using 
// the ISelDag to gen Mips code.
bool MipsTargetMachine::
addInstSelector(FunctionPassManager &PM, bool Fast) 
{
  PM.add(createMipsISelDag(*this));
  return false;
}

// Implemented by targets that want to run passes immediately before 
// machine code is emitted. return true if -print-machineinstrs should 
// print out the code after the passes.
// TODO: Delay slot must be implemented here.
bool MipsTargetMachine::
addPreEmitPass(FunctionPassManager &PM, bool Fast) 
{
  return false;
}

// Implements the AssemblyEmitter for the target. Must return
// true if AssemblyEmitter is supported
bool MipsTargetMachine::
addAssemblyEmitter(FunctionPassManager &PM, bool Fast, 
                   std::ostream &Out) 
{
  // Output assembly language.
  PM.add(createMipsCodePrinterPass(Out, *this));
  return false;
}
