//===-- PIC16TargetMachine.cpp - Define TargetMachine for PIC16 -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PIC16 target.
//
//===----------------------------------------------------------------------===//

#include "PIC16.h"
#include "PIC16TargetAsmInfo.h"
#include "PIC16TargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetMachineRegistry.h"

using namespace llvm;

namespace {
  // Register the targets
  RegisterTarget<PIC16TargetMachine> X("pic16", "PIC16 14-bit [experimental]");
}

PIC16TargetMachine::
PIC16TargetMachine(const Module &M, const std::string &FS) :
  Subtarget(*this, M, FS), DataLayout("e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"), 
  InstrInfo(*this), TLInfo(*this),
  FrameInfo(TargetFrameInfo::StackGrowsUp, 8, 0) { }


const TargetAsmInfo *PIC16TargetMachine::createTargetAsmInfo() const 
{
  return new PIC16TargetAsmInfo(*this);
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool PIC16TargetMachine::addInstSelector(PassManagerBase &PM, bool Fast) 
{
  // Install an instruction selector.
  PM.add(createPIC16ISelDag(*this));
  return false;
}

bool PIC16TargetMachine::
addPrologEpilogInserter(PassManagerBase &PM, bool Fast) 
{
  return false;
}

bool PIC16TargetMachine::addPreEmitPass(PassManagerBase &PM, bool Fast) 
{
  return true;
}

bool PIC16TargetMachine::
addAssemblyEmitter(PassManagerBase &PM, bool Fast, raw_ostream &Out) 
{
  // Output assembly language.
  PM.add(createPIC16CodePrinterPass(Out, *this));
  return false;
}

