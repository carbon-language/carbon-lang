//===-- SkeletonTargetMachine.cpp - Define TargetMachine for Skeleton -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "SkeletonTargetMachine.h"
#include "Skeleton.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
using namespace llvm;

namespace {
  // Register the target.
  RegisterTarget<SkeletonTargetMachine> X("skeleton",
                                          "  Target Skeleton (unusable)");
}

/// SkeletonTargetMachine ctor - Create an ILP32 architecture model
///
SkeletonTargetMachine::SkeletonTargetMachine(const Module &M,
                                           IntrinsicLowering *IL,
                                           const std::string &FS)
  : TargetMachine("Skeleton", IL, true, 4, 4, 4, 4, 4),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, -4), JITInfo(*this) {
}

/// addPassesToEmitFile - Add passes to the specified pass manager
/// to implement a static compiler for this target.
///
bool SkeletonTargetMachine::addPassesToEmitFile(PassManager &PM,
                                                std::ostream &Out,
                                                CodeGenFileType FileType) {
  if (FileType != TargetMachine::AssemblyFile) return true;
  // <insert instruction selector passes here>
  PM.add(createRegisterAllocator());
  PM.add(createPrologEpilogCodeInserter());
  // <insert assembly code output passes here>
  PM.add(createMachineCodeDeleter());
  return true; // change to `return false' when this actually works.
}

/// addPassesToJITCompile - Add passes to the specified pass manager to
/// implement a fast dynamic compiler for this target.
///
void SkeletonJITInfo::addPassesToJITCompile(FunctionPassManager &PM) {
  // <insert instruction selector passes here>
  PM.add(createRegisterAllocator());
  PM.add(createPrologEpilogCodeInserter());
}

