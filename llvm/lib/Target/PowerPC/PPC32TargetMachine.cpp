//===-- PowerPCTargetMachine.cpp - Define TargetMachine for PowerPC -------===//
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

#include "PPC32.h"
#include "PPC32JITInfo.h"
#include "PPC32TargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include <iostream>
using namespace llvm;

namespace {
  const std::string PPC32 = "Darwin/PowerPC";
  // Register the target
  RegisterTarget<PPC32TargetMachine> 
  X("powerpc-darwin", "  Darwin/PowerPC (experimental)");
}

/// PowerPCTargetMachine ctor - Create an ILP32 architecture model
///
PPC32TargetMachine::PPC32TargetMachine(const Module &M,
                                               IntrinsicLowering *IL)
  : PowerPCTargetMachine(PPC32, IL, 
                         TargetData(PPC32,false,4,4,4,4,4,4,2,1,4),
                         TargetFrameInfo(TargetFrameInfo::StackGrowsDown,16,-4),
                         PPC32JITInfo(*this)) {}

/// addPassesToEmitAssembly - Add passes to the specified pass manager
/// to implement a static compiler for this target.
///
bool PPC32TargetMachine::addPassesToEmitAssembly(PassManager &PM,
                                                     std::ostream &Out) {
  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  PM.add(createLowerConstantExpressionsPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  PM.add(createPPC32ISelSimple(*this));

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  PM.add(createRegisterAllocator());

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  // I want a PowerPC specific prolog/epilog code inserter so I can put the 
  // fills/spills in the right spots.
  PM.add(createPowerPCPEI());
  
  // Must run branch selection immediately preceding the printer
  PM.add(createPPCBranchSelectionPass());
  PM.add(createPPC32AsmPrinter(Out, *this));
  PM.add(createMachineCodeDeleter());
  return false;
}

/// addPassesToJITCompile - Add passes to the specified pass manager to
/// implement a fast dynamic compiler for this target.
///
void PPC32JITInfo::addPassesToJITCompile(FunctionPassManager &PM) {
  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  PM.add(createLowerConstantExpressionsPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  PM.add(createPPC32ISelSimple(TM));
  PM.add(createRegisterAllocator());
  PM.add(createPrologEpilogCodeInserter());
}

unsigned PPC32TargetMachine::getModuleMatchQuality(const Module &M) {
  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer32)
    return 10;                                   // Direct match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return getJITMatchQuality()/2;
}
