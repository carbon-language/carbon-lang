//===-- SparcV9.cpp - General implementation file for the SparcV9 Target ------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// Primary interface to machine description for the UltraSPARC.  Primarily just
// initializes machine-dependent parameters in class TargetMachine, and creates
// machine-dependent subclasses for classes such as TargetInstrInfo.
// 
//===----------------------------------------------------------------------===//

#include "llvm/Function.h"
#include "llvm/IntrinsicLowering.h"
#include "llvm/PassManager.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/InstrScheduling.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "llvm/Transforms/Scalar.h"
#include "MappingInfo.h" 
#include "SparcV9Internals.h"
#include "SparcV9TargetMachine.h"
#include "Support/CommandLine.h"

using namespace llvm;

static const unsigned ImplicitRegUseList[] = { 0 }; /* not used yet */
// Build the MachineInstruction Description Array...
const TargetInstrDescriptor llvm::SparcV9MachineInstrDesc[] = {
#define I(ENUM, OPCODESTRING, NUMOPERANDS, RESULTPOS, MAXIMM, IMMSE, \
          NUMDELAYSLOTS, LATENCY, SCHEDCLASS, INSTFLAGS)             \
  { OPCODESTRING, NUMOPERANDS, RESULTPOS, MAXIMM, IMMSE,             \
          NUMDELAYSLOTS, LATENCY, SCHEDCLASS, INSTFLAGS, 0,          \
          ImplicitRegUseList, ImplicitRegUseList },
#include "SparcV9Instr.def"
};

//---------------------------------------------------------------------------
// Command line options to control choice of code generation passes.
//---------------------------------------------------------------------------

namespace {
  cl::opt<bool> DisableSched("disable-sched",
                             cl::desc("Disable local scheduling pass"));

  cl::opt<bool> DisablePeephole("disable-peephole",
                                cl::desc("Disable peephole optimization pass"));

  cl::opt<bool> EmitMappingInfo("enable-maps",
                 cl::desc("Emit LLVM-to-MachineCode mapping info to assembly"));

  cl::opt<bool> DisableStrip("disable-strip",
                      cl::desc("Do not strip the LLVM bytecode in executable"));
}

//===---------------------------------------------------------------------===//
// Code generation/destruction passes
//===---------------------------------------------------------------------===//

namespace {
  class ConstructMachineFunction : public FunctionPass {
    TargetMachine &Target;
  public:
    ConstructMachineFunction(TargetMachine &T) : Target(T) {}
    
    const char *getPassName() const {
      return "ConstructMachineFunction";
    }
    
    bool runOnFunction(Function &F) {
      MachineFunction::construct(&F, Target).getInfo()->CalculateArgSize();
      return false;
    }
  };

  struct DestroyMachineFunction : public FunctionPass {
    const char *getPassName() const { return "DestroyMachineFunction"; }
    
    static void freeMachineCode(Instruction &I) {
      MachineCodeForInstruction::destroy(&I);
    }
    
    bool runOnFunction(Function &F) {
      for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
        for (BasicBlock::iterator I = FI->begin(), E = FI->end(); I != E; ++I)
          MachineCodeForInstruction::get(I).dropAllReferences();
      
      for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
        for_each(FI->begin(), FI->end(), freeMachineCode);
      
      MachineFunction::destruct(&F);
      return false;
    }
  };
  
  FunctionPass *createMachineCodeConstructionPass(TargetMachine &Target) {
    return new ConstructMachineFunction(Target);
  }
}

FunctionPass *llvm::createSparcV9MachineCodeDestructionPass() {
  return new DestroyMachineFunction();
}


SparcV9TargetMachine::SparcV9TargetMachine(IntrinsicLowering *il)
  : TargetMachine("UltraSparcV9-Native", il, false),
    schedInfo(*this),
    regInfo(*this),
    frameInfo(*this),
    cacheInfo(*this),
    jitInfo(*this) {
}

/// addPassesToEmitAssembly - This method controls the entire code generation
/// process for the ultra sparc.
///
bool
SparcV9TargetMachine::addPassesToEmitAssembly(PassManager &PM, std::ostream &Out)
{
  // The following 3 passes used to be inserted specially by llc.
  // Replace malloc and free instructions with library calls.
  PM.add(createLowerAllocationsPass());
  
  // Strip all of the symbols from the bytecode so that it will be smaller...
  if (!DisableStrip)
    PM.add(createSymbolStrippingPass());

  // FIXME: implement the switch instruction in the instruction selector.
  PM.add(createLowerSwitchPass());

  // FIXME: implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());
  
  // decompose multi-dimensional array references into single-dim refs
  PM.add(createDecomposeMultiDimRefsPass());
  
  // Construct and initialize the MachineFunction object for this fn.
  PM.add(createMachineCodeConstructionPass(*this));

  //Insert empty stackslots in the stack frame of each function
  //so %fp+offset-8 and %fp+offset-16 are empty slots now!
  PM.add(createStackSlotsPass(*this));

  // Specialize LLVM code for this target machine and then
  // run basic dataflow optimizations on LLVM code.
  PM.add(createPreSelectionPass(*this));
  PM.add(createReassociatePass());
  PM.add(createLICMPass());
  PM.add(createGCSEPass());

  PM.add(createInstructionSelectionPass(*this));

  if (!DisableSched)
    PM.add(createInstructionSchedulingWithSSAPass(*this));

  PM.add(getRegisterAllocator(*this));
  PM.add(createPrologEpilogInsertionPass());

  if (!DisablePeephole)
    PM.add(createPeepholeOptsPass(*this));

  if (EmitMappingInfo)
    PM.add(getMappingInfoAsmPrinterPass(Out));  

  // Output assembly language to the .s file.  Assembly emission is split into
  // two parts: Function output and Global value output.  This is because
  // function output is pipelined with all of the rest of code generation stuff,
  // allowing machine code representations for functions to be free'd after the
  // function has been emitted.
  PM.add(createAsmPrinterPass(Out, *this));

  // FIXME: this pass crashes if added; there is a double deletion going on
  // somewhere inside it. This is caught when running the SparcV9 code generator
  // on X86, but is typically ignored when running natively.
  // Free machine-code IR which is no longer needed:
  // PM.add(createSparcV9MachineCodeDestructionPass());

  // Emit bytecode to the assembly file into its special section next
  if (EmitMappingInfo)
    PM.add(createBytecodeAsmPrinterPass(Out));

  return false;
}

/// addPassesToJITCompile - This method controls the JIT method of code
/// generation for the UltraSparcV9.
///
void SparcV9JITInfo::addPassesToJITCompile(FunctionPassManager &PM) {
  const TargetData &TD = TM.getTargetData();

  PM.add(new TargetData("lli", TD.isLittleEndian(), TD.getPointerSize(),
                        TD.getPointerAlignment(), TD.getDoubleAlignment()));

  // Replace malloc and free instructions with library calls.
  // Do this after tracing until lli implements these lib calls.
  // For now, it will emulate malloc and free internally.
  PM.add(createLowerAllocationsPass());

  // FIXME: implement the switch instruction in the instruction selector.
  PM.add(createLowerSwitchPass());

  // FIXME: implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // decompose multi-dimensional array references into single-dim refs
  PM.add(createDecomposeMultiDimRefsPass());
  
  // Construct and initialize the MachineFunction object for this fn.
  PM.add(createMachineCodeConstructionPass(TM));

  // Specialize LLVM code for this target machine and then
  // run basic dataflow optimizations on LLVM code.
  PM.add(createPreSelectionPass(TM));
  PM.add(createReassociatePass());
  // FIXME: these passes crash the FunctionPassManager when being added...
  //PM.add(createLICMPass());
  //PM.add(createGCSEPass());

  PM.add(createInstructionSelectionPass(TM));

  PM.add(getRegisterAllocator(TM));
  PM.add(createPrologEpilogInsertionPass());

  if (!DisablePeephole)
    PM.add(createPeepholeOptsPass(TM));
}

/// allocateSparcV9TargetMachine - Allocate and return a subclass of TargetMachine
/// that implements the SparcV9 backend. (the llvm/CodeGen/SparcV9.h interface)
///
TargetMachine *llvm::allocateSparcV9TargetMachine(const Module &M,
                                                IntrinsicLowering *IL) {
  return new SparcV9TargetMachine(IL);
}
