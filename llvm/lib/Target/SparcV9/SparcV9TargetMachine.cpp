//===-- SparcV9TargetMachine.cpp - SparcV9 Target Machine Implementation --===//
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
#include "llvm/PassManager.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/InstrScheduling.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
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
    jitInfo(*this) {
}

/// addPassesToEmitAssembly - This method controls the entire code generation
/// process for the ultra sparc.
///
bool
SparcV9TargetMachine::addPassesToEmitAssembly(PassManager &PM, std::ostream &Out)
{
  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // Replace malloc and free instructions with library calls.
  PM.add(createLowerAllocationsPass());
  
  // FIXME: implement the switch instruction in the instruction selector.
  PM.add(createLowerSwitchPass());

  // FIXME: implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());
  
  // decompose multi-dimensional array references into single-dim refs
  PM.add(createDecomposeMultiDimRefsPass());

  // Lower LLVM code to the form expected by the SPARCv9 instruction selector.
  PM.add(createPreSelectionPass(*this));
  PM.add(createLowerSelectPass());

  // Run basic LLVM dataflow optimizations, to clean up after pre-selection.
  PM.add(createReassociatePass());
  PM.add(createLICMPass());
  PM.add(createGCSEPass());

  // If the user's trying to read the generated code, they'll need to see the
  // transformed input.
  if (PrintMachineCode)
    PM.add(new PrintModulePass());

  // Construct and initialize the MachineFunction object for this fn.
  PM.add(createMachineCodeConstructionPass(*this));

  // Insert empty stackslots in the stack frame of each function
  // so %fp+offset-8 and %fp+offset-16 are empty slots now!
  PM.add(createStackSlotsPass(*this));
  
  PM.add(createInstructionSelectionPass(*this));

  if (!DisableSched)
    PM.add(createInstructionSchedulingWithSSAPass(*this));

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr, "Before reg alloc:\n"));

  PM.add(getRegisterAllocator(*this));

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr, "After reg alloc:\n"));

  PM.add(createPrologEpilogInsertionPass());

  if (!DisablePeephole)
    PM.add(createPeepholeOptsPass(*this));

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr, "Final code:\n"));

  if (EmitMappingInfo) {
    PM.add(createInternalGlobalMapperPass());
    PM.add(getMappingInfoAsmPrinterPass(Out));
  }

  // Output assembly language to the .s file.  Assembly emission is split into
  // two parts: Function output and Global value output.  This is because
  // function output is pipelined with all of the rest of code generation stuff,
  // allowing machine code representations for functions to be free'd after the
  // function has been emitted.
  PM.add(createAsmPrinterPass(Out, *this));

  // Free machine-code IR which is no longer needed:
  PM.add(createSparcV9MachineCodeDestructionPass());

  // Emit bytecode to the assembly file into its special section next
  if (EmitMappingInfo) {
    // Strip all of the symbols from the bytecode so that it will be smaller...
    if (!DisableStrip)
      PM.add(createSymbolStrippingPass());
    PM.add(createBytecodeAsmPrinterPass(Out));
  }
  
  return false;
}

/// addPassesToJITCompile - This method controls the JIT method of code
/// generation for the UltraSparcV9.
///
void SparcV9JITInfo::addPassesToJITCompile(FunctionPassManager &PM) {
  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // Replace malloc and free instructions with library calls.
  PM.add(createLowerAllocationsPass());
  
  // FIXME: implement the switch instruction in the instruction selector.
  PM.add(createLowerSwitchPass());

  // FIXME: implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());
  
  // decompose multi-dimensional array references into single-dim refs
  PM.add(createDecomposeMultiDimRefsPass());

  // Lower LLVM code to the form expected by the SPARCv9 instruction selector.
  PM.add(createPreSelectionPass(TM));
  PM.add(createLowerSelectPass());

  // Run basic LLVM dataflow optimizations, to clean up after pre-selection.
  PM.add(createReassociatePass());
  // FIXME: these passes crash the FunctionPassManager when being added...
  //PM.add(createLICMPass());
  //PM.add(createGCSEPass());

  // If the user's trying to read the generated code, they'll need to see the
  // transformed input.
  if (PrintMachineCode)
    PM.add(new PrintFunctionPass());

  // Construct and initialize the MachineFunction object for this fn.
  PM.add(createMachineCodeConstructionPass(TM));

  PM.add(createInstructionSelectionPass(TM));

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr, "Before reg alloc:\n"));

  PM.add(getRegisterAllocator(TM));

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr, "After reg alloc:\n"));

  PM.add(createPrologEpilogInsertionPass());

  if (!DisablePeephole)
    PM.add(createPeepholeOptsPass(TM));

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr, "Final code:\n"));
}

/// allocateSparcV9TargetMachine - Allocate and return a subclass of TargetMachine
/// that implements the SparcV9 backend. (the llvm/CodeGen/SparcV9.h interface)
///
TargetMachine *llvm::allocateSparcV9TargetMachine(const Module &M,
                                                IntrinsicLowering *IL) {
  return new SparcV9TargetMachine(IL);
}
