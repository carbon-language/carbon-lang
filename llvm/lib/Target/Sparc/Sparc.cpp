//===-- Sparc.cpp - General implementation file for the Sparc Target ------===//
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
#include "SparcInternals.h"
#include "SparcTargetMachine.h"
#include "Support/CommandLine.h"

using namespace llvm;

static const unsigned ImplicitRegUseList[] = { 0 }; /* not used yet */
// Build the MachineInstruction Description Array...
const TargetInstrDescriptor llvm::SparcMachineInstrDesc[] = {
#define I(ENUM, OPCODESTRING, NUMOPERANDS, RESULTPOS, MAXIMM, IMMSE, \
          NUMDELAYSLOTS, LATENCY, SCHEDCLASS, INSTFLAGS)             \
  { OPCODESTRING, NUMOPERANDS, RESULTPOS, MAXIMM, IMMSE,             \
          NUMDELAYSLOTS, LATENCY, SCHEDCLASS, INSTFLAGS, 0,          \
          ImplicitRegUseList, ImplicitRegUseList },
#include "SparcInstr.def"
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

  cl::opt<bool> DumpInput("dump-input",
                          cl::desc("Print bytecode before code generation"),
                          cl::Hidden);
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
    const char *getPassName() const { return "FreeMachineFunction"; }
    
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

FunctionPass *llvm::createSparcMachineCodeDestructionPass() {
  return new DestroyMachineFunction();
}


SparcTargetMachine::SparcTargetMachine(IntrinsicLowering *il)
  : TargetMachine("UltraSparc-Native", false),
    IL(il ? il : new DefaultIntrinsicLowering()),
    schedInfo(*this),
    regInfo(*this),
    frameInfo(*this),
    cacheInfo(*this),
    jitInfo(*this, *IL) {
}

SparcTargetMachine::~SparcTargetMachine() {
  delete IL;
}

// addPassesToEmitAssembly - This method controls the entire code generation
// process for the ultra sparc.
//
bool
SparcTargetMachine::addPassesToEmitAssembly(PassManager &PM, std::ostream &Out)
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

  // Specialize LLVM code for this target machine
  PM.add(createPreSelectionPass(*this));
  // Run basic dataflow optimizations on LLVM code
  PM.add(createReassociatePass());
  PM.add(createLICMPass());
  PM.add(createGCSEPass());

  // If LLVM dumping after transformations is requested, add it to the pipeline
  if (DumpInput)
    PM.add(new PrintFunctionPass("Input code to instr. selection:\n",
                                 &std::cerr));

  PM.add(createInstructionSelectionPass(*this, *IL));

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
  //
  PM.add(createAsmPrinterPass(Out, *this));
  PM.add(createSparcMachineCodeDestructionPass()); // Free mem no longer needed

  // Emit bytecode to the assembly file into its special section next
  if (EmitMappingInfo)
    PM.add(createBytecodeAsmPrinterPass(Out));

  return false;
}

// addPassesToJITCompile - This method controls the JIT method of code
// generation for the UltraSparc.
//
void SparcJITInfo::addPassesToJITCompile(FunctionPassManager &PM) {
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
  // Run basic dataflow optimizations on LLVM code
  PM.add(createReassociatePass());

  // FIXME: these passes crash the FunctionPassManager when being added...
  //PM.add(createLICMPass());
  //PM.add(createGCSEPass());

  PM.add(createInstructionSelectionPass(TM, IL));

  PM.add(getRegisterAllocator(TM));
  PM.add(createPrologEpilogInsertionPass());

  if (!DisablePeephole)
    PM.add(createPeepholeOptsPass(TM));
}

//----------------------------------------------------------------------------
// allocateSparcTargetMachine - Allocate and return a subclass of TargetMachine
// that implements the Sparc backend. (the llvm/CodeGen/Sparc.h interface)
//----------------------------------------------------------------------------

TargetMachine *llvm::allocateSparcTargetMachine(const Module &M,
                                                IntrinsicLowering *IL) {
  return new SparcTargetMachine(IL);
}
