//===-- TargetMachine.cpp - General Target Information ---------------------==//
//
// This file describes the general parts of a Target machine.
// This file also implements MachineInstrInfo and MachineCacheInfo.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Target/MachineCacheInfo.h"
#include "llvm/CodeGen/PreSelection.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/InstrScheduling.h"
#include "llvm/CodeGen/RegisterAllocation.h"
#include "llvm/CodeGen/PeepholeOpts.h"
#include "llvm/CodeGen/MachineCodeForMethod.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/Reoptimizer/Mapping/MappingInfo.h" 
#include "llvm/Reoptimizer/Mapping/FInfo.h" 
#include "llvm/Transforms/Scalar.h"
#include "Support/CommandLine.h"
#include "llvm/PassManager.h"
#include "llvm/Function.h"
#include "llvm/DerivedTypes.h"

//---------------------------------------------------------------------------
// Command line options to control choice of code generation passes.
//---------------------------------------------------------------------------

static cl::opt<bool> DisablePreSelect("nopreselect",
                                      cl::desc("Disable preselection pass"));

static cl::opt<bool> DisableSched("nosched",
                                  cl::desc("Disable local scheduling pass"));

static cl::opt<bool> DisablePeephole("nopeephole",
                                     cl::desc("Disable peephole optimization pass"));

//---------------------------------------------------------------------------
// class TargetMachine
// 
// Purpose:
//   Machine description.
// 
//---------------------------------------------------------------------------


// function TargetMachine::findOptimalStorageSize 
// 
// Purpose:
//   This default implementation assumes that all sub-word data items use
//   space equal to optSizeForSubWordData, and all other primitive data
//   items use space according to the type.
//   
unsigned int
TargetMachine::findOptimalStorageSize(const Type* ty) const
{
  switch(ty->getPrimitiveID())
    {
    case Type::BoolTyID:
    case Type::UByteTyID:
    case Type::SByteTyID:     
    case Type::UShortTyID:
    case Type::ShortTyID:     
      return optSizeForSubWordData;
    
    default:
      return DataLayout.getTypeSize(ty);
    }
}


//===---------------------------------------------------------------------===//
// Default code generation passes.
// 
// Native code generation for a specified target.
//===---------------------------------------------------------------------===//

class ConstructMachineCodeForFunction : public FunctionPass {
  TargetMachine &Target;
public:
  inline ConstructMachineCodeForFunction(TargetMachine &T) : Target(T) {}

  const char *getPassName() const {
    return "ConstructMachineCodeForFunction";
  }

  bool runOnFunction(Function &F) {
    MachineCodeForMethod::construct(&F, Target);
    return false;
  }
};

struct FreeMachineCodeForFunction : public FunctionPass {
  const char *getPassName() const { return "FreeMachineCodeForFunction"; }

  static void freeMachineCode(Instruction &I) {
    MachineCodeForInstruction::destroy(&I);
  }
  
  bool runOnFunction(Function &F) {
    for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
      for (BasicBlock::iterator I = FI->begin(), E = FI->end(); I != E; ++I)
        MachineCodeForInstruction::get(I).dropAllReferences();
    
    for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
      for_each(FI->begin(), FI->end(), freeMachineCode);
    
    return false;
  }
};

// addPassesToEmitAssembly - This method controls the entire code generation
// process for the ultra sparc.
//
void
TargetMachine::addPassesToEmitAssembly(PassManager &PM, std::ostream &Out)
{
  // Construct and initialize the MachineCodeForMethod object for this fn.
  PM.add(new ConstructMachineCodeForFunction(*this));

  // Specialize LLVM code for this target machine and then
  // run basic dataflow optimizations on LLVM code.
  if (!DisablePreSelect)
    {
      PM.add(createPreSelectionPass(*this));
      PM.add(createReassociatePass());
      PM.add(createGCSEPass());
      PM.add(createLICMPass());
    }

  PM.add(createInstructionSelectionPass(*this));

  if (!DisableSched)
    PM.add(createInstructionSchedulingWithSSAPass(*this));

  PM.add(getRegisterAllocator(*this));

  PM.add(getPrologEpilogInsertionPass());

  if (!DisablePeephole)
    PM.add(createPeepholeOptsPass(*this));

  PM.add(MappingInfoForFunction(Out));  

  // Output assembly language to the .s file.  Assembly emission is split into
  // two parts: Function output and Global value output.  This is because
  // function output is pipelined with all of the rest of code generation stuff,
  // allowing machine code representations for functions to be free'd after the
  // function has been emitted.
  //
  PM.add(getFunctionAsmPrinterPass(Out));
  PM.add(new FreeMachineCodeForFunction());  // Free stuff no longer needed

  // Emit Module level assembly after all of the functions have been processed.
  PM.add(getModuleAsmPrinterPass(Out));

  // Emit bytecode to the assembly file into its special section next
  PM.add(getEmitBytecodeToAsmPass(Out));
  PM.add(getFunctionInfo(Out)); 
}


//---------------------------------------------------------------------------
// class MachineInstructionInfo
//	Interface to description of machine instructions
//---------------------------------------------------------------------------


/*ctor*/
MachineInstrInfo::MachineInstrInfo(const TargetMachine& tgt,
                                   const MachineInstrDescriptor* _desc,
				   unsigned int _descSize,
				   unsigned int _numRealOpCodes)
  : target(tgt),
    desc(_desc), descSize(_descSize), numRealOpCodes(_numRealOpCodes)
{
  // FIXME: TargetInstrDescriptors should not be global
  assert(TargetInstrDescriptors == NULL && desc != NULL);
  TargetInstrDescriptors = desc;	// initialize global variable
}  


MachineInstrInfo::~MachineInstrInfo()
{
  TargetInstrDescriptors = NULL;	// reset global variable
}


bool
MachineInstrInfo::constantFitsInImmedField(MachineOpCode opCode,
					   int64_t intValue) const
{
  // First, check if opCode has an immed field.
  bool isSignExtended;
  uint64_t maxImmedValue = maxImmedConstant(opCode, isSignExtended);
  if (maxImmedValue != 0)
    {
      // NEED TO HANDLE UNSIGNED VALUES SINCE THEY MAY BECOME MUCH
      // SMALLER AFTER CASTING TO SIGN-EXTENDED int, short, or char.
      // See CreateUIntSetInstruction in SparcInstrInfo.cpp.
      
      // Now check if the constant fits
      if (intValue <= (int64_t) maxImmedValue &&
	  intValue >= -((int64_t) maxImmedValue+1))
	return true;
    }
  
  return false;
}


//---------------------------------------------------------------------------
// class MachineCacheInfo 
// 
// Purpose:
//   Describes properties of the target cache architecture.
//---------------------------------------------------------------------------

/*ctor*/
MachineCacheInfo::MachineCacheInfo(const TargetMachine& tgt)
  : target(tgt)
{
  Initialize();
}

void
MachineCacheInfo::Initialize()
{
  numLevels = 2;
  cacheLineSizes.push_back(16);  cacheLineSizes.push_back(32); 
  cacheSizes.push_back(1 << 15); cacheSizes.push_back(1 << 20);
  cacheAssoc.push_back(1);       cacheAssoc.push_back(4);
}
