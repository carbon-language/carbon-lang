//===-- llc.cpp - Implement the LLVM Compiler -----------------------------===//
//
// This is the llc compiler driver.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bytecode/Reader.h"
#include "llvm/Optimizations/Normalize.h"
#include "llvm/Target/Sparc.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include <memory>
#include <fstream>

cl::String InputFilename ("", "Input filename", cl::NoFlags, "-");
cl::String OutputFilename("o", "Output filename", cl::NoFlags, "");
cl::Flag   Force         ("f", "Overwrite output files", cl::NoFlags, false);
cl::Flag   DumpAsm       ("d", "Print assembly as compiled", cl::Hidden, false);

#include "llvm/Assembly/Writer.h"   // For DumpAsm

//-------------------------- Internal Functions ------------------------------//


/////
// TODO: Remove to external file.... When Chris gets back he'll do it
/////
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/SymbolTable.h"


Method *MallocMeth = 0, *FreeMeth = 0;

// InsertMallocFreeDecls - Insert an external declaration for malloc and an
// external declaration for free for use by the ReplaceMallocFree function.
//
static void InsertMallocFreeDecls(Module *M) {
  const MethodType *MallocType = 
    MethodType::get(PointerType::get(Type::UByteTy),
                    vector<const Type*>(1, Type::UIntTy));

  SymbolTable *SymTab = M->getSymbolTableSure();
  
  // Check for a definition of malloc
  if (Value *V = SymTab->lookup(PointerType::get(MallocType), "malloc")) {
    MallocMeth = cast<Method>(V);      // Yup, got it
  } else {                             // Nope, add one
    M->getMethodList().push_back(MallocMeth = new Method(MallocType, "malloc"));
  }

  const MethodType *FreeType = 
    MethodType::get(Type::VoidTy,
                    vector<const Type*>(1, PointerType::get(Type::UByteTy)));

  // Check for a definition of free
  if (Value *V = SymTab->lookup(PointerType::get(FreeType), "free")) {
    FreeMeth = cast<Method>(V);      // Yup, got it
  } else {                             // Nope, add one
    M->getMethodList().push_back(FreeMeth = new Method(FreeType, "free"));
  }
}


static void ReplaceMallocFree(Method *M, const TargetData &DataLayout) {
  assert(MallocMeth && FreeMeth && M && "Must call InsertMallocFreeDecls!");

  // Loop over all of the instructions, looking for malloc or free instructions
  for (Method::iterator BBI = M->begin(), BBE = M->end(); BBI != BBE; ++BBI) {
    BasicBlock *BB = *BBI;
    for (unsigned i = 0; i < BB->size(); ++i) {
      BasicBlock::InstListType &BBIL = BB->getInstList();
      if (MallocInst *MI = dyn_cast<MallocInst>(*(BBIL.begin()+i))) {
        BBIL.remove(BBIL.begin()+i);   // remove the malloc instr...
        
        const Type *AllocTy = cast<PointerType>(MI->getType())->getValueType();

        // If the user is allocating an unsized array with a dynamic size arg,
        // start by getting the size of one element.
        //
        if (const ArrayType *ATy = dyn_cast<ArrayType>(AllocTy)) 
          if (ATy->isUnsized()) AllocTy = ATy->getElementType();

        // Get the number of bytes to be allocated for one element of the
        // requested type...
        unsigned Size = DataLayout.getTypeSize(AllocTy);

        // malloc(type) becomes sbyte *malloc(constint)
        Value *MallocArg = ConstPoolUInt::get(Type::UIntTy, Size);
        if (MI->getNumOperands() && Size == 1) {
          MallocArg = MI->getOperand(0);         // Operand * 1 = Operand
        } else if (MI->getNumOperands()) {
          // Multiply it by the array size if neccesary...
          MallocArg = BinaryOperator::create(Instruction::Mul,MI->getOperand(0),
                                             MallocArg);
          BBIL.insert(BBIL.begin()+i++, cast<Instruction>(MallocArg));
        }

        // Create the call to Malloc...
        CallInst *MCall = new CallInst(MallocMeth,
                                       vector<Value*>(1, MallocArg));
        BBIL.insert(BBIL.begin()+i, MCall);

        // Create a cast instruction to convert to the right type...
        CastInst *MCast = new CastInst(MCall, MI->getType());
        BBIL.insert(BBIL.begin()+i+1, MCast);

        // Replace all uses of the old malloc inst with the cast inst
        MI->replaceAllUsesWith(MCast);
        delete MI;                          // Delete the malloc inst
      } else if (FreeInst *FI = dyn_cast<FreeInst>(*(BBIL.begin()+i))) {
        BBIL.remove(BB->getInstList().begin()+i);

        // Cast the argument to free into a ubyte*...
        CastInst *MCast = new CastInst(FI->getOperand(0), 
                                       PointerType::get(Type::UByteTy));
        BBIL.insert(BBIL.begin()+i, MCast);

        // Insert a call to the free function...
        CallInst *FCall = new CallInst(FreeMeth,
                                       vector<Value*>(1, MCast));
        BBIL.insert(BBIL.begin()+i+1, FCall);

        // Delete the old free instruction
        delete FI;
      }
    }
  }
}


// END TODO: Remove to external file....

static void NormalizeMethod(Method *M) {
  NormalizePhiConstantArgs(M);
}


//===---------------------------------------------------------------------===//
// Function main()
// 
// Entry point for the llc compiler.
//===---------------------------------------------------------------------===//

int main(int argc, char **argv) {
  // Parse command line options...
  cl::ParseCommandLineOptions(argc, argv, " llvm system compiler\n");

  // Allocate a target... in the future this will be controllable on the
  // command line.
  auto_ptr<TargetMachine> Target(allocateSparcTargetMachine());

  // Load the module to be compiled...
  auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }

  InsertMallocFreeDecls(M.get());

  // Loop over all of the methods in the module, compiling them.
  for (Module::const_iterator MI = M->begin(), ME = M->end(); MI != ME; ++MI) {
    Method *Meth = *MI;
    
    NormalizeMethod(Meth);
    ReplaceMallocFree(Meth, Target->DataLayout);
    
    if (DumpAsm)
      cerr << "Method after xformations: \n" << Meth;

    if (Target->compileMethod(Meth)) {
      cerr << "Error compiling " << InputFilename << "!\n";
      return 1;
    }
  }
  
  // Figure out where we are going to send the output...
  ostream *Out = 0;
  if (OutputFilename != "") {   // Specified an output filename?
    Out = new ofstream(OutputFilename.c_str(), 
                       (Force ? 0 : ios::noreplace)|ios::out);
  } else {
    if (InputFilename == "-") {
      OutputFilename = "-";
      Out = &cout;
    } else {
      string IFN = InputFilename;
      int Len = IFN.length();
      if (IFN[Len-3] == '.' && IFN[Len-2] == 'b' && IFN[Len-1] == 'c') {
        OutputFilename = string(IFN.begin(), IFN.end()-3); // s/.bc/.s/
      } else {
        OutputFilename = IFN;   // Append a .s to it
      }
      OutputFilename += ".s";
      Out = new ofstream(OutputFilename.c_str(), 
                         (Force ? 0 : ios::noreplace)|ios::out);
      if (!Out->good()) {
        cerr << "Error opening " << OutputFilename << "!\n";
        delete Out;
        return 1;
      }
    }
  }

  // Emit the output...
  Target->emitAssembly(M.get(), *Out);

  if (Out != &cout) delete Out;
  return 0;
}


