//===-- llc.cpp - Implement the LLVM Compiler -----------------------------===//
//
// This is the llc compiler driver.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bytecode/Reader.h"
#include "llvm/Optimizations/Normalize.h"
#include "llvm/Target/Sparc.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Instrumentation/TraceValues.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include <memory>
#include <string>
#include <fstream>

cl::String InputFilename ("", "Input filename", cl::NoFlags, "-");
cl::String OutputFilename("o", "Output filename", cl::NoFlags, "");
cl::Flag   Force         ("f", "Overwrite output files", cl::NoFlags, false);
cl::Flag   DumpAsm       ("d", "Print bytecode before native code generation", cl::Hidden,false);
cl::Flag   DoNotEmitAssembly("noasm", "Do not emit assembly code", cl::Hidden, false);
cl::Flag   TraceBBValues ("trace",
                          "Trace values at basic block and method exits",
                          cl::NoFlags, false);
cl::Flag   TraceMethodValues("tracem", "Trace values only at method exits",
                             cl::NoFlags, false);

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
                    vector<const Type*>(1, Type::UIntTy), false);

  SymbolTable *SymTab = M->getSymbolTableSure();
  
  // Check for a definition of malloc
  if (Value *V = SymTab->lookup(PointerType::get(MallocType), "malloc")) {
    MallocMeth = cast<Method>(V);      // Yup, got it
  } else {                             // Nope, add one
    M->getMethodList().push_back(MallocMeth = new Method(MallocType, "malloc"));
  }

  const MethodType *FreeType = 
    MethodType::get(Type::VoidTy,
                    vector<const Type*>(1, PointerType::get(Type::UByteTy)),
		    false);

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

inline string
GetFileNameRoot(const string& InputFilename)
{
  string IFN = InputFilename;
  string outputFilename;
  int Len = IFN.length();
  if (IFN[Len-3] == '.' && IFN[Len-2] == 'b' && IFN[Len-1] == 'c') {
    outputFilename = string(IFN.begin(), IFN.end()-3); // s/.bc/.s/
  } else {
    outputFilename = IFN;   // Append a .s to it
  }
  return outputFilename;
}

inline string
GetTraceAssemblyFileName(const string& inFilename)
{
  assert(inFilename != "-" && "files on stdin not supported with tracing");
  string traceFileName = GetFileNameRoot(inFilename);
  traceFileName += ".trace.ll"; 
  return traceFileName;
}

//===---------------------------------------------------------------------===//
// Function PreprocessModule()
// 
// Normalization to simplify later passes.
//===---------------------------------------------------------------------===//

int
PreprocessModule(Module* module)
{
  InsertMallocFreeDecls(module);
  
  for (Module::const_iterator MI=module->begin(); MI != module->end(); ++MI)
    if (! (*MI)->isExternal())
      NormalizeMethod(*MI);
  
  return 0;
}


//===---------------------------------------------------------------------===//
// Function OptimizeModule()
// 
// Module optimization.
//===---------------------------------------------------------------------===//

int
OptimizeModule(Module* module)
{
  return 0;
}


//===---------------------------------------------------------------------===//
// Function GenerateCodeForModule()
// 
// Native code generation for a specified target.
//===---------------------------------------------------------------------===//

int
GenerateCodeForModule(Module* module, TargetMachine* target)
{
  // Since any transformation pass may introduce external function decls
  // into the method list, find current methods first and then walk only those.
  // 
  vector<Method*> initialMethods(module->begin(), module->end());
  
  
  // Replace malloc and free instructions with library calls
  // 
  for (unsigned i=0, N = initialMethods.size(); i < N; i++)
    if (! initialMethods[i]->isExternal())
      ReplaceMallocFree(initialMethods[i], target->DataLayout);
  
  
  // Insert trace code to assist debugging
  // 
  if (TraceBBValues || TraceMethodValues)
    {
      // Insert trace code in all methods in the module
      for (unsigned i=0, N = initialMethods.size(); i < N; i++)
        if (! initialMethods[i]->isExternal())
          InsertCodeToTraceValues(initialMethods[i], TraceBBValues,
                                  TraceBBValues || TraceMethodValues);
      
      // Then write the module with tracing code out in assembly form
      string traceFileName = GetTraceAssemblyFileName(InputFilename);
      ofstream* ofs = new ofstream(traceFileName.c_str(), 
                                   (Force ? 0 : ios::noreplace)|ios::out);
      if (!ofs->good()) {
        cerr << "Error opening " << traceFileName << "!\n";
        delete ofs;
        return 1;
      }
      WriteToAssembly(module, *ofs);
      delete ofs;
    }
  
  
  // Generate native target code for all methods
  // 
  for (unsigned i=0, N = initialMethods.size(); i < N; i++)
    if (! initialMethods[i]->isExternal())
      {
        if (DumpAsm)
          cerr << "Method after xformations: \n" << initialMethods[i];
        
        if (target->compileMethod(initialMethods[i])) {
          cerr << "Error compiling " << InputFilename << "!\n";
          return 1;
        }
      }
  
  return 0;
}


//===---------------------------------------------------------------------===//
// Function EmitAssemblyForModule()
// 
// Write assembly code to specified output file; <ModuleName>.s by default.
//===---------------------------------------------------------------------===//

int
EmitAssemblyForModule(Module* module, TargetMachine* target)
{
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
      string OutputFilename = GetFileNameRoot(InputFilename); 
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
  target->emitAssembly(module, *Out);

  if (Out != &cout) delete Out;

  return 0;
}


//===---------------------------------------------------------------------===//
// Function main()
// 
// Entry point for the llc compiler.
//===---------------------------------------------------------------------===//

int
main(int argc, char **argv)
{
  // Parse command line options...
  cl::ParseCommandLineOptions(argc, argv, " llvm system compiler\n");
  
  // Allocate a target... in the future this will be controllable on the
  // command line.
  auto_ptr<TargetMachine> target(allocateSparcTargetMachine());
  
  // Load the module to be compiled...
  auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }
  
  int failed = PreprocessModule(M.get());
  
  if (!failed)
    failed = OptimizeModule(M.get());
  
  if (!failed)
    failed = GenerateCodeForModule(M.get(), target.get());
  
  if (!failed && ! DoNotEmitAssembly)
    failed = EmitAssemblyForModule(M.get(), target.get());
  
  return failed;
}

