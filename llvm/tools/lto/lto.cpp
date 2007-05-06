//===-lto.cpp - LLVM Link Time Optimizer ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Devang Patel and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Link Time Optimization library. This library is 
// intended to be used by linker to optimize code at link time.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Linker.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Program.h"
#include "llvm/System/Signals.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/CodeGen/FileWriters.h"
#include "llvm/Target/SubtargetFeature.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/LoadValueNumbering.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/LinkTimeOptimizer.h"
#include <fstream>
#include <ostream>
using namespace llvm;

extern "C"
llvm::LinkTimeOptimizer *createLLVMOptimizer()
{
  llvm::LTO *l = new llvm::LTO();
  return l;
}

/// If symbol is not used then make it internal and let optimizer takes 
/// care of it.
void LLVMSymbol::mayBeNotUsed() { 
  gv->setLinkage(GlobalValue::InternalLinkage); 
}

// Map LLVM LinkageType to LTO LinakgeType
static LTOLinkageTypes
getLTOLinkageType(GlobalValue *v)
{
  LTOLinkageTypes lt;
  if (v->hasExternalLinkage())
    lt = LTOExternalLinkage;
  else if (v->hasLinkOnceLinkage())
    lt = LTOLinkOnceLinkage;
  else if (v->hasWeakLinkage())
    lt = LTOWeakLinkage;
  else
    // Otherwise it is internal linkage for link time optimizer
    lt = LTOInternalLinkage;
  return lt;
}

// Find exeternal symbols referenced by VALUE. This is a recursive function.
static void
findExternalRefs(Value *value, std::set<std::string> &references, 
                 Mangler &mangler) {

  if (GlobalValue *gv = dyn_cast<GlobalValue>(value)) {
    LTOLinkageTypes lt = getLTOLinkageType(gv);
    if (lt != LTOInternalLinkage && strncmp (gv->getName().c_str(), "llvm.", 5))
      references.insert(mangler.getValueName(gv));
  }

  // GlobalValue, even with InternalLinkage type, may have operands with 
  // ExternalLinkage type. Do not ignore these operands.
  if (Constant *c = dyn_cast<Constant>(value))
    // Handle ConstantExpr, ConstantStruct, ConstantArry etc..
    for (unsigned i = 0, e = c->getNumOperands(); i != e; ++i)
      findExternalRefs(c->getOperand(i), references, mangler);
}

/// If Module with InputFilename is available then remove it from allModules
/// and call delete on it.
void
LTO::removeModule (const std::string &InputFilename)
{
  NameToModuleMap::iterator pos = allModules.find(InputFilename.c_str());
  if (pos == allModules.end()) 
    return;

  Module *m = pos->second;
  allModules.erase(pos);
  delete m;
}

/// InputFilename is a LLVM bytecode file. If Module with InputFilename is
/// available then return it. Otherwise parseInputFilename.
Module *
LTO::getModule(const std::string &InputFilename)
{
  Module *m = NULL;

  NameToModuleMap::iterator pos = allModules.find(InputFilename.c_str());
  if (pos != allModules.end())
    m = allModules[InputFilename.c_str()];
  else {
    if (MemoryBuffer *Buffer
        = MemoryBuffer::getFile(&InputFilename[0], InputFilename.size())) {
      m = ParseBitcodeFile(Buffer);
      delete Buffer;
    }
    allModules[InputFilename.c_str()] = m;
  }
  return m;
}

/// InputFilename is a LLVM bytecode file. Reade this bytecode file and 
/// set corresponding target triplet string.
void
LTO::getTargetTriple(const std::string &InputFilename, 
                     std::string &targetTriple)
{
  Module *m = getModule(InputFilename);
  if (m)
    targetTriple = m->getTargetTriple();
}

/// InputFilename is a LLVM bytecode file. Read it using bytecode reader.
/// Collect global functions and symbol names in symbols vector.
/// Collect external references in references vector.
/// Return LTO_READ_SUCCESS if there is no error.
enum LTOStatus
LTO::readLLVMObjectFile(const std::string &InputFilename,
                        NameToSymbolMap &symbols,
                        std::set<std::string> &references)
{
  Module *m = getModule(InputFilename);
  if (!m)
    return LTO_READ_FAILURE;

  // Collect Target info
  getTarget(m);

  if (!Target)
    return LTO_READ_FAILURE;
  
  // Use mangler to add GlobalPrefix to names to match linker names.
  // FIXME : Instead of hard coding "-" use GlobalPrefix.
  Mangler mangler(*m, Target->getTargetAsmInfo()->getGlobalPrefix());
  modules.push_back(m);
  
  for (Module::iterator f = m->begin(), e = m->end(); f != e; ++f) {

    LTOLinkageTypes lt = getLTOLinkageType(f);

    if (!f->isDeclaration() && lt != LTOInternalLinkage
        && strncmp (f->getName().c_str(), "llvm.", 5)) {
      int alignment = ( 16 > f->getAlignment() ? 16 : f->getAlignment());
      LLVMSymbol *newSymbol = new LLVMSymbol(lt, f, f->getName(), 
                                             mangler.getValueName(f),
                                             Log2_32(alignment));
      symbols[newSymbol->getMangledName()] = newSymbol;
      allSymbols[newSymbol->getMangledName()] = newSymbol;
    }

    // Collect external symbols referenced by this function.
    for (Function::iterator b = f->begin(), fe = f->end(); b != fe; ++b) 
      for (BasicBlock::iterator i = b->begin(), be = b->end(); 
           i != be; ++i)
        for (unsigned count = 0, total = i->getNumOperands(); 
             count != total; ++count)
          findExternalRefs(i->getOperand(count), references, mangler);
  }
    
  for (Module::global_iterator v = m->global_begin(), e = m->global_end();
       v !=  e; ++v) {
    LTOLinkageTypes lt = getLTOLinkageType(v);
    if (!v->isDeclaration() && lt != LTOInternalLinkage
        && strncmp (v->getName().c_str(), "llvm.", 5)) {
      const TargetData *TD = Target->getTargetData();
      LLVMSymbol *newSymbol = new LLVMSymbol(lt, v, v->getName(), 
                                             mangler.getValueName(v),
                                             TD->getPreferredAlignmentLog(v));
      symbols[newSymbol->getMangledName()] = newSymbol;
      allSymbols[newSymbol->getMangledName()] = newSymbol;

      for (unsigned count = 0, total = v->getNumOperands(); 
           count != total; ++count)
        findExternalRefs(v->getOperand(count), references, mangler);

    }
  }
  
  return LTO_READ_SUCCESS;
}

/// Get TargetMachine.
/// Use module M to find appropriate Target.
void
LTO::getTarget (Module *M) {

  if (Target)
    return;

  std::string Err;
  const TargetMachineRegistry::Entry* March = 
    TargetMachineRegistry::getClosestStaticTargetForModule(*M, Err);
  
  if (March == 0)
    return;
  
  // Create target
  std::string Features;
  Target = March->CtorFn(*M, Features);
}

/// Optimize module M using various IPO passes. Use exportList to 
/// internalize selected symbols. Target platform is selected
/// based on information available to module M. No new target
/// features are selected. 
enum LTOStatus 
LTO::optimize(Module *M, std::ostream &Out,
              std::vector<const char *> &exportList)
{
  // Instantiate the pass manager to organize the passes.
  PassManager Passes;
  
  // Collect Target info
  getTarget(M);

  if (!Target)
    return LTO_NO_TARGET;
  
  // Start off with a verification pass.
  Passes.add(createVerifierPass());
  
  // Add an appropriate TargetData instance for this module...
  Passes.add(new TargetData(*Target->getTargetData()));
  
  // Internalize symbols if export list is nonemty
  if (!exportList.empty())
    Passes.add(createInternalizePass(exportList));

  // Now that we internalized some globals, see if we can hack on them!
  Passes.add(createGlobalOptimizerPass());
  
  // Linking modules together can lead to duplicated global constants, only
  // keep one copy of each constant...
  Passes.add(createConstantMergePass());
  
  // If the -s command line option was specified, strip the symbols out of the
  // resulting program to make it smaller.  -s is a GLD option that we are
  // supporting.
  Passes.add(createStripSymbolsPass());
  
  // Propagate constants at call sites into the functions they call.
  Passes.add(createIPConstantPropagationPass());
  
  // Remove unused arguments from functions...
  Passes.add(createDeadArgEliminationPass());
  
  Passes.add(createFunctionInliningPass()); // Inline small functions
  
  Passes.add(createPruneEHPass());            // Remove dead EH info

  Passes.add(createGlobalDCEPass());          // Remove dead functions

  // If we didn't decide to inline a function, check to see if we can
  // transform it to pass arguments by value instead of by reference.
  Passes.add(createArgumentPromotionPass());

  // The IPO passes may leave cruft around.  Clean up after them.
  Passes.add(createInstructionCombiningPass());
  
  Passes.add(createScalarReplAggregatesPass()); // Break up allocas
  
  // Run a few AA driven optimizations here and now, to cleanup the code.
  Passes.add(createGlobalsModRefPass());      // IP alias analysis
  
  Passes.add(createLICMPass());               // Hoist loop invariants
  Passes.add(createLoadValueNumberingPass()); // GVN for load instrs
  Passes.add(createGCSEPass());               // Remove common subexprs
  Passes.add(createDeadStoreEliminationPass()); // Nuke dead stores

  // Cleanup and simplify the code after the scalar optimizations.
  Passes.add(createInstructionCombiningPass());
 
  // Delete basic blocks, which optimization passes may have killed...
  Passes.add(createCFGSimplificationPass());
  
  // Now that we have optimized the program, discard unreachable functions...
  Passes.add(createGlobalDCEPass());
  
  // Make sure everything is still good.
  Passes.add(createVerifierPass());

  FunctionPassManager *CodeGenPasses =
    new FunctionPassManager(new ExistingModuleProvider(M));

  CodeGenPasses->add(new TargetData(*Target->getTargetData()));

  MachineCodeEmitter *MCE = 0;

  switch (Target->addPassesToEmitFile(*CodeGenPasses, Out,
                                      TargetMachine::AssemblyFile, true)) {
  default:
  case FileModel::Error:
    return LTO_WRITE_FAILURE;
  case FileModel::AsmFile:
    break;
  case FileModel::MachOFile:
    MCE = AddMachOWriter(*CodeGenPasses, Out, *Target);
    break;
  case FileModel::ElfFile:
    MCE = AddELFWriter(*CodeGenPasses, Out, *Target);
    break;
  }

  if (Target->addPassesToEmitFileFinish(*CodeGenPasses, MCE, true))
    return LTO_WRITE_FAILURE;

  // Run our queue of passes all at once now, efficiently.
  Passes.run(*M);

  // Run the code generator, if present.
  CodeGenPasses->doInitialization();
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I) {
    if (!I->isDeclaration())
      CodeGenPasses->run(*I);
  }
  CodeGenPasses->doFinalization();

  return LTO_OPT_SUCCESS;
}

///Link all modules together and optimize them using IPO. Generate
/// native object file using OutputFilename
/// Return appropriate LTOStatus.
enum LTOStatus
LTO::optimizeModules(const std::string &OutputFilename,
                     std::vector<const char *> &exportList,
                     std::string &targetTriple,
                     bool saveTemps,
                     const char *FinalOutputFilename)
{
  if (modules.empty())
    return LTO_NO_WORK;

  std::ios::openmode io_mode = 
    std::ios::out | std::ios::trunc | std::ios::binary; 
  std::string *errMsg = NULL;
  Module *bigOne = modules[0];
  Linker theLinker("LinkTimeOptimizer", bigOne, false);
  for (unsigned i = 1, e = modules.size(); i != e; ++i)
    if (theLinker.LinkModules(bigOne, modules[i], errMsg))
      return LTO_MODULE_MERGE_FAILURE;
  //  all modules have been handed off to the linker.
  modules.clear();

  sys::Path FinalOutputPath(FinalOutputFilename);
  FinalOutputPath.eraseSuffix();

  if (saveTemps) {
    std::string tempFileName(FinalOutputPath.c_str());
    tempFileName += "0.bc";
    std::ofstream Out(tempFileName.c_str(), io_mode);
    WriteBitcodeToFile(bigOne, Out);
  }

  // Strip leading underscore because it was added to match names
  // seen by linker.
  for (unsigned i = 0, e = exportList.size(); i != e; ++i) {
    const char *name = exportList[i];
    NameToSymbolMap::iterator itr = allSymbols.find(name);
    if (itr != allSymbols.end())
      exportList[i] = allSymbols[name]->getName();
  }


  std::string ErrMsg;
  sys::Path TempDir = sys::Path::GetTemporaryDirectory(&ErrMsg);
  if (TempDir.isEmpty()) {
    cerr << "lto: " << ErrMsg << "\n";
    return LTO_WRITE_FAILURE;
  }
  sys::Path tmpAsmFilePath(TempDir);
  if (!tmpAsmFilePath.appendComponent("lto")) {
    cerr << "lto: " << ErrMsg << "\n";
    TempDir.eraseFromDisk(true);
    return LTO_WRITE_FAILURE;
  }
  if (tmpAsmFilePath.createTemporaryFileOnDisk(&ErrMsg)) {
    cerr << "lto: " << ErrMsg << "\n";
    TempDir.eraseFromDisk(true);
    return LTO_WRITE_FAILURE;
  }
  sys::RemoveFileOnSignal(tmpAsmFilePath);

  std::ofstream asmFile(tmpAsmFilePath.c_str(), io_mode);
  if (!asmFile.is_open() || asmFile.bad()) {
    if (tmpAsmFilePath.exists()) {
      tmpAsmFilePath.eraseFromDisk();
      TempDir.eraseFromDisk(true);
    }
    return LTO_WRITE_FAILURE;
  }

  enum LTOStatus status = optimize(bigOne, asmFile, exportList);
  asmFile.close();
  if (status != LTO_OPT_SUCCESS) {
    tmpAsmFilePath.eraseFromDisk();
    TempDir.eraseFromDisk(true);
    return status;
  }

  if (saveTemps) {
    std::string tempFileName(FinalOutputPath.c_str());
    tempFileName += "1.bc";
    std::ofstream Out(tempFileName.c_str(), io_mode);
    WriteBitcodeToFile(bigOne, Out);
  }

  targetTriple = bigOne->getTargetTriple();

  // Run GCC to assemble and link the program into native code.
  //
  // Note:
  //  We can't just assemble and link the file with the system assembler
  //  and linker because we don't know where to put the _start symbol.
  //  GCC mysteriously knows how to do it.
  const sys::Path gcc = sys::Program::FindProgramByName("gcc");
  if (gcc.isEmpty()) {
    tmpAsmFilePath.eraseFromDisk();
    TempDir.eraseFromDisk(true);
    return LTO_ASM_FAILURE;
  }

  std::vector<const char*> args;
  args.push_back(gcc.c_str());
  args.push_back("-c");
  args.push_back("-x");
  args.push_back("assembler");
  args.push_back("-o");
  args.push_back(OutputFilename.c_str());
  args.push_back(tmpAsmFilePath.c_str());
  args.push_back(0);

  if (sys::Program::ExecuteAndWait(gcc, &args[0], 0, 0, 1, 0, &ErrMsg)) {
    cerr << "lto: " << ErrMsg << "\n";
    return LTO_ASM_FAILURE;
  }

  tmpAsmFilePath.eraseFromDisk();
  TempDir.eraseFromDisk(true);

  return LTO_OPT_SUCCESS;
}

void LTO::printVersion() {
    cl::PrintVersionMessage();
}

/// Unused pure-virtual destructor. Must remain empty.
LinkTimeOptimizer::~LinkTimeOptimizer() {}

/// Destruct LTO. Delete all modules, symbols and target.
LTO::~LTO() {
  
  for (std::vector<Module *>::iterator itr = modules.begin(), e = modules.end();
       itr != e; ++itr)
    delete *itr;

  modules.clear();

  for (NameToSymbolMap::iterator itr = allSymbols.begin(), e = allSymbols.end(); 
       itr != e; ++itr)
    delete itr->second;

  allSymbols.clear();

  delete Target;
}
