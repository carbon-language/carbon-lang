//===- LTOBugPoint.cpp - Top-Level LTO BugPoint class ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class contains all of the shared state and information that is used by
// the LTO BugPoint tool to track down bit code files that cause errors.
//
//===----------------------------------------------------------------------===//

#include "LTOBugPoint.h"
#include "llvm/PassManager.h"
#include "llvm/ModuleProvider.h"
#include "llvm/CodeGen/FileWriters.h"
#include "llvm/Target/SubtargetFeature.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Config/config.h"
#include <fstream>
#include <iostream>

using namespace llvm;
using namespace Reloc;
/// LTOBugPoint -- Constructor. Popuate list of linker options and
/// list of linker input files.
LTOBugPoint::LTOBugPoint(std::istream &args, std::istream &ins) {

  // Read linker options. Order is important here.
  std::string option;
  while (getline(args, option))
    LinkerOptions.push_back(option);
  
  // Read linker input files. Order is important here.
  std::string inFile;
  while(getline(ins, inFile))
    LinkerInputFiles.push_back(inFile);

  TempDir = sys::Path::GetTemporaryDirectory();
}

LTOBugPoint::~LTOBugPoint() {
  TempDir.eraseFromDisk(true);
}

/// findTroubleMakers - Find minimum set of input files that causes error
/// identified by the script.
bool
LTOBugPoint::findTroubleMakers(SmallVector<std::string, 4> &TroubleMakers,
                               std::string &Script) {

  // First, build native object files set.
  bool bitcodeFileSeen = false;
  unsigned Size = LinkerInputFiles.size();
  for (unsigned I = 0; I < Size; ++I) {
    std::string &FileName = LinkerInputFiles[I];
    sys::Path InputFile(FileName.c_str());
    if (InputFile.isDynamicLibrary() || InputFile.isArchive()) {
      ErrMsg = "Unable to handle input file ";
      ErrMsg += FileName;
      return false;
    }
    else if (InputFile.isBitcodeFile()) {
      bitcodeFileSeen = true;
      if (getNativeObjectFile(FileName) == false)
        return false;
    }
    else
      NativeInputFiles.push_back(FileName);
  }

  if (!bitcodeFileSeen) {
    ErrMsg = "Unable to help!";
    ErrMsg += " Need at least one input file that contains llvm bitcode";
    return false;
  }

  return true;
}

/// getFeatureString - Return a string listing the features associated with the
/// target triple.
///
/// FIXME: This is an inelegant way of specifying the features of a
/// subtarget. It would be better if we could encode this information into the
/// IR.
std::string LTOBugPoint::getFeatureString(const char *TargetTriple) {
  SubtargetFeatures Features;

  if (strncmp(TargetTriple, "powerpc-apple-", 14) == 0) {
    Features.AddFeature("altivec", true);
  } else if (strncmp(TargetTriple, "powerpc64-apple-", 16) == 0) {
    Features.AddFeature("64bit", true);
    Features.AddFeature("altivec", true);
  }

  return Features.getString();
}

/// assembleBitcode - Generate assembly code from the module. Return false
/// in case of an error.
bool LTOBugPoint::assembleBitcode(llvm::Module *M, const char *AsmFileName) {
  std::string TargetTriple = M->getTargetTriple();
  std::string FeatureStr =
    getFeatureString(TargetTriple.c_str());

  const TargetMachineRegistry::entry* Registry =
    TargetMachineRegistry::getClosestStaticTargetForModule(
                                                       *M, ErrMsg);
  if ( Registry == NULL )
    return false;

  TargetMachine *Target = Registry->CtorFn(*M, FeatureStr.c_str());

  // If target supports exception handling then enable it now.
  if (Target->getTargetAsmInfo()->doesSupportExceptionHandling())
    ExceptionHandling = true;
  
  // FIXME
  Target->setRelocationModel(Reloc::PIC_);

  FunctionPassManager* CGPasses =
    new FunctionPassManager(new ExistingModuleProvider(M));
  
  CGPasses->add(new TargetData(*Target->getTargetData()));
  MachineCodeEmitter* mce = NULL;

  std::ofstream *Out = new std::ofstream(AsmFileName, std::ios::out);

  switch (Target->addPassesToEmitFile(*CGPasses, *Out,
                                      TargetMachine::AssemblyFile, true)) {
  case FileModel::MachOFile:
    mce = AddMachOWriter(*CGPasses, *Out, *Target);
    break;
  case FileModel::ElfFile:
    mce = AddELFWriter(*CGPasses, *Out, *Target);
    break;
  case FileModel::AsmFile:
    break;
  case FileModel::Error:
  case FileModel::None:
    ErrMsg = "target file type not supported";
    return false;
  }
  
  if (Target->addPassesToEmitFileFinish(*CGPasses, mce, true)) {
    ErrMsg = "target does not support generation of this file type";
    return false;
  }

  CGPasses->doInitialization();
  for (Module::iterator
         it = M->begin(), e = M->end(); it != e; ++it)
    if (!it->isDeclaration())
      CGPasses->run(*it);
  CGPasses->doFinalization();
  delete Out;
  return true;
}

/// getNativeObjectFile - Generate native object file based from llvm
/// bitcode file. Return false in case of an error.
bool LTOBugPoint::getNativeObjectFile(std::string &FileName) {

  std::auto_ptr<Module> M;
  MemoryBuffer *Buffer
    = MemoryBuffer::getFile(FileName.c_str(), &ErrMsg);
  if (!Buffer) {
    ErrMsg = "Unable to read ";
    ErrMsg += FileName;
    return false;
  }
  M.reset(ParseBitcodeFile(Buffer, &ErrMsg));
  std::string TargetTriple = M->getTargetTriple();

  sys::Path AsmFile(TempDir);
  if(AsmFile.createTemporaryFileOnDisk(false, &ErrMsg))
    return false;

  if (assembleBitcode(M.get(), AsmFile.c_str()) == false) {
    AsmFile.eraseFromDisk();
    return false;
  }

  sys::Path NativeFile(TempDir);
  if(NativeFile.createTemporaryFileOnDisk(false, &ErrMsg)) {
    AsmFile.eraseFromDisk();
    return false;
  }

  // find compiler driver
  const sys::Path gcc = sys::Program::FindProgramByName("gcc");
  if ( gcc.isEmpty() ) {
    ErrMsg = "can't locate gcc";
    AsmFile.eraseFromDisk();
    NativeFile.eraseFromDisk();
    return false;
  }

  // build argument list
  std::vector<const char*> args;
  args.push_back(gcc.c_str());
  if ( TargetTriple.find("darwin") != TargetTriple.size() ) {
    if (strncmp(TargetTriple.c_str(), "i686-apple-", 11) == 0) {
      args.push_back("-arch");
      args.push_back("i386");
    }
    else if (strncmp(TargetTriple.c_str(), "x86_64-apple-", 13) == 0) {
      args.push_back("-arch");
      args.push_back("x86_64");
    }
    else if (strncmp(TargetTriple.c_str(), "powerpc-apple-", 14) == 0) {
      args.push_back("-arch");
      args.push_back("ppc");
    }
    else if (strncmp(TargetTriple.c_str(), "powerpc64-apple-", 16) == 0) {
      args.push_back("-arch");
      args.push_back("ppc64");
    }
  }
  args.push_back("-c");
  args.push_back("-x");
  args.push_back("assembler");
  args.push_back("-o");
  args.push_back(NativeFile.c_str());
  args.push_back(AsmFile.c_str());
  args.push_back(0);
  
  // invoke assembler
  if (sys::Program::ExecuteAndWait(gcc, &args[0], 0, 0, 0, 0, &ErrMsg)) {
    ErrMsg = "error in assembly";
    AsmFile.eraseFromDisk();
    NativeFile.eraseFromDisk();
    return false;
  }

  AsmFile.eraseFromDisk();
  return true;
}
