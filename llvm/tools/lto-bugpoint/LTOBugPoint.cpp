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
#include "llvm/Support/raw_ostream.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Config/config.h"
#include <fstream>
#include <iostream>

using namespace llvm;
using namespace Reloc;

/// printBitVector - Helper function.
static void printBitVector(BitVector &BV, const char *Title) {
  std::cerr << Title;
  for (unsigned i = 0, e = BV.size(); i < e; i++) {
    if (BV[i])
      std::cerr << " " << i;
  }
  std::cerr << "\n";
}

/// printBitVector - Helper function.
static void printBitVectorFiles(BitVector &BV, const char *Title,
                                SmallVector<std::string, 16> &InFiles) {
  std::cerr << Title << "\n";
  for (unsigned i = 0, e = BV.size(); i < e; i++) {
    if (BV[i])
      std::cerr << "\t" << InFiles[i] << "\n";
  }
}

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

  // FIXME - Use command line option to set this.
  findLinkingFailure = true;
}

LTOBugPoint::~LTOBugPoint() {
  TempDir.eraseFromDisk(true);
}

/// findTroubleMakers - Find minimum set of input files that causes error
/// identified by the script.
bool
LTOBugPoint::findTroubleMakers(SmallVector<std::string, 4> &TroubleMakers,
                               std::string &Script) {

  // Reproduce original error.
  if (!relinkProgram(LinkerInputFiles) && !findLinkingFailure) {
    ErrMsg = " Unable to reproduce original error!";
    return false;
  }

  if (!findLinkingFailure && !reproduceProgramError(Script)) {
    ErrMsg = " Unable to reproduce original error!";
    return false;
  }
    
  // Build native object files set.
  unsigned Size = LinkerInputFiles.size();
  BCFiles.resize(Size);
  ConfirmedClean.resize(Size);
  ConfirmedGuilty.resize(Size);
  for (unsigned I = 0; I < Size; ++I) {
    std::string &FileName = LinkerInputFiles[I];
    sys::Path InputFile(FileName.c_str());
    if (InputFile.isDynamicLibrary() || InputFile.isArchive()) {
      ErrMsg = "Unable to handle input file " + FileName;
      return false;
    }
    else if (InputFile.isBitcodeFile()) {
      BCFiles.set(I);
      if (getNativeObjectFile(FileName) == false)
        return false;
    }
    else {
      // Original native object input files are always clean.
      ConfirmedClean.set(I);
      NativeInputFiles.push_back(FileName);
    }
  }

  if (BCFiles.none()) {
    ErrMsg = "Unable to help!";
    ErrMsg = " Need at least one input file that contains llvm bitcode";
    return false;
  }

  // Try to reproduce error using native object files first. If the error
  // occurs then this is not a LTO error.
  if (!relinkProgram(NativeInputFiles))  {
    ErrMsg = " Unable to link the program using all native object files!";
    return false;
  }
  if (!findLinkingFailure && reproduceProgramError(Script) == true) {
    ErrMsg = " Unable to fix program error using all native object files!";
    return false;
  }

  printBitVector(BCFiles, "Initial set of llvm bitcode files");
  identifyTroubleMakers(BCFiles);
  printBitVectorFiles(ConfirmedGuilty, 
                      "Identified minimal set of bitcode files!",
                      LinkerInputFiles);
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

  std::string error;
  raw_ostream *Out = new raw_fd_ostream(AsmFileName, error);
  if (!error.empty()) {
    std::cerr << error << '\n';
    delete Out;
    return false;
  }

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
    ErrMsg = "Unable to read " + FileName;
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
  NativeInputFiles.push_back(NativeFile.c_str());
  return true;
}

/// relinkProgram - Relink program. Return false if linking fails.
bool LTOBugPoint::relinkProgram(llvm::SmallVector<std::string, 16> &InFiles) {
  if (InFiles.empty())
    return false;

  // Atleast three options: linker path, -o and output file name.
  if (LinkerOptions.size() < 3)
    return false;

  const sys::Path linker = sys::Program::FindProgramByName(LinkerOptions[0]);
  if (linker.isEmpty()) {
    ErrMsg = "can't locate linker";
    return false;
  }
    
  std::vector<const char*> Args;
  for (unsigned i = 0, e = LinkerOptions.size(); i < e; ++i)
    Args.push_back(LinkerOptions[i].c_str());

  for (unsigned i = 0, e = InFiles.size(); i < e; ++i)
    Args.push_back(InFiles[i].c_str());

  Args.push_back(0);
  
  if (sys::Program::ExecuteAndWait(linker, &Args[0], 0, 0, 0, 0, &ErrMsg)) {
      ErrMsg = "error while linking program";
      return false;
  }
  return true;
}

/// reproduceProgramError - Validate program using user provided script.
/// Return true if program error is reproduced.
bool LTOBugPoint::reproduceProgramError(std::string &Script) {

  const sys::Path validator = sys::Program::FindProgramByName(Script);
  if (validator.isEmpty()) {
    ErrMsg = "can't locate validation script";
    return false;
  }
    
  std::vector<const char*> Args;
  Args.push_back(Script.c_str());
  Args.push_back(0);

  int result = 
    sys::Program::ExecuteAndWait(validator, &Args[0], 0, 0, 0, 0, &ErrMsg);

  // Validation scrip returns non-zero if the error is reproduced.
  if (result > 0) 
    // Able to reproduce program error.
    return true;

  else if (result < 0)
    // error occured while running validation script. ErrMsg contains error
    // description.
    return false;

  return false;
}

/// identifyTroubleMakers - Identify set of bit code files that are causing
/// the error. This is a recursive function.
void LTOBugPoint::identifyTroubleMakers(llvm::BitVector &In) {

  assert (In.size() == LinkerInputFiles.size() 
          && "Invalid identifyTroubleMakers input!\n");

  printBitVector(In, "Processing files ");
  BitVector CandidateVector;
  CandidateVector.resize(LinkerInputFiles.size());

  // Process first half
  unsigned count = 0;
  for (unsigned i = 0, e =  In.size(); i < e; ++i) {
    if (!ConfirmedClean[i]) {
      count++;
      CandidateVector.set(i);
    }
    if (count >= In.count()/2)
      break;
  }

  if (CandidateVector.none())
    return;

  printBitVector(CandidateVector, "Candidate vector ");

  // Reproduce the error using native object files for candidate files.
  SmallVector<std::string, 16> CandidateFiles;
  for (unsigned i = 0, e = CandidateVector.size(); i < e; ++i) {
    if (CandidateVector[i] || ConfirmedClean[i])
      CandidateFiles.push_back(NativeInputFiles[i]);
    else
      CandidateFiles.push_back(LinkerInputFiles[i]);
  }

  bool result = relinkProgram(CandidateFiles);
  if (findLinkingFailure) {
    if (result == true) {
      // Candidate files are suspected.
      if (CandidateVector.count() == 1) {
        ConfirmedGuilty.set(CandidateVector.find_first());
        return;
      }
      else
        identifyTroubleMakers(CandidateVector);
    } else {
      // Candidate files are not causing this error.
      for (unsigned i = 0, e = CandidateVector.size(); i < e; ++i) {
        if (CandidateVector[i])
          ConfirmedClean.set(i);
      }
    }
  } else {
    std::cerr << "FIXME : Not yet implemented!\n";
  }

  // Process remaining cadidates
  CandidateVector.clear();
  CandidateVector.resize(LinkerInputFiles.size());
  for (unsigned i = 0, e = LinkerInputFiles.size(); i < e; ++i) {
    if (!ConfirmedClean[i] && !ConfirmedGuilty[i])
      CandidateVector.set(i);
  }
  identifyTroubleMakers(CandidateVector);
}
