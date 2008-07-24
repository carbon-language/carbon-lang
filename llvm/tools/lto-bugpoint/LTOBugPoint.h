//===- LTOBugPoint.h - Top-Level LTO BugPoint class -------------*- C++ -*-===//
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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Module.h"
#include "llvm/System/Path.h"
#include <string>
#include <fstream>

class LTOBugPoint {
 public:

  LTOBugPoint(std::istream &args, std::istream &ins);
  ~LTOBugPoint();

  /// findTroubleMakers - Find minimum set of input files that causes error
  /// identified by the script.
  bool findTroubleMakers(llvm::SmallVector<std::string, 4> &TroubleMakers,
			std::string &Script);

  /// getNativeObjectFile - Generate native object file based from llvm
  /// bitcode file. Return false in case of an error. Generated native
  /// object file is inserted in to the NativeInputFiles list.
  bool getNativeObjectFile(std::string &FileName);

  std::string &getErrMsg() { return ErrMsg; }

 private:
  /// LinkerInputFiles - This is a list of linker input files. Once populated
  /// this list is not modified.
  llvm::SmallVector<std::string, 16> LinkerInputFiles;

  /// LinkerOptions - List of linker command line options.
  llvm::SmallVector<std::string, 16> LinkerOptions;

  /// NativeInputFiles - This is a list of input files that are not llvm
  /// bitcode files. The order in this list is important. The a file
  /// in LinkerInputFiles at index 4 is a llvm bitcode file then the file
  /// at index 4 in NativeInputFiles is corresponding native object file.
  llvm::SmallVector<std::string, 16> NativeInputFiles;

  /// BCFiles - This bit vector tracks input bitcode files.
  llvm::BitVector BCFiles;

  /// ConfirmedClean - This bit vector tracks input files that are confirmed
  /// to be clean.
  llvm::BitVector ConfirmedClean;

  /// ConfirmedGuilty - This bit vector tracks input files that are confirmed
  /// to contribute to the bug being investigated.
  llvm::BitVector ConfirmedGuilty;
  std::string getFeatureString(const char *TargetTriple);
  std::string ErrMsg;

  llvm::sys::Path TempDir;

  /// findLinkingFailure - If true, investigate link failure bugs when
  /// one or more linker input files are llvm bitcode files. If false,
  /// investigate optimization or code generation bugs in LTO mode.
  bool findLinkingFailure;

private:
  /// assembleBitcode - Generate assembly code from the module. Return false
  /// in case of an error.
  bool assembleBitcode(llvm::Module *M, const char *AsmFileName);

  /// relinkProgram - Relink program. Return false if linking fails.
  bool relinkProgram(llvm::SmallVector<std::string, 16> &InputFiles);

  /// reproduceProgramError - Validate program using user provided script.
  bool reproduceProgramError(std::string &Script);

  /// identifyTroubleMakers - Identify set of inputs from the given 
  /// bitvector that are causing the bug under investigation.
  void identifyTroubleMakers(llvm::BitVector &In);
};
