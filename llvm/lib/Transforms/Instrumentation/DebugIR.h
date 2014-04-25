//===- llvm/Transforms/Instrumentation/DebugIR.h - Interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface of the DebugIR pass. For most users,
// including Instrumentation.h and calling createDebugIRPass() is sufficient and
// there is no need to include this file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_DEBUGIR_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_DEBUGIR_H

#include "llvm/Pass.h"

namespace llvm {

class DebugIR : public llvm::ModulePass {
  /// If true, write a source file to disk.
  bool WriteSourceToDisk;

  /// Hide certain (non-essential) debug information (only relevant if
  /// createSource is true.
  bool HideDebugIntrinsics;
  bool HideDebugMetadata;

  /// The location of the source file.
  std::string Directory;
  std::string Filename;

  /// True if a temporary file name was generated.
  bool GeneratedPath;

  /// True if the file name was read from the Module.
  bool ParsedPath;

public:
  static char ID;

  const char *getPassName() const override { return "DebugIR"; }

  /// Generate a file on disk to be displayed in a debugger. If Filename and
  /// Directory are empty, a temporary path will be generated.
  DebugIR(bool HideDebugIntrinsics, bool HideDebugMetadata,
          llvm::StringRef Directory, llvm::StringRef Filename)
      : ModulePass(ID), WriteSourceToDisk(true),
        HideDebugIntrinsics(HideDebugIntrinsics),
        HideDebugMetadata(HideDebugMetadata), Directory(Directory),
        Filename(Filename), GeneratedPath(false), ParsedPath(false) {}

  /// Modify input in-place; do not generate additional files, and do not hide
  /// any debug intrinsics/metadata that might be present.
  DebugIR()
      : ModulePass(ID), WriteSourceToDisk(false), HideDebugIntrinsics(false),
        HideDebugMetadata(false), GeneratedPath(false), ParsedPath(false) {}

  /// Run pass on M and set Path to the source file path in the output module.
  bool runOnModule(llvm::Module &M, std::string &Path);
  bool runOnModule(llvm::Module &M) override;

private:

  /// Returns the concatenated Directory + Filename, without error checking
  std::string getPath();

  /// Attempts to read source information from debug information in M, and if
  /// that fails, from M's identifier. Returns true on success, false otherwise.
  bool getSourceInfo(const llvm::Module &M);

  /// Replace the extension of Filename with NewExtension, and return true if
  /// successful. Return false if extension could not be found or Filename is
  /// empty.
  bool updateExtension(llvm::StringRef NewExtension);

  /// Generate a temporary filename and open an fd
  void generateFilename(std::unique_ptr<int> &fd);

  /// Creates DWARF CU/Subroutine metadata
  void createDebugInfo(llvm::Module &M,
                       std::unique_ptr<llvm::Module> &DisplayM);

  /// Returns true if either Directory or Filename is missing, false otherwise.
  bool isMissingPath();

  /// Write M to disk, optionally passing in an fd to an open file which is
  /// closed by this function after writing. If no fd is specified, a new file
  /// is opened, written, and closed.
  void writeDebugBitcode(const llvm::Module *M, int *fd = nullptr);
};

} // llvm namespace

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_DEBUGIR_H
