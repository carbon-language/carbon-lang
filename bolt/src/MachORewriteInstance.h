//===--- MachORewriteInstance.h - Instance of a rewriting process. --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to control an instance of a macho binary rewriting process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_MACHO_REWRITE_INSTANCE_H
#define LLVM_TOOLS_LLVM_BOLT_MACHO_REWRITE_INSTANCE_H

#include "llvm/Object/MachO.h"
#include <memory>

namespace llvm {
namespace bolt {

class BinaryContext;
class DataReader;

class MachORewriteInstance {
  object::MachOObjectFile *InputFile;
  std::unique_ptr<BinaryContext> BC;

  void readSpecialSections();
  void discoverFileObjects();
  void disassembleFunctions();

public:
  MachORewriteInstance(object::MachOObjectFile *InputFile, DataReader &DR);
  ~MachORewriteInstance();

  /// Run all the necessary steps to read, optimize and rewrite the binary.
  void run();
};

} // namespace bolt
} // namespace llvm

#endif
