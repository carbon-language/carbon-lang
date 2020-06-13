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

#include "NameResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/Object/MachO.h"
#include <memory>

namespace llvm {

class ToolOutputFile;

namespace orc {

class SymbolStringPool;
class ExecutionSession;
class RTDyldObjectLinkingLayer;

} // namespace orc

namespace bolt {

class BinaryContext;

class MachORewriteInstance {
  object::MachOObjectFile *InputFile;
  std::unique_ptr<BinaryContext> BC;

  NameResolver NR;

  std::unique_ptr<orc::SymbolStringPool> SSP;
  std::unique_ptr<orc::ExecutionSession> ES;
  std::unique_ptr<orc::RTDyldObjectLinkingLayer> OLT;

  std::unique_ptr<ToolOutputFile> Out;

  static StringRef getOrgSecPrefix() { return ".bolt.org"; }

  void mapCodeSections(orc::VModuleKey Key);

  void adjustCommandLineOptions();
  void readSpecialSections();
  void discoverFileObjects();
  void disassembleFunctions();
  void postProcessFunctions();
  void runOptimizationPasses();
  void emitAndLink();
  void rewriteFile();

public:
  explicit MachORewriteInstance(object::MachOObjectFile *InputFile);
  ~MachORewriteInstance();

  /// Run all the necessary steps to read, optimize and rewrite the binary.
  void run();
};

} // namespace bolt
} // namespace llvm

#endif
