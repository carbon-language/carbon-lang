//===--- MachORewriteInstance.h - Instance of a rewriting process. --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface to control an instance of a macho binary rewriting process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_MACHO_REWRITE_INSTANCE_H
#define LLVM_TOOLS_LLVM_BOLT_MACHO_REWRITE_INSTANCE_H

#include "NameResolver.h"
#include "ProfileReaderBase.h"
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
  StringRef ToolPath;
  std::unique_ptr<BinaryContext> BC;

  NameResolver NR;

  std::unique_ptr<RuntimeDyld> RTDyld;

  std::unique_ptr<ToolOutputFile> Out;

  std::unique_ptr<ProfileReaderBase> ProfileReader;
  void preprocessProfileData();
  void processProfileDataPreCFG();
  void processProfileData();

  static StringRef getOrgSecPrefix() { return ".bolt.org"; }

  void mapInstrumentationSection(StringRef SectionName);
  void mapCodeSections();

  void adjustCommandLineOptions();
  void readSpecialSections();
  void discoverFileObjects();
  void disassembleFunctions();
  void buildFunctionsCFG();
  void postProcessFunctions();
  void runOptimizationPasses();
  void emitAndLink();

  void writeInstrumentationSection(StringRef SectionName, raw_pwrite_stream &OS);
  void rewriteFile();

public:
  MachORewriteInstance(object::MachOObjectFile *InputFile, StringRef ToolPath);
  ~MachORewriteInstance();

  Error setProfile(StringRef FileName);

  /// Run all the necessary steps to read, optimize and rewrite the binary.
  void run();
};

} // namespace bolt
} // namespace llvm

#endif
