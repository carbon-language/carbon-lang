//===- MIRParser.cpp - MIR serialization format parser implementation -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the class that parses the optional LLVM IR and machine
// functions that are stored in MIR files.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MIR/MIRParser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include <memory>

using namespace llvm;

namespace {

/// This class implements the parsing of LLVM IR that's embedded inside a MIR
/// file.
class MIRParserImpl {
  SourceMgr SM;
  StringRef Filename;
  LLVMContext &Context;

public:
  MIRParserImpl(std::unique_ptr<MemoryBuffer> Contents, StringRef Filename,
                LLVMContext &Context);

  /// Try to parse the optional LLVM module in the MIR file.
  ///
  /// Return null if an error occurred while parsing the LLVM module.
  std::unique_ptr<Module> parseLLVMModule(SMDiagnostic &Error);
};

} // end anonymous namespace

MIRParserImpl::MIRParserImpl(std::unique_ptr<MemoryBuffer> Contents,
                             StringRef Filename, LLVMContext &Context)
    : SM(), Filename(Filename), Context(Context) {
  SM.AddNewSourceBuffer(std::move(Contents), SMLoc());
}

std::unique_ptr<Module> MIRParserImpl::parseLLVMModule(SMDiagnostic &Error) {
  yaml::Input In(SM.getMemoryBuffer(SM.getMainFileID())->getBuffer());

  // Parse the block scalar manually so that we can return unique pointer
  // without having to go trough YAML traits.
  if (In.setCurrentDocument()) {
    if (const auto *BSN =
            dyn_cast_or_null<yaml::BlockScalarNode>(In.getCurrentNode())) {
      return parseAssembly(MemoryBufferRef(BSN->getValue(), Filename), Error,
                           Context);
    }
  }

  // Create an new, empty module.
  return llvm::make_unique<Module>(Filename, Context);
}

std::unique_ptr<Module> llvm::parseMIRFile(StringRef Filename,
                                           SMDiagnostic &Error,
                                           LLVMContext &Context) {
  auto FileOrErr = MemoryBuffer::getFile(Filename);
  if (std::error_code EC = FileOrErr.getError()) {
    Error = SMDiagnostic(Filename, SourceMgr::DK_Error,
                         "Could not open input file: " + EC.message());
    return std::unique_ptr<Module>();
  }
  return parseMIR(std::move(FileOrErr.get()), Error, Context);
}

std::unique_ptr<Module> llvm::parseMIR(std::unique_ptr<MemoryBuffer> Contents,
                                       SMDiagnostic &Error,
                                       LLVMContext &Context) {
  auto Filename = Contents->getBufferIdentifier();
  MIRParserImpl Parser(std::move(Contents), Filename, Context);
  return Parser.parseLLVMModule(Error);
}
