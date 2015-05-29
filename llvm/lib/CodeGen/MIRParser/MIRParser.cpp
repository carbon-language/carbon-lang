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

#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/LineIterator.h"
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

  /// Try to parse the optional LLVM module and the machine functions in the MIR
  /// file.
  ///
  /// Return null if an error occurred.
  std::unique_ptr<Module> parse(SMDiagnostic &Error);

  /// Parse the machine function in the current YAML document.
  ///
  /// Return true if an error occurred.
  bool parseMachineFunction(yaml::Input &In);

private:
  /// Return a MIR diagnostic converted from an LLVM assembly diagnostic.
  SMDiagnostic diagFromLLVMAssemblyDiag(const SMDiagnostic &Error,
                                        SMRange SourceRange);
};

} // end anonymous namespace

MIRParserImpl::MIRParserImpl(std::unique_ptr<MemoryBuffer> Contents,
                             StringRef Filename, LLVMContext &Context)
    : SM(), Filename(Filename), Context(Context) {
  SM.AddNewSourceBuffer(std::move(Contents), SMLoc());
}

static void handleYAMLDiag(const SMDiagnostic &Diag, void *Context) {
  *reinterpret_cast<SMDiagnostic *>(Context) = Diag;
}

std::unique_ptr<Module> MIRParserImpl::parse(SMDiagnostic &Error) {
  yaml::Input In(SM.getMemoryBuffer(SM.getMainFileID())->getBuffer(),
                 /*Ctxt=*/nullptr, handleYAMLDiag, &Error);

  if (!In.setCurrentDocument()) {
    if (!Error.getMessage().empty())
      return nullptr;
    // Create an empty module when the MIR file is empty.
    return llvm::make_unique<Module>(Filename, Context);
  }

  std::unique_ptr<Module> M;
  // Parse the block scalar manually so that we can return unique pointer
  // without having to go trough YAML traits.
  if (const auto *BSN =
          dyn_cast_or_null<yaml::BlockScalarNode>(In.getCurrentNode())) {
    M = parseAssembly(MemoryBufferRef(BSN->getValue(), Filename), Error,
                      Context);
    if (!M) {
      Error = diagFromLLVMAssemblyDiag(Error, BSN->getSourceRange());
      return M;
    }
    In.nextDocument();
    if (!In.setCurrentDocument())
      return M;
  } else {
    // Create an new, empty module.
    M = llvm::make_unique<Module>(Filename, Context);
  }

  // Parse the machine functions.
  do {
    if (parseMachineFunction(In))
      return nullptr;
    In.nextDocument();
  } while (In.setCurrentDocument());

  return M;
}

bool MIRParserImpl::parseMachineFunction(yaml::Input &In) {
  yaml::MachineFunction MF;
  yaml::yamlize(In, MF, false);
  if (In.error())
    return true;
  // TODO: Initialize the real machine function with the state in the yaml
  // machine function later on.
  return false;
}

SMDiagnostic MIRParserImpl::diagFromLLVMAssemblyDiag(const SMDiagnostic &Error,
                                                     SMRange SourceRange) {
  assert(SourceRange.isValid());

  // Translate the location of the error from the location in the llvm IR string
  // to the corresponding location in the MIR file.
  auto LineAndColumn = SM.getLineAndColumn(SourceRange.Start);
  unsigned Line = LineAndColumn.first + Error.getLineNo() - 1;
  unsigned Column = Error.getColumnNo();
  StringRef LineStr = Error.getLineContents();
  SMLoc Loc = Error.getLoc();

  // Get the full line and adjust the column number by taking the indentation of
  // LLVM IR into account.
  for (line_iterator L(*SM.getMemoryBuffer(SM.getMainFileID()), false), E;
       L != E; ++L) {
    if (L.line_number() == Line) {
      LineStr = *L;
      Loc = SMLoc::getFromPointer(LineStr.data());
      auto Indent = LineStr.find(Error.getLineContents());
      if (Indent != StringRef::npos)
        Column += Indent;
      break;
    }
  }

  return SMDiagnostic(SM, Loc, Filename, Line, Column, Error.getKind(),
                      Error.getMessage(), LineStr, Error.getRanges(),
                      Error.getFixIts());
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
  return Parser.parse(Error);
}
