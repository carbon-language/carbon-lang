//===- Parser.cpp - Main dispatch module for the Parser library -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library implements the functionality defined in llvm/AsmParser/Parser.h
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "LLParser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <system_error>
using namespace llvm;

bool llvm::parseAssemblyInto(std::unique_ptr<MemoryBuffer> F, Module &M,
                             SMDiagnostic &Err) {
  SourceMgr SM;
  StringRef Buf = F->getBuffer();
  SM.AddNewSourceBuffer(std::move(F), SMLoc());

  return LLParser(Buf, SM, Err, &M).Run();
}

std::unique_ptr<Module> llvm::parseAssembly(std::unique_ptr<MemoryBuffer> F,
                                            SMDiagnostic &Err,
                                            LLVMContext &Context) {
  std::unique_ptr<Module> M =
      make_unique<Module>(F->getBufferIdentifier(), Context);

  if (parseAssemblyInto(std::move(F), *M, Err))
    return nullptr;

  return std::move(M);
}

std::unique_ptr<Module> llvm::parseAssemblyFile(StringRef Filename,
                                                SMDiagnostic &Err,
                                                LLVMContext &Context) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = FileOrErr.getError()) {
    Err = SMDiagnostic(Filename, SourceMgr::DK_Error,
                       "Could not open input file: " + EC.message());
    return nullptr;
  }

  return parseAssembly(std::move(FileOrErr.get()), Err, Context);
}

std::unique_ptr<Module> llvm::parseAssemblyString(StringRef AsmString,
                                                  SMDiagnostic &Err,
                                                  LLVMContext &Context) {
  std::unique_ptr<MemoryBuffer> F(
      MemoryBuffer::getMemBuffer(AsmString, "<string>"));

  return parseAssembly(std::move(F), Err, Context);
}
