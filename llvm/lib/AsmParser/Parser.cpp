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

Module *llvm::ParseAssembly(MemoryBuffer *F,
                            Module *M,
                            SMDiagnostic &Err,
                            LLVMContext &Context) {
  SourceMgr SM;
  SM.AddNewSourceBuffer(F, SMLoc());

  // If we are parsing into an existing module, do it.
  if (M)
    return LLParser(F, SM, Err, M).Run() ? nullptr : M;

  // Otherwise create a new module.
  std::unique_ptr<Module> M2(new Module(F->getBufferIdentifier(), Context));
  if (LLParser(F, SM, Err, M2.get()).Run())
    return nullptr;
  return M2.release();
}

Module *llvm::ParseAssemblyFile(const std::string &Filename, SMDiagnostic &Err,
                                LLVMContext &Context) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = FileOrErr.getError()) {
    Err = SMDiagnostic(Filename, SourceMgr::DK_Error,
                       "Could not open input file: " + EC.message());
    return nullptr;
  }

  return ParseAssembly(FileOrErr.get().release(), nullptr, Err, Context);
}

Module *llvm::ParseAssemblyString(const char *AsmString, Module *M,
                                  SMDiagnostic &Err, LLVMContext &Context) {
  MemoryBuffer *F =
      MemoryBuffer::getMemBuffer(StringRef(AsmString), "<string>");

  return ParseAssembly(F, M, Err, Context);
}
