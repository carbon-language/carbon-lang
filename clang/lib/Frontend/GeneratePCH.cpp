//===--- GeneratePCH.cpp - AST Consumer for PCH Generation ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CreatePCHGenerate function, which creates an
//  ASTConsume that generates a PCH file.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/PCHWriter.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/FileManager.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/System/Path.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace clang;
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN PCHGenerator : public SemaConsumer {
    const Preprocessor &PP;
    const char *isysroot;
    llvm::raw_ostream *Out;
    Sema *SemaPtr;
    MemorizeStatCalls *StatCalls; // owned by the FileManager

  public:
    explicit PCHGenerator(const Preprocessor &PP,
                          const char *isysroot,
                          llvm::raw_ostream *Out);
    virtual void InitializeSema(Sema &S) { SemaPtr = &S; }
    virtual void HandleTranslationUnit(ASTContext &Ctx);
  };
}

PCHGenerator::PCHGenerator(const Preprocessor &PP,
                           const char *isysroot,
                           llvm::raw_ostream *OS)
  : PP(PP), isysroot(isysroot), Out(OS), SemaPtr(0), StatCalls(0) {

  // Install a stat() listener to keep track of all of the stat()
  // calls.
  StatCalls = new MemorizeStatCalls;
  PP.getFileManager().addStatCache(StatCalls, /*AtBeginning=*/true);
}

void PCHGenerator::HandleTranslationUnit(ASTContext &Ctx) {
  if (PP.getDiagnostics().hasErrorOccurred())
    return;

  // Write the PCH contents into a buffer
  std::vector<unsigned char> Buffer;
  BitstreamWriter Stream(Buffer);
  PCHWriter Writer(Stream);

  // Emit the PCH file
  assert(SemaPtr && "No Sema?");
  Writer.WritePCH(*SemaPtr, StatCalls, isysroot);

  // Write the generated bitstream to "Out".
  Out->write((char *)&Buffer.front(), Buffer.size());

  // Make sure it hits disk now.
  Out->flush();
}

ASTConsumer *clang::CreatePCHGenerator(const Preprocessor &PP,
                                       llvm::raw_ostream *OS,
                                       const char *isysroot) {
  return new PCHGenerator(PP, isysroot, OS);
}
