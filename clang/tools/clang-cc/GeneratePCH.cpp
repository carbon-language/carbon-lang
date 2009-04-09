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

#include "clang/Frontend/PCHWriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/System/Path.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Streams.h"
#include <string>

using namespace clang;
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN PCHGenerator : public ASTConsumer {
    Diagnostic &Diags;
    std::string OutFile;

  public:
    explicit PCHGenerator(Diagnostic &Diags, const std::string &OutFile)
      : Diags(Diags), OutFile(OutFile) { }

    virtual void HandleTranslationUnit(ASTContext &Ctx);
  };
}

void PCHGenerator::HandleTranslationUnit(ASTContext &Ctx) {
  if (Diags.hasErrorOccurred())
    return;

 // Write the PCH contents into a buffer
  std::vector<unsigned char> Buffer;
  BitstreamWriter Stream(Buffer);
  PCHWriter Writer(Stream);

  // Emit the PCH file
  Writer.WritePCH(Ctx);

  // Open up the PCH file.
  std::string ErrMsg;
  llvm::raw_fd_ostream Out(OutFile.c_str(), true, ErrMsg);
  
  if (!ErrMsg.empty()) {
    llvm::errs() << "PCH error: " << ErrMsg << "\n";
    return;
  }

  // Write the generated bitstream to "Out".
  Out.write((char *)&Buffer.front(), Buffer.size());

  // Make sure it hits disk now.
  Out.flush();
}

namespace clang {

ASTConsumer *CreatePCHGenerator(Diagnostic &Diags,
                                const LangOptions &Features,
                                const std::string& InFile,
                                const std::string& OutFile) {
  return new PCHGenerator(Diags, OutFile);
}

}
