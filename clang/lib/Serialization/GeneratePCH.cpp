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
//  ASTConsumeR that generates a PCH file.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/ASTConsumers.h"
#include "clang/Serialization/ASTWriter.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/FileManager.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace clang;

PCHGenerator::PCHGenerator(const Preprocessor &PP,
                           bool Chaining,
                           const char *isysroot,
                           llvm::raw_ostream *OS)
  : PP(PP), isysroot(isysroot), Out(OS), SemaPtr(0),
    StatCalls(0), Stream(Buffer), Writer(Stream) {

  // Install a stat() listener to keep track of all of the stat()
  // calls.
  StatCalls = new MemorizeStatCalls;
  // If we have a chain, we want new stat calls only, so install the memorizer
  // *after* the already installed PCHReader's stat cache.
  PP.getFileManager().addStatCache(StatCalls,
    /*AtBeginning=*/!Chaining);
}

void PCHGenerator::HandleTranslationUnit(ASTContext &Ctx) {
  if (PP.getDiagnostics().hasErrorOccurred())
    return;

  // Emit the PCH file
  assert(SemaPtr && "No Sema?");
  Writer.WriteAST(*SemaPtr, StatCalls, isysroot);

  // Write the generated bitstream to "Out".
  Out->write((char *)&Buffer.front(), Buffer.size());

  // Make sure it hits disk now.
  Out->flush();

  // Free up some memory, in case the process is kept alive.
  Buffer.clear();
}

PCHDeserializationListener *PCHGenerator::GetPCHDeserializationListener() {
  return &Writer;
}
