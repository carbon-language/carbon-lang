//===--- TranslationUnit.cpp - Abstraction for Translation Units ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// FIXME: This should eventually be moved out of the driver, or replaced
//        with its eventual successor.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/TranslationUnit.h"

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/AST.h"

#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Path.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/DenseSet.h"

using namespace clang;


TranslationUnit::~TranslationUnit() {
  if (OwnsMetaData && Context) {
    // The ASTContext object has the sole references to the IdentifierTable
    // Selectors, and the Target information.  Go and delete them, since
    // the TranslationUnit effectively owns them.
    delete &Context->Idents;
    delete &Context->Selectors;
    delete &Context->Target;
    delete Context;
  }  
}

bool clang::EmitASTBitcodeBuffer(const ASTContext &Ctx, 
                                 std::vector<unsigned char>& Buffer) {
  // Create bitstream.
  llvm::BitstreamWriter Stream(Buffer);
  
  // Emit the preamble.
  Stream.Emit((unsigned)'B', 8);
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit(0xC, 4);
  Stream.Emit(0xF, 4);
  Stream.Emit(0xE, 4);
  Stream.Emit(0x0, 4);
  
  { 
    // Create serializer.  Placing it in its own scope assures any necessary
    // finalization of bits to the buffer in the serializer's dstor.    
    llvm::Serializer Sezr(Stream);  
    
    // Emit the translation unit.
    Ctx.EmitAll(Sezr);
  }
  
  return true;
}

TranslationUnit*
clang::ReadASTBitcodeBuffer(llvm::MemoryBuffer& MBuffer, FileManager& FMgr) {

  // Check if the file is of the proper length.
  if (MBuffer.getBufferSize() & 0x3) {
    // FIXME: Provide diagnostic: "Length should be a multiple of 4 bytes."
    return NULL;
  }
  
  // Create the bitstream reader.
  unsigned char *BufPtr = (unsigned char *) MBuffer.getBufferStart();
  llvm::BitstreamReader Stream(BufPtr,BufPtr+MBuffer.getBufferSize());
  
  if (Stream.Read(8) != 'B' ||
      Stream.Read(8) != 'C' ||
      Stream.Read(4) != 0xC ||
      Stream.Read(4) != 0xF ||
      Stream.Read(4) != 0xE ||
      Stream.Read(4) != 0x0) {
    // FIXME: Provide diagnostic.
    return NULL;
  }
  
  // Create the deserializer.
  llvm::Deserializer Dezr(Stream);
  
  return TranslationUnit::Create(Dezr,FMgr);
}

TranslationUnit* TranslationUnit::Create(llvm::Deserializer& Dezr,
                                         FileManager& FMgr) {
  
  // Create the translation unit object.
  TranslationUnit* TU = new TranslationUnit();
  
  TU->Context = ASTContext::CreateAll(Dezr, FMgr);
  
  return TU;
}
