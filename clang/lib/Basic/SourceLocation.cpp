//==--- SourceLocation.cpp - Compact identifier for Source Files -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines serialization methods for the SourceLocation class.
//  This file defines accessor methods for the FullSourceLoc class.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"
using namespace clang;

void SourceLocation::Emit(llvm::Serializer& S) const {
  S.EmitInt(getRawEncoding());  
}

SourceLocation SourceLocation::ReadVal(llvm::Deserializer& D) {
  return SourceLocation::getFromRawEncoding(D.ReadInt());   
}

void SourceLocation::dump(const SourceManager &SM) const {
  if (!isValid()) {
    fprintf(stderr, "<invalid loc>");
    return;
  }
  
  if (isFileID()) {
    PresumedLoc PLoc = SM.getPresumedLoc(*this);
    
    // The instantiation and spelling pos is identical for file locs.
    fprintf(stderr, "%s:%d:%d",
            PLoc.getFilename(), PLoc.getLine(), PLoc.getColumn());
    return;
  }
  
  SM.getInstantiationLoc(*this).dump(SM);
  
  fprintf(stderr, " <Spelling=");
  SM.getSpellingLoc(*this).dump(SM);
  fprintf(stderr, ">");
}


void SourceRange::Emit(llvm::Serializer& S) const {
  B.Emit(S);
  E.Emit(S);
}

SourceRange SourceRange::ReadVal(llvm::Deserializer& D) {
  SourceLocation A = SourceLocation::ReadVal(D);
  SourceLocation B = SourceLocation::ReadVal(D);
  return SourceRange(A,B);
}

FileID FullSourceLoc::getFileID() const {
  assert(isValid());
  return SrcMgr->getFileID(*this);
}


FullSourceLoc FullSourceLoc::getInstantiationLoc() const {
  assert(isValid());
  return FullSourceLoc(SrcMgr->getInstantiationLoc(*this), *SrcMgr);
}

FullSourceLoc FullSourceLoc::getSpellingLoc() const {
  assert(isValid());
  return FullSourceLoc(SrcMgr->getSpellingLoc(*this), *SrcMgr);
}

unsigned FullSourceLoc::getLineNumber() const {
  assert(isValid());
  return SrcMgr->getLineNumber(*this);
}

unsigned FullSourceLoc::getColumnNumber() const {
  assert(isValid());
  return SrcMgr->getColumnNumber(*this);
}


unsigned FullSourceLoc::getInstantiationLineNumber() const {
  assert(isValid());
  return SrcMgr->getInstantiationLineNumber(*this);
}

unsigned FullSourceLoc::getInstantiationColumnNumber() const {
  assert(isValid());
  return SrcMgr->getInstantiationColumnNumber(*this);
}

unsigned FullSourceLoc::getSpellingLineNumber() const {
  assert(isValid());
  return SrcMgr->getSpellingLineNumber(*this);
}

unsigned FullSourceLoc::getSpellingColumnNumber() const {
  assert(isValid());
  return SrcMgr->getSpellingColumnNumber(*this);
}

bool FullSourceLoc::isInSystemHeader() const {
  assert(isValid());
  return SrcMgr->isInSystemHeader(*this);
}

const char *FullSourceLoc::getCharacterData() const {
  assert(isValid());
  return SrcMgr->getCharacterData(*this);
}

const llvm::MemoryBuffer* FullSourceLoc::getBuffer() const {
  assert(isValid());
  return SrcMgr->getBuffer(SrcMgr->getFileID(*this));
}

