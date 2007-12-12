//==--- SourceLocation.cpp - Compact identifier for Source Files -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

void SourceRange::Emit(llvm::Serializer& S) const {
  B.Emit(S);
  E.Emit(S);
}

SourceRange SourceRange::ReadVal(llvm::Deserializer& D) {
  SourceLocation A = SourceLocation::ReadVal(D);
  SourceLocation B = SourceLocation::ReadVal(D);
  return SourceRange(A,B);
}

FullSourceLoc FullSourceLoc::getLogicalLoc() {
  assert (isValid());
  return FullSourceLoc(SrcMgr->getLogicalLoc(Loc),*SrcMgr);
}

FullSourceLoc FullSourceLoc::getIncludeLoc() {
  assert (isValid());
  return FullSourceLoc(SrcMgr->getIncludeLoc(Loc),*SrcMgr);
}

unsigned FullSourceLoc::getLineNumber() {
  assert (isValid());
  return SrcMgr->getLineNumber(Loc);
}

unsigned FullSourceLoc::getColumnNumber() {
  assert (isValid());
  return SrcMgr->getColumnNumber(Loc);
}

const char* FullSourceLoc::getSourceName() const {
  assert (isValid());
  return SrcMgr->getSourceName(Loc);
}

const FileEntry* FullSourceLoc::getFileEntryForLoc() const { 
  assert (isValid());
  return SrcMgr->getFileEntryForLoc(Loc);
}

const char * FullSourceLoc::getCharacterData() const {
  assert (isValid());
  return SrcMgr->getCharacterData(Loc);
}

const llvm::MemoryBuffer* FullSourceLoc::getBuffer() const {
  assert (isValid());
  return SrcMgr->getBuffer(Loc.getFileID());
}
