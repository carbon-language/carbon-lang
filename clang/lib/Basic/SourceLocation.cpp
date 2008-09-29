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

void SourceRange::Emit(llvm::Serializer& S) const {
  B.Emit(S);
  E.Emit(S);
}

SourceRange SourceRange::ReadVal(llvm::Deserializer& D) {
  SourceLocation A = SourceLocation::ReadVal(D);
  SourceLocation B = SourceLocation::ReadVal(D);
  return SourceRange(A,B);
}

FullSourceLoc FullSourceLoc::getLogicalLoc() const {
  assert (isValid());
  return FullSourceLoc(SrcMgr->getLogicalLoc(Loc), *SrcMgr);
}

FullSourceLoc FullSourceLoc::getPhysicalLoc() const {
  assert (isValid());
  return FullSourceLoc(SrcMgr->getPhysicalLoc(Loc), *SrcMgr);
}

FullSourceLoc FullSourceLoc::getIncludeLoc() const {
  assert (isValid());
  return FullSourceLoc(SrcMgr->getIncludeLoc(Loc), *SrcMgr);
}

unsigned FullSourceLoc::getLineNumber() const {
  assert (isValid());
  return SrcMgr->getLineNumber(Loc);
}

unsigned FullSourceLoc::getColumnNumber() const {
  assert (isValid());
  return SrcMgr->getColumnNumber(Loc);
}


unsigned FullSourceLoc::getLogicalLineNumber() const {
  assert (isValid());
  return SrcMgr->getLogicalLineNumber(Loc);
}

unsigned FullSourceLoc::getLogicalColumnNumber() const {
  assert (isValid());
  return SrcMgr->getLogicalColumnNumber(Loc);
}

unsigned FullSourceLoc::getPhysicalLineNumber() const {
  assert (isValid());
  return SrcMgr->getPhysicalLineNumber(Loc);
}

unsigned FullSourceLoc::getPhysicalColumnNumber() const {
  assert (isValid());
  return SrcMgr->getPhysicalColumnNumber(Loc);
}

const char* FullSourceLoc::getSourceName() const {
  assert (isValid());
  return SrcMgr->getSourceName(Loc);
}

const FileEntry* FullSourceLoc::getFileEntryForLoc() const { 
  assert (isValid());
  return SrcMgr->getFileEntryForLoc(Loc);
}

bool FullSourceLoc::isInSystemHeader() const {
  assert (isValid());
  return SrcMgr->isInSystemHeader(Loc);
}


const char * FullSourceLoc::getCharacterData() const {
  assert (isValid());
  return SrcMgr->getCharacterData(Loc);
}

const llvm::MemoryBuffer* FullSourceLoc::getBuffer() const {
  assert (isValid());
  return SrcMgr->getBuffer(Loc.getFileID());
}

unsigned FullSourceLoc::getCanonicalFileID() const {
  return SrcMgr->getCanonicalFileID(Loc);
}

void FullSourceLoc::dump() const {
  if (!isValid()) {
    fprintf(stderr, "Invalid Loc\n");
    return;
  }
  
  if (isFileID()) {
    // The logical and physical pos is identical for file locs.
    fprintf(stderr, "File Loc from '%s': %d: %d\n",
            getSourceName(), getLogicalLineNumber(),
            getLogicalColumnNumber());
  } else {
    fprintf(stderr, "Macro Loc (\n  Physical: ");
    getPhysicalLoc().dump();
    fprintf(stderr, "  Logical: ");
    getLogicalLoc().dump();
    fprintf(stderr, ")\n");
  }
}
