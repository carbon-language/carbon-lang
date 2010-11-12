//==--- SourceLocation.cpp - Compact identifier for Source Files -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines accessor methods for the FullSourceLoc class.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
using namespace clang;

//===----------------------------------------------------------------------===//
// PrettyStackTraceLoc
//===----------------------------------------------------------------------===//

void PrettyStackTraceLoc::print(llvm::raw_ostream &OS) const {
  if (Loc.isValid()) {
    Loc.print(OS, SM);
    OS << ": ";
  }
  OS << Message << '\n';
}

//===----------------------------------------------------------------------===//
// SourceLocation
//===----------------------------------------------------------------------===//

void SourceLocation::print(llvm::raw_ostream &OS, const SourceManager &SM)const{
  if (!isValid()) {
    OS << "<invalid loc>";
    return;
  }

  if (isFileID()) {
    PresumedLoc PLoc = SM.getPresumedLoc(*this);
    
    if (PLoc.isInvalid()) {
      OS << "<invalid>";
      return;
    }
    // The instantiation and spelling pos is identical for file locs.
    OS << PLoc.getFilename() << ':' << PLoc.getLine()
       << ':' << PLoc.getColumn();
    return;
  }

  SM.getInstantiationLoc(*this).print(OS, SM);

  OS << " <Spelling=";
  SM.getSpellingLoc(*this).print(OS, SM);
  OS << '>';
}

void SourceLocation::dump(const SourceManager &SM) const {
  print(llvm::errs(), SM);
}

//===----------------------------------------------------------------------===//
// FullSourceLoc
//===----------------------------------------------------------------------===//

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

unsigned FullSourceLoc::getInstantiationLineNumber(bool *Invalid) const {
  assert(isValid());
  return SrcMgr->getInstantiationLineNumber(*this, Invalid);
}

unsigned FullSourceLoc::getInstantiationColumnNumber(bool *Invalid) const {
  assert(isValid());
  return SrcMgr->getInstantiationColumnNumber(*this, Invalid);
}

unsigned FullSourceLoc::getSpellingLineNumber(bool *Invalid) const {
  assert(isValid());
  return SrcMgr->getSpellingLineNumber(*this, Invalid);
}

unsigned FullSourceLoc::getSpellingColumnNumber(bool *Invalid) const {
  assert(isValid());
  return SrcMgr->getSpellingColumnNumber(*this, Invalid);
}

bool FullSourceLoc::isInSystemHeader() const {
  assert(isValid());
  return SrcMgr->isInSystemHeader(*this);
}

const char *FullSourceLoc::getCharacterData(bool *Invalid) const {
  assert(isValid());
  return SrcMgr->getCharacterData(*this, Invalid);
}

const llvm::MemoryBuffer* FullSourceLoc::getBuffer(bool *Invalid) const {
  assert(isValid());
  return SrcMgr->getBuffer(SrcMgr->getFileID(*this), Invalid);
}

llvm::StringRef FullSourceLoc::getBufferData(bool *Invalid) const {
  return getBuffer(Invalid)->getBuffer();
}

std::pair<FileID, unsigned> FullSourceLoc::getDecomposedLoc() const {
  return SrcMgr->getDecomposedLoc(*this);
}
