//===---------- PlistSupport.h - Plist Output Utilities ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PLISTSUPPORT_H
#define LLVM_CLANG_PLISTSUPPORT_H

#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace markup {
typedef llvm::DenseMap<FileID, unsigned> FIDMap;

static const char *PlistHeader =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<!DOCTYPE plist PUBLIC \"-//Apple Computer//DTD PLIST 1.0//EN\" "
    "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
    "<plist version=\"1.0\">\n";

static void AddFID(FIDMap &FIDs, SmallVectorImpl<FileID> &V,
                   const SourceManager &SM, SourceLocation L) {
  FileID FID = SM.getFileID(SM.getExpansionLoc(L));
  FIDMap::iterator I = FIDs.find(FID);
  if (I != FIDs.end())
    return;
  FIDs[FID] = V.size();
  V.push_back(FID);
}

static unsigned GetFID(const FIDMap &FIDs, const SourceManager &SM,
                       SourceLocation L) {
  FileID FID = SM.getFileID(SM.getExpansionLoc(L));
  FIDMap::const_iterator I = FIDs.find(FID);
  assert(I != FIDs.end());
  return I->second;
}

static raw_ostream &Indent(raw_ostream &o, const unsigned indent) {
  for (unsigned i = 0; i < indent; ++i)
    o << ' ';
  return o;
}

static void EmitLocation(raw_ostream &o, const SourceManager &SM,
                         const LangOptions &LangOpts, SourceLocation L,
                         const FIDMap &FM, unsigned indent,
                         bool extend = false) {
  FullSourceLoc Loc(SM.getExpansionLoc(L), const_cast<SourceManager &>(SM));

  // Add in the length of the token, so that we cover multi-char tokens.
  unsigned offset =
      extend ? Lexer::MeasureTokenLength(Loc, SM, LangOpts) - 1 : 0;

  Indent(o, indent) << "<dict>\n";
  Indent(o, indent) << " <key>line</key><integer>"
                    << Loc.getExpansionLineNumber() << "</integer>\n";
  Indent(o, indent) << " <key>col</key><integer>"
                    << Loc.getExpansionColumnNumber() + offset
                    << "</integer>\n";
  Indent(o, indent) << " <key>file</key><integer>" << GetFID(FM, SM, Loc)
                    << "</integer>\n";
  Indent(o, indent) << "</dict>\n";
}

static inline void EmitRange(raw_ostream &o, const SourceManager &SM,
                             const LangOptions &LangOpts, CharSourceRange R,
                             const FIDMap &FM, unsigned indent) {
  Indent(o, indent) << "<array>\n";
  EmitLocation(o, SM, LangOpts, R.getBegin(), FM, indent + 1);
  EmitLocation(o, SM, LangOpts, R.getEnd(), FM, indent + 1, R.isTokenRange());
  Indent(o, indent) << "</array>\n";
}

static raw_ostream &EmitString(raw_ostream &o, StringRef s) {
  o << "<string>";
  for (StringRef::const_iterator I = s.begin(), E = s.end(); I != E; ++I) {
    char c = *I;
    switch (c) {
    default:
      o << c;
      break;
    case '&':
      o << "&amp;";
      break;
    case '<':
      o << "&lt;";
      break;
    case '>':
      o << "&gt;";
      break;
    case '\'':
      o << "&apos;";
      break;
    case '\"':
      o << "&quot;";
      break;
    }
  }
  o << "</string>";
  return o;
}
}
}

#endif
