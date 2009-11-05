//===--- PlistDiagnostics.cpp - Plist Diagnostics for Paths -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PlistDiagnostics object.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/PathDiagnosticClients.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Casting.h"
#include "llvm/System/Path.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
using namespace clang;
using llvm::cast;

typedef llvm::DenseMap<FileID, unsigned> FIDMap;

namespace clang {
  class Preprocessor;
}

namespace {
  class VISIBILITY_HIDDEN PlistDiagnostics : public PathDiagnosticClient {
    std::vector<const PathDiagnostic*> BatchedDiags;
    const std::string OutputFile;
    const LangOptions &LangOpts;
    llvm::OwningPtr<PathDiagnosticClient> SubPD;
  public:
    PlistDiagnostics(const std::string& prefix, const LangOptions &LangOpts,
                     PathDiagnosticClient *subPD);

    ~PlistDiagnostics() { FlushDiagnostics(NULL); }

    void FlushDiagnostics(llvm::SmallVectorImpl<std::string> *FilesMade);
    
    void HandlePathDiagnostic(const PathDiagnostic* D);
    
    virtual llvm::StringRef getName() const {
      return "PlistDiagnostics";
    }

    PathGenerationScheme getGenerationScheme() const;
    bool supportsLogicalOpControlFlow() const { return true; }
    bool supportsAllBlockEdges() const { return true; }
    virtual bool useVerboseDescription() const { return false; }
  };
} // end anonymous namespace

PlistDiagnostics::PlistDiagnostics(const std::string& output,
                                   const LangOptions &LO,
                                   PathDiagnosticClient *subPD)
  : OutputFile(output), LangOpts(LO), SubPD(subPD) {}

PathDiagnosticClient*
clang::CreatePlistDiagnosticClient(const std::string& s, Preprocessor *PP,
                                   PathDiagnosticClient *subPD) {
  return new PlistDiagnostics(s, PP->getLangOptions(), subPD);
}

PathDiagnosticClient::PathGenerationScheme
PlistDiagnostics::getGenerationScheme() const {
  if (const PathDiagnosticClient *PD = SubPD.get())
    return PD->getGenerationScheme();

  return Extensive;
}

static void AddFID(FIDMap &FIDs, llvm::SmallVectorImpl<FileID> &V,
                   const SourceManager* SM, SourceLocation L) {

  FileID FID = SM->getFileID(SM->getInstantiationLoc(L));
  FIDMap::iterator I = FIDs.find(FID);
  if (I != FIDs.end()) return;
  FIDs[FID] = V.size();
  V.push_back(FID);
}

static unsigned GetFID(const FIDMap& FIDs, const SourceManager &SM,
                       SourceLocation L) {
  FileID FID = SM.getFileID(SM.getInstantiationLoc(L));
  FIDMap::const_iterator I = FIDs.find(FID);
  assert(I != FIDs.end());
  return I->second;
}

static llvm::raw_ostream& Indent(llvm::raw_ostream& o, const unsigned indent) {
  for (unsigned i = 0; i < indent; ++i) o << ' ';
  return o;
}

static void EmitLocation(llvm::raw_ostream& o, const SourceManager &SM,
                         const LangOptions &LangOpts,
                         SourceLocation L, const FIDMap &FM,
                         unsigned indent, bool extend = false) {

  FullSourceLoc Loc(SM.getInstantiationLoc(L), const_cast<SourceManager&>(SM));

  // Add in the length of the token, so that we cover multi-char tokens.
  unsigned offset =
    extend ? Lexer::MeasureTokenLength(Loc, SM, LangOpts) - 1 : 0;

  Indent(o, indent) << "<dict>\n";
  Indent(o, indent) << " <key>line</key><integer>"
                    << Loc.getInstantiationLineNumber() << "</integer>\n";
  Indent(o, indent) << " <key>col</key><integer>"
                    << Loc.getInstantiationColumnNumber() + offset << "</integer>\n";
  Indent(o, indent) << " <key>file</key><integer>"
                    << GetFID(FM, SM, Loc) << "</integer>\n";
  Indent(o, indent) << "</dict>\n";
}

static void EmitLocation(llvm::raw_ostream& o, const SourceManager &SM,
                         const LangOptions &LangOpts,
                         const PathDiagnosticLocation &L, const FIDMap& FM,
                         unsigned indent, bool extend = false) {
  EmitLocation(o, SM, LangOpts, L.asLocation(), FM, indent, extend);
}

static void EmitRange(llvm::raw_ostream& o, const SourceManager &SM,
                      const LangOptions &LangOpts,
                      PathDiagnosticRange R, const FIDMap &FM,
                      unsigned indent) {
  Indent(o, indent) << "<array>\n";
  EmitLocation(o, SM, LangOpts, R.getBegin(), FM, indent+1);
  EmitLocation(o, SM, LangOpts, R.getEnd(), FM, indent+1, !R.isPoint);
  Indent(o, indent) << "</array>\n";
}

static llvm::raw_ostream& EmitString(llvm::raw_ostream& o,
                                     const std::string& s) {
  o << "<string>";
  for (std::string::const_iterator I=s.begin(), E=s.end(); I!=E; ++I) {
    char c = *I;
    switch (c) {
      default:   o << c; break;
      case '&':  o << "&amp;"; break;
      case '<':  o << "&lt;"; break;
      case '>':  o << "&gt;"; break;
      case '\'': o << "&apos;"; break;
      case '\"': o << "&quot;"; break;
    }
  }
  o << "</string>";
  return o;
}

static void ReportControlFlow(llvm::raw_ostream& o,
                              const PathDiagnosticControlFlowPiece& P,
                              const FIDMap& FM,
                              const SourceManager &SM,
                              const LangOptions &LangOpts,
                              unsigned indent) {

  Indent(o, indent) << "<dict>\n";
  ++indent;

  Indent(o, indent) << "<key>kind</key><string>control</string>\n";

  // Emit edges.
  Indent(o, indent) << "<key>edges</key>\n";
  ++indent;
  Indent(o, indent) << "<array>\n";
  ++indent;
  for (PathDiagnosticControlFlowPiece::const_iterator I=P.begin(), E=P.end();
       I!=E; ++I) {
    Indent(o, indent) << "<dict>\n";
    ++indent;
    Indent(o, indent) << "<key>start</key>\n";
    EmitRange(o, SM, LangOpts, I->getStart().asRange(), FM, indent+1);
    Indent(o, indent) << "<key>end</key>\n";
    EmitRange(o, SM, LangOpts, I->getEnd().asRange(), FM, indent+1);
    --indent;
    Indent(o, indent) << "</dict>\n";
  }
  --indent;
  Indent(o, indent) << "</array>\n";
  --indent;

  // Output any helper text.
  const std::string& s = P.getString();
  if (!s.empty()) {
    Indent(o, indent) << "<key>alternate</key>";
    EmitString(o, s) << '\n';
  }

  --indent;
  Indent(o, indent) << "</dict>\n";
}

static void ReportEvent(llvm::raw_ostream& o, const PathDiagnosticPiece& P,
                        const FIDMap& FM,
                        const SourceManager &SM,
                        const LangOptions &LangOpts,
                        unsigned indent) {

  Indent(o, indent) << "<dict>\n";
  ++indent;

  Indent(o, indent) << "<key>kind</key><string>event</string>\n";

  // Output the location.
  FullSourceLoc L = P.getLocation().asLocation();

  Indent(o, indent) << "<key>location</key>\n";
  EmitLocation(o, SM, LangOpts, L, FM, indent);

  // Output the ranges (if any).
  PathDiagnosticPiece::range_iterator RI = P.ranges_begin(),
  RE = P.ranges_end();

  if (RI != RE) {
    Indent(o, indent) << "<key>ranges</key>\n";
    Indent(o, indent) << "<array>\n";
    ++indent;
    for (; RI != RE; ++RI)
      EmitRange(o, SM, LangOpts, *RI, FM, indent+1);
    --indent;
    Indent(o, indent) << "</array>\n";
  }

  // Output the text.
  assert(!P.getString().empty());
  Indent(o, indent) << "<key>extended_message</key>\n";
  Indent(o, indent);
  EmitString(o, P.getString()) << '\n';

  // Output the short text.
  // FIXME: Really use a short string.
  Indent(o, indent) << "<key>message</key>\n";
  EmitString(o, P.getString()) << '\n';

  // Finish up.
  --indent;
  Indent(o, indent); o << "</dict>\n";
}

static void ReportMacro(llvm::raw_ostream& o,
                        const PathDiagnosticMacroPiece& P,
                        const FIDMap& FM, const SourceManager &SM,
                        const LangOptions &LangOpts,
                        unsigned indent) {

  for (PathDiagnosticMacroPiece::const_iterator I=P.begin(), E=P.end();
       I!=E; ++I) {

    switch ((*I)->getKind()) {
      default:
        break;
      case PathDiagnosticPiece::Event:
        ReportEvent(o, cast<PathDiagnosticEventPiece>(**I), FM, SM, LangOpts,
                    indent);
        break;
      case PathDiagnosticPiece::Macro:
        ReportMacro(o, cast<PathDiagnosticMacroPiece>(**I), FM, SM, LangOpts,
                    indent);
        break;
    }
  }
}

static void ReportDiag(llvm::raw_ostream& o, const PathDiagnosticPiece& P,
                       const FIDMap& FM, const SourceManager &SM,
                       const LangOptions &LangOpts) {

  unsigned indent = 4;

  switch (P.getKind()) {
    case PathDiagnosticPiece::ControlFlow:
      ReportControlFlow(o, cast<PathDiagnosticControlFlowPiece>(P), FM, SM,
                        LangOpts, indent);
      break;
    case PathDiagnosticPiece::Event:
      ReportEvent(o, cast<PathDiagnosticEventPiece>(P), FM, SM, LangOpts,
                  indent);
      break;
    case PathDiagnosticPiece::Macro:
      ReportMacro(o, cast<PathDiagnosticMacroPiece>(P), FM, SM, LangOpts,
                  indent);
      break;
  }
}

void PlistDiagnostics::HandlePathDiagnostic(const PathDiagnostic* D) {
  if (!D)
    return;

  if (D->empty()) {
    delete D;
    return;
  }

  // We need to flatten the locations (convert Stmt* to locations) because
  // the referenced statements may be freed by the time the diagnostics
  // are emitted.
  const_cast<PathDiagnostic*>(D)->flattenLocations();
  BatchedDiags.push_back(D);
}

void PlistDiagnostics::FlushDiagnostics(llvm::SmallVectorImpl<std::string>
                                        *FilesMade) {

  // Build up a set of FIDs that we use by scanning the locations and
  // ranges of the diagnostics.
  FIDMap FM;
  llvm::SmallVector<FileID, 10> Fids;
  const SourceManager* SM = 0;

  if (!BatchedDiags.empty())
    SM = &(*BatchedDiags.begin())->begin()->getLocation().getManager();

  for (std::vector<const PathDiagnostic*>::iterator DI = BatchedDiags.begin(),
       DE = BatchedDiags.end(); DI != DE; ++DI) {

    const PathDiagnostic *D = *DI;

    for (PathDiagnostic::const_iterator I=D->begin(), E=D->end(); I!=E; ++I) {
      AddFID(FM, Fids, SM, I->getLocation().asLocation());

      for (PathDiagnosticPiece::range_iterator RI=I->ranges_begin(),
           RE=I->ranges_end(); RI!=RE; ++RI) {
        AddFID(FM, Fids, SM, RI->getBegin());
        AddFID(FM, Fids, SM, RI->getEnd());
      }
    }
  }

  // Open the file.
  std::string ErrMsg;
  llvm::raw_fd_ostream o(OutputFile.c_str(), ErrMsg);
  if (!ErrMsg.empty()) {
    llvm::errs() << "warning: could not creat file: " << OutputFile << '\n';
    return;
  }

  // Write the plist header.
  o << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
  "<!DOCTYPE plist PUBLIC \"-//Apple Computer//DTD PLIST 1.0//EN\" "
  "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
  "<plist version=\"1.0\">\n";

  // Write the root object: a <dict> containing...
  //  - "files", an <array> mapping from FIDs to file names
  //  - "diagnostics", an <array> containing the path diagnostics
  o << "<dict>\n"
       " <key>files</key>\n"
       " <array>\n";

  for (llvm::SmallVectorImpl<FileID>::iterator I=Fids.begin(), E=Fids.end();
       I!=E; ++I) {
    o << "  ";
    EmitString(o, SM->getFileEntryForID(*I)->getName()) << '\n';
  }

  o << " </array>\n"
       " <key>diagnostics</key>\n"
       " <array>\n";

  for (std::vector<const PathDiagnostic*>::iterator DI=BatchedDiags.begin(),
       DE = BatchedDiags.end(); DI!=DE; ++DI) {

    o << "  <dict>\n"
         "   <key>path</key>\n";

    const PathDiagnostic *D = *DI;
    // Create an owning smart pointer for 'D' just so that we auto-free it
    // when we exit this method.
    llvm::OwningPtr<PathDiagnostic> OwnedD(const_cast<PathDiagnostic*>(D));

    o << "   <array>\n";

    for (PathDiagnostic::const_iterator I=D->begin(), E=D->end(); I != E; ++I)
      ReportDiag(o, *I, FM, *SM, LangOpts);

    o << "   </array>\n";

    // Output the bug type and bug category.
    o << "   <key>description</key>";
    EmitString(o, D->getDescription()) << '\n';
    o << "   <key>category</key>";
    EmitString(o, D->getCategory()) << '\n';
    o << "   <key>type</key>";
    EmitString(o, D->getBugType()) << '\n';

    // Output the location of the bug.
    o << "  <key>location</key>\n";
    EmitLocation(o, *SM, LangOpts, D->getLocation(), FM, 2);

    // Output the diagnostic to the sub-diagnostic client, if any.
    if (SubPD) {
      SubPD->HandlePathDiagnostic(OwnedD.take());
      llvm::SmallVector<std::string, 1> SubFilesMade;
      SubPD->FlushDiagnostics(SubFilesMade);

      if (!SubFilesMade.empty()) {
        o << "  <key>" << SubPD->getName() << "_files</key>\n";
        o << "  <array>\n";
        for (size_t i = 0, n = SubFilesMade.size(); i < n ; ++i)
          o << "   <string>" << SubFilesMade[i] << "</string>\n";
        o << "  </array>\n";
      }
    }

    // Close up the entry.
    o << "  </dict>\n";
  }

  o << " </array>\n";

  // Finish.
  o << "</dict>\n</plist>";
  
  if (FilesMade)
    FilesMade->push_back(OutputFile);
}
