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

#include "clang/Driver/PathDiagnosticClients.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;
typedef llvm::DenseMap<unsigned,unsigned> FIDMap;

namespace {
  class VISIBILITY_HIDDEN PlistDiagnostics : public PathDiagnosticClient {
    llvm::sys::Path Directory, FilePrefix;
    bool createdDir, noDir;
  public:
    PlistDiagnostics(const std::string& prefix);
    ~PlistDiagnostics() {}
    void HandlePathDiagnostic(const PathDiagnostic* D);  
  };  
} // end anonymous namespace

PlistDiagnostics::PlistDiagnostics(const std::string& prefix)
  : Directory(prefix), FilePrefix(prefix), createdDir(false), noDir(false) {
  FilePrefix.appendComponent("report"); // All Plist files begin with "report" 
}

PathDiagnosticClient* clang::CreatePlistDiagnosticClient(const std::string& s) {
  return new PlistDiagnostics(s);
}

static void AddFID(FIDMap& FIDs,
                   llvm::SmallVectorImpl<unsigned>& V,
                   SourceManager& SM, SourceLocation L) {

  unsigned fid = SM.getCanonicalFileID(SM.getLogicalLoc(L));
  FIDMap::iterator I = FIDs.find(fid);
  if (I != FIDs.end()) return;
  FIDs[fid] = V.size();
  V.push_back(fid);
}

static unsigned GetFID(const FIDMap& FIDs,
                       SourceManager& SM, SourceLocation L) {

  unsigned fid = SM.getCanonicalFileID(SM.getLogicalLoc(L));
  FIDMap::const_iterator I = FIDs.find(fid);
  assert (I != FIDs.end());
  return I->second;
}

static llvm::raw_ostream& Indent(llvm::raw_ostream& o, const unsigned indent) {
  for (unsigned i = 0; i < indent; ++i) o << ' ';
  return o;
}

static void EmitLocation(llvm::raw_ostream& o, SourceManager& SM,
                         SourceLocation L, const FIDMap& FM,
                         const unsigned indent) {

  Indent(o, indent) << "<dict>\n";
  Indent(o, indent) << " <key>line</key><integer>"
                    << SM.getLogicalLineNumber(L) << "</integer>\n";
  Indent(o, indent) << " <key>col</key><integer>"
                    << SM.getLogicalColumnNumber(L) << "</integer>\n";
  Indent(o, indent) << " <key>file</key><integer>"
                    << GetFID(FM, SM, L) << "</integer>\n";
  Indent(o, indent) << "</dict>\n";
}

static void EmitRange(llvm::raw_ostream& o, SourceManager& SM, SourceRange R,
                      const FIDMap& FM, const unsigned indent) {
 
  Indent(o, indent) << "<array>\n";
  EmitLocation(o, SM, R.getBegin(), FM, indent+1);
  EmitLocation(o, SM, R.getEnd(), FM, indent+1);
  Indent(o, indent) << "</array>\n";
}

static void ReportDiag(llvm::raw_ostream& o, const PathDiagnosticPiece& P, 
                       const FIDMap& FM, SourceManager& SM) {
  
  unsigned indent = 2;
  Indent(o, indent) << "<dict>\n";
  ++indent;
  
  // Output the location.
  FullSourceLoc L = P.getLocation();

  Indent(o, indent) << "<key>location</key>\n";
  EmitLocation(o, SM, L.getLocation(), FM, indent);

  // Output the ranges (if any).
  PathDiagnosticPiece::range_iterator RI = P.ranges_begin(),
                                      RE = P.ranges_end();
  
  if (RI != RE) {
    Indent(o, indent) << "<key>ranges</key>\n";
    Indent(o, indent) << "<array>\n";
    for ( ; RI != RE; ++RI ) EmitRange(o, SM, *RI, FM, indent+1);
    Indent(o, indent) << "</array>\n";
  }
  
  // Output the text.
  Indent(o, indent) << "<key>message</key>\n";
  Indent(o, indent) << "<string>" << P.getString() << "</string>";
  
  // Output the hint.
  Indent(o, indent) << "<key>displayhint</key>\n";
  Indent(o, indent) << "<string>"
                    << (P.getDisplayHint() == PathDiagnosticPiece::Above 
                        ? "above" : "below")
                    << "</string>\n";
  
  
  // Finish up.
  --indent;
  Indent(o, indent); o << "</dict>\n";
}

void PlistDiagnostics::HandlePathDiagnostic(const PathDiagnostic* D) {

  // Create an owning smart pointer for 'D' just so that we auto-free it
  // when we exit this method.
  llvm::OwningPtr<PathDiagnostic> OwnedD(const_cast<PathDiagnostic*>(D));
  
  // Create the directory to contain the plist files if it is missing.
  if (!createdDir) {
    createdDir = true;
    std::string ErrorMsg;
    Directory.createDirectoryOnDisk(true, &ErrorMsg);
  
    if (!Directory.isDirectory()) {
      llvm::errs() << "warning: could not create directory '"
                  << Directory.toString() << "'\n"
                  << "reason: " << ErrorMsg << '\n'; 
      
      noDir = true;
      
      return;
    }
  }
  
  if (noDir)
    return;

  // Get the source manager.
  SourceManager& SM = D->begin()->getLocation().getManager();

  // Build up a set of FIDs that we use by scanning the locations and
  // ranges of the diagnostics.
  FIDMap FM;
  llvm::SmallVector<unsigned, 10> Fids;
  
  for (PathDiagnostic::const_iterator I=D->begin(), E=D->end(); I != E; ++I) {
    AddFID(FM, Fids, SM, I->getLocation().getLocation());

    for (PathDiagnosticPiece::range_iterator RI=I->ranges_begin(),
                                             RE=I->ranges_end(); RI!=RE; ++RI) {      
      AddFID(FM, Fids, SM, RI->getBegin());
      AddFID(FM, Fids, SM, RI->getEnd());
    }
  }

  // Create a path for the target Plist file.
  llvm::sys::Path F(FilePrefix);
  F.makeUnique(false, NULL);
  
  // Rename the file with an Plist extension.
  llvm::sys::Path H(F);
  H.appendSuffix("plist");
  F.renamePathOnDisk(H, NULL);
  
  // Now create the plist file.
  std::string ErrMsg;
  llvm::raw_fd_ostream o(H.toString().c_str(), ErrMsg);
  
  if (!ErrMsg.empty()) {
    llvm::errs() << "warning: could not creat file: " << H.toString() << '\n';
    return;
  }
  
  // Write the plist header.
  o << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
       "<!DOCTYPE plist PUBLIC \"-//Apple Computer//DTD PLIST 1.0//EN\" "
       "http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
       "<plist version=\"1.0\">\n";
  
  // Write the root object: a <dict> containing...
  //  - "files", an <array> mapping from FIDs to file names
  //  - "diagnostics", an <array> containing the path diagnostics  
  o << "<dict>\n"
       " <key>files</key>\n"
       " <array>\n";
  
  for (llvm::SmallVectorImpl<unsigned>::iterator I=Fids.begin(), E=Fids.end();
       I!=E; ++I)
    o << "  <string>" << SM.getFileEntryForID(*I)->getName() << "</string>\n";    
  
  o << " </array>\n"
       " <key>diagnostics</key>\n"
       " <array>\n";
  
  for (PathDiagnostic::const_iterator I=D->begin(), E=D->end(); I != E; ++I)
    ReportDiag(o, *I, FM, SM);

  o << " </array>\n";
    
  // Output the bug type and bug category.  
  o << " <key>description</key><string>" << D->getDescription() << "</string>\n"
       " <key>category</key><string>" << D->getCategory() << "</string>\n";

  // Finish.
  o << "</dict>\n";
}
