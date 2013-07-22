//===-- Core/FileOverrides.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides types and functionality for dealing with source
/// and header file content overrides.
///
//===----------------------------------------------------------------------===//

#include "Core/FileOverrides.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

using namespace clang;
using namespace clang::tooling;

SourceOverrides::SourceOverrides(llvm::StringRef MainFileName)
    : MainFileName(MainFileName) {}

void SourceOverrides::applyReplacements(tooling::Replacements &Replaces) {
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts(
      new DiagnosticOptions());
  DiagnosticsEngine Diagnostics(
      llvm::IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
      DiagOpts.getPtr());
  FileManager Files((FileSystemOptions()));
  SourceManager SM(Diagnostics, Files);
  applyReplacements(Replaces, SM);
}

void SourceOverrides::applyReplacements(tooling::Replacements &Replaces,
                                        SourceManager &SM) {
  applyOverrides(SM);

  Rewriter Rewrites(SM, LangOptions());

  // FIXME: applyAllReplacements will indicate if it couldn't apply all
  // replacements. Handle that case.
  bool Success = tooling::applyAllReplacements(Replaces, Rewrites);

  if (!Success)
    llvm::errs() << "error: failed to apply some replacements.";

  applyRewrites(Rewrites);
}

void SourceOverrides::applyRewrites(Rewriter &Rewrites) {
  std::string ResultBuf;

  for (Rewriter::buffer_iterator I = Rewrites.buffer_begin(),
                                 E = Rewrites.buffer_end();
       I != E; ++I) {
    const FileEntry *Entry =
        Rewrites.getSourceMgr().getFileEntryForID(I->first);
    assert(Entry != NULL && "unexpected null FileEntry");
    assert(Entry->getName() != NULL &&
           "unexpected null return from FileEntry::getName()");
    llvm::StringRef FileName = Entry->getName();

    // Get a copy of the rewritten buffer from the Rewriter.
    ResultBuf.clear();
    llvm::raw_string_ostream StringStream(ResultBuf);
    I->second.write(StringStream);
    StringStream.flush();

    if (MainFileName == FileName) {
      MainFileOverride.swap(ResultBuf);
      continue;
    }

    // Header overrides are treated differently. Eventually, raw replacements
    // will be stored as well for later output to disk. Applying replacements
    // in memory will always be necessary as the source goes down the transform
    // pipeline.

    HeaderOverride &HeaderOv = Headers[FileName];
    HeaderOv.FileOverride.swap(ResultBuf);
    // "Create" HeaderOverride if not already existing
    if (HeaderOv.FileName.empty())
      HeaderOv.FileName = FileName;
  }
}

void SourceOverrides::applyOverrides(SourceManager &SM) const {
  FileManager &FM = SM.getFileManager();

  if (isSourceOverriden())
    SM.overrideFileContents(FM.getFile(MainFileName),
                            llvm::MemoryBuffer::getMemBuffer(MainFileOverride));

  for (HeaderOverrides::const_iterator I = Headers.begin(), E = Headers.end();
       I != E; ++I) {
    assert(!I->second.FileOverride.empty() &&
           "Header override should not be empty!");
    SM.overrideFileContents(
        FM.getFile(I->second.FileName),
        llvm::MemoryBuffer::getMemBuffer(I->second.FileOverride));
  }
}

bool generateReplacementsFileName(llvm::StringRef SourceFile,
                                    llvm::StringRef HeaderFile,
                                    llvm::SmallVectorImpl<char> &Result,
                                    llvm::SmallVectorImpl<char> &Error) {
  using namespace llvm::sys;
  std::string UniqueHeaderNameModel;

  // Get the filename portion of the path.
  llvm::StringRef SourceFileRef(path::filename(SourceFile));
  llvm::StringRef HeaderFileRef(path::filename(HeaderFile));

  // Get the actual path for the header file.
  llvm::SmallString<128> HeaderPath(HeaderFile);
  path::remove_filename(HeaderPath);

  // Build the model of the filename.
  llvm::raw_string_ostream UniqueHeaderNameStream(UniqueHeaderNameModel);
  UniqueHeaderNameStream << SourceFileRef << "_" << HeaderFileRef
                         << "_%%_%%_%%_%%_%%_%%" << ".yaml";
  path::append(HeaderPath, UniqueHeaderNameStream.str());

  Error.clear();
  if (llvm::error_code EC =
          fs::createUniqueFile(HeaderPath.c_str(), Result)) {
    Error.append(EC.message().begin(), EC.message().end());
    return false;
  }

  return true;
}

FileOverrides::~FileOverrides() {
  for (SourceOverridesMap::iterator I = Overrides.begin(), E = Overrides.end();
       I != E; ++I)
    delete I->getValue();
}

SourceOverrides &FileOverrides::getOrCreate(llvm::StringRef Filename) {
  SourceOverrides *&Override = Overrides[Filename];

  if (Override == NULL)
    Override = new SourceOverrides(Filename);
  return *Override;
}
