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

#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

void SourceOverrides::applyOverrides(clang::SourceManager &SM) const {
  clang::FileManager &FM = SM.getFileManager();

  if (isSourceOverriden())
    SM.overrideFileContents(FM.getFile(MainFileName),
                            llvm::MemoryBuffer::getMemBuffer(MainFileOverride));

  for (HeaderOverrides::const_iterator I = Headers.begin(),
       E = Headers.end(); I != E; ++I) {
    assert(!I->second.FileOverride.empty() && "Header override should not be empty!");
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

