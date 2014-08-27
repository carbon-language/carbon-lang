//===--- VirtualFileHelper.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \brief This file defines an utility class for tests that needs a source
/// manager for a virtual file with customizable content.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_MODERNIZE_VIRTUAL_FILE_HELPER_H
#define CLANG_MODERNIZE_VIRTUAL_FILE_HELPER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"

namespace clang {

/// \brief Class that provides easy access to a SourceManager and that allows to
/// map virtual files conveniently.
class VirtualFileHelper {
  struct VirtualFile {
    std::string FileName;
    std::string Code;
  };

public:
  VirtualFileHelper()
      : DiagOpts(new DiagnosticOptions()),
        Diagnostics(IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
                    &*DiagOpts),
        DiagnosticPrinter(llvm::outs(), &*DiagOpts),
        Files((FileSystemOptions())) {}

  /// \brief Create a virtual file \p FileName, with content \p Code.
  void mapFile(llvm::StringRef FileName, llvm::StringRef Code) {
    VirtualFile VF = { FileName, Code };
    VirtualFiles.push_back(VF);
  }

  /// \brief Create a new \c SourceManager with the virtual files and contents
  /// mapped to it.
  SourceManager &getNewSourceManager() {
    Sources.reset(new SourceManager(Diagnostics, Files));
    mapVirtualFiles(*Sources);
    return *Sources;
  }

  /// \brief Map the virtual file contents in the given \c SourceManager.
  void mapVirtualFiles(SourceManager &SM) const {
    for (llvm::SmallVectorImpl<VirtualFile>::const_iterator
             I = VirtualFiles.begin(),
             E = VirtualFiles.end();
         I != E; ++I) {
      std::unique_ptr<llvm::MemoryBuffer> Buf =
          llvm::MemoryBuffer::getMemBuffer(I->Code);
      const FileEntry *Entry = SM.getFileManager().getVirtualFile(
          I->FileName, Buf->getBufferSize(), /*ModificationTime=*/0);
      SM.overrideFileContents(Entry, std::move(Buf));
    }
  }

private:
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  DiagnosticsEngine Diagnostics;
  TextDiagnosticPrinter DiagnosticPrinter;
  FileManager Files;
  // most tests don't need more than one file
  llvm::SmallVector<VirtualFile, 1> VirtualFiles;
  std::unique_ptr<SourceManager> Sources;
};

} // end namespace clang

#endif // CLANG_MODERNIZE_VIRTUAL_FILE_HELPER_H
