//===- CIndex.cpp - Clang-C Source Indexing Library -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Clang-C Source Indexing library.
//
//===----------------------------------------------------------------------===//

#include "CIndexer.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Program.h"

#include <cstdio>
#include <vector>
#include <sstream>

#ifdef LLVM_ON_WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace clang;

const llvm::sys::Path& CIndexer::getClangPath() {
  // Did we already compute the path?
  if (!ClangPath.empty())
    return ClangPath;

  // Find the location where this library lives (libCIndex.dylib).
#ifdef LLVM_ON_WIN32
  MEMORY_BASIC_INFORMATION mbi;
  char path[MAX_PATH];
  VirtualQuery((void *)(uintptr_t)clang_createTranslationUnit, &mbi,
               sizeof(mbi));
  GetModuleFileNameA((HINSTANCE)mbi.AllocationBase, path, MAX_PATH);

  llvm::sys::Path CIndexPath(path);

  CIndexPath.eraseComponent();
  CIndexPath.appendComponent("clang");
  CIndexPath.appendSuffix("exe");
  CIndexPath.makeAbsolute();
#else
  // This silly cast below avoids a C++ warning.
  Dl_info info;
  if (dladdr((void *)(uintptr_t)clang_createTranslationUnit, &info) == 0)
    assert(0 && "Call to dladdr() failed");

  llvm::sys::Path CIndexPath(info.dli_fname);

  // We now have the CIndex directory, locate clang relative to it.
  CIndexPath.eraseComponent();
  CIndexPath.appendComponent("..");
  CIndexPath.appendComponent("bin");
  CIndexPath.appendComponent("clang");
#endif

  // Cache our result.
  ClangPath = CIndexPath;
  return ClangPath;
}

std::string CIndexer::getClangResourcesPath() {
  llvm::sys::Path P = getClangPath();

  if (!P.empty()) {
    P.eraseComponent();  // Remove /clang from foo/bin/clang
    P.eraseComponent();  // Remove /bin   from foo/bin

    // Get foo/lib/clang/<version>/include
    P.appendComponent("lib");
    P.appendComponent("clang");
    P.appendComponent(CLANG_VERSION_STRING);
  }

  return P.str();
}

static llvm::sys::Path GetTemporaryPath() {
  // FIXME: This is lame; sys::Path should provide this function (in particular,
  // it should know how to find the temporary files dir).
  std::string Error;
  const char *TmpDir = ::getenv("TMPDIR");
  if (!TmpDir)
    TmpDir = ::getenv("TEMP");
  if (!TmpDir)
    TmpDir = ::getenv("TMP");
  if (!TmpDir)
    TmpDir = "/tmp";
  llvm::sys::Path P(TmpDir);
  P.appendComponent("remap");
  if (P.makeUnique(false, &Error))
    return llvm::sys::Path("");

  // FIXME: Grumble, makeUnique sometimes leaves the file around!?  PR3837.
  P.eraseFromDisk(false, 0);

  return P;
}

bool clang::RemapFiles(unsigned num_unsaved_files,
                       struct CXUnsavedFile *unsaved_files,
                       std::vector<std::string> &RemapArgs,
                       std::vector<llvm::sys::Path> &TemporaryFiles) {
  for (unsigned i = 0; i != num_unsaved_files; ++i) {
    // Write the contents of this unsaved file into the temporary file.
    llvm::sys::Path SavedFile(GetTemporaryPath());
    if (SavedFile.empty())
      return true;

    std::string ErrorInfo;
    llvm::raw_fd_ostream OS(SavedFile.c_str(), ErrorInfo);
    if (!ErrorInfo.empty())
      return true;
    
    OS.write(unsaved_files[i].Contents, unsaved_files[i].Length);
    OS.close();
    if (OS.has_error()) {
      SavedFile.eraseFromDisk();
      return true;
    }
    
    // Remap the file.
    std::string RemapArg = unsaved_files[i].Filename;
    RemapArg += ';';
    RemapArg += SavedFile.str();
    RemapArgs.push_back("-Xclang");
    RemapArgs.push_back("-remap-file");
    RemapArgs.push_back("-Xclang");
    RemapArgs.push_back(RemapArg);
    TemporaryFiles.push_back(SavedFile);
  }
  
  return false;
}

