//===- ARCMigrate.cpp - Clang-C ARC Migration Library ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the main API hooks in the Clang-C ARC Migration library.
//
//===----------------------------------------------------------------------===//

#include "clang-c/ARCMigrate.h"

#include "CXString.h"
#include "clang/ARCMigrate/ARCMT.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "llvm/Support/FileSystem.h"

using namespace clang;
using namespace arcmt;

namespace {

struct Remap {
  std::vector<std::pair<std::string, std::string> > Vec;
};

} // anonymous namespace.

//===----------------------------------------------------------------------===//
// libClang public APIs.
//===----------------------------------------------------------------------===//

extern "C" {

CMTRemap arcmt_getRemappings(const char *migrate_dir_path) {
  bool Logging = ::getenv("LIBCLANG_LOGGING");

  if (!migrate_dir_path) {
    if (Logging)
      llvm::errs() << "arcmt_getRemappings was called with NULL parameter\n";
    return 0;
  }

  bool exists = false;
  llvm::sys::fs::exists(migrate_dir_path, exists);
  if (!exists) {
    if (Logging) {
      llvm::errs() << "Error by arcmt_getRemappings(\"" << migrate_dir_path
                   << "\")\n";
      llvm::errs() << "\"" << migrate_dir_path << "\" does not exist\n";
    }
    return 0;
  }

  TextDiagnosticBuffer diagBuffer;
  llvm::OwningPtr<Remap> remap(new Remap());

  bool err = arcmt::getFileRemappings(remap->Vec, migrate_dir_path,&diagBuffer);

  if (err) {
    if (Logging) {
      llvm::errs() << "Error by arcmt_getRemappings(\"" << migrate_dir_path
                   << "\")\n";
      for (TextDiagnosticBuffer::const_iterator
             I = diagBuffer.err_begin(), E = diagBuffer.err_end(); I != E; ++I)
        llvm::errs() << I->second << '\n';
    }
    return 0;
  }

  return remap.take();
}

unsigned arcmt_remap_getNumFiles(CMTRemap map) {
  return static_cast<Remap *>(map)->Vec.size();
  
}

CXString arcmt_remap_getOriginalFile(CMTRemap map, unsigned index) {
  return cxstring::createCXString(static_cast<Remap *>(map)->Vec[index].first,
                                  /*DupString =*/ true);
}

CXString arcmt_remap_getTransformedFile(CMTRemap map, unsigned index) {
  return cxstring::createCXString(static_cast<Remap *>(map)->Vec[index].second,
                                  /*DupString =*/ true);
}

void arcmt_remap_dispose(CMTRemap map) {
  delete static_cast<Remap *>(map);
}

} // end: extern "C"
