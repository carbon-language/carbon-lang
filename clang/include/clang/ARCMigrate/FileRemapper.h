//===-- FileRemapper.h - File Remapping Helper ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ARCMIGRATE_FILEREMAPPER_H
#define LLVM_CLANG_ARCMIGRATE_FILEREMAPPER_H

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
  class MemoryBuffer;
}

namespace clang {
  class FileManager;
  class FileEntry;
  class Diagnostic;
  class CompilerInvocation;

namespace arcmt {

class FileRemapper {
  // FIXME: Reuse the same FileManager for multiple ASTContexts.
  llvm::OwningPtr<FileManager> FileMgr;

  typedef llvm::PointerUnion<const FileEntry *, llvm::MemoryBuffer *> Target;
  typedef llvm::DenseMap<const FileEntry *, Target> MappingsTy;
  MappingsTy FromToMappings;

  llvm::DenseMap<const FileEntry *, const FileEntry *> ToFromMappings;

public:
  FileRemapper();
  ~FileRemapper();
  
  bool initFromDisk(llvm::StringRef outputDir, Diagnostic &Diag,
                    bool ignoreIfFilesChanged);
  bool flushToDisk(llvm::StringRef outputDir, Diagnostic &Diag);

  bool overwriteOriginal(Diagnostic &Diag,
                         llvm::StringRef outputDir = llvm::StringRef());

  void remap(llvm::StringRef filePath, llvm::MemoryBuffer *memBuf);
  void remap(llvm::StringRef filePath, llvm::StringRef newPath);

  void applyMappings(CompilerInvocation &CI) const;

  void transferMappingsAndClear(CompilerInvocation &CI);

  void clear(llvm::StringRef outputDir = llvm::StringRef());

private:
  void remap(const FileEntry *file, llvm::MemoryBuffer *memBuf);
  void remap(const FileEntry *file, const FileEntry *newfile);

  const FileEntry *getOriginalFile(llvm::StringRef filePath);
  void resetTarget(Target &targ);

  bool report(const std::string &err, Diagnostic &Diag);

  std::string getRemapInfoFile(llvm::StringRef outputDir);
};

} // end namespace arcmt

}  // end namespace clang

#endif
