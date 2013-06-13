#include "Core/FileOverrides.h"
#include "clang/Basic/SourceManager.h"

void SourceOverrides::applyOverrides(clang::SourceManager &SM,
                                     clang::FileManager &FM) const {
  assert(!MainFileOverride.empty() &&
         "Main source file override should exist!");
  SM.overrideFileContents(FM.getFile(MainFileName),
                          llvm::MemoryBuffer::getMemBuffer(MainFileOverride));

  for (HeaderOverrides::const_iterator I = Headers.begin(),
       E = Headers.end(); I != E; ++I)
    SM.overrideFileContents(
        FM.getFile(I->second.FileName),
        llvm::MemoryBuffer::getMemBuffer(I->second.FileOverride));
}
