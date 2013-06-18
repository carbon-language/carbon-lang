#include "Core/FileOverrides.h"
#include "clang/Basic/SourceManager.h"

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
