#include "Core/FileOverrides.h"
#include "clang/Basic/SourceManager.h"

void SourceOverrides::applyOverrides(clang::SourceManager &SM,
                                     clang::FileManager &FM) const {
  assert(!MainFileOverride.empty() &&
         "Main source file override should exist!");
  SM.overrideFileContents(FM.getFile(MainFileName),
                          llvm::MemoryBuffer::getMemBuffer(MainFileOverride));
}
