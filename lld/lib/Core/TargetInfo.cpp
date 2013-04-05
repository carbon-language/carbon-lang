//===- lib/Core/TargetInfo.cpp - Linker Target Info Interface -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/TargetInfo.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/ADT/Triple.h"

namespace lld {

TargetInfo::TargetInfo()
  : Reader(*this)
  , _deadStrip(false)
  , _globalsAreDeadStripRoots(false)
  , _searchArchivesToOverrideTentativeDefinitions(false)
  , _searchSharedLibrariesToOverrideTentativeDefinitions(false)
  , _warnIfCoalesableAtomsHaveDifferentCanBeNull(false)
  , _warnIfCoalesableAtomsHaveDifferentLoadName(false)
  , _forceLoadAllArchives(false)
  , _printRemainingUndefines(true)
  , _allowRemainingUndefines(false)
  , _logInputFiles(false)
{
}

TargetInfo::~TargetInfo() {}


error_code TargetInfo::readFile(StringRef path,
                        std::vector<std::unique_ptr<File>> &result) const {
  OwningPtr<llvm::MemoryBuffer> opmb;
  if (error_code ec = llvm::MemoryBuffer::getFileOrSTDIN(path, opmb))
    return ec;

  std::unique_ptr<MemoryBuffer> mb(opmb.take());
  return this->parseFile(mb, result);
}

error_code TargetInfo::writeFile(const File &linkedFile) const {
   return this->writer().writeFile(linkedFile, _outputPath);
}

void TargetInfo::addImplicitFiles(InputFiles& inputs) const {
   this->writer().addFiles(inputs);
}

void TargetInfo::addPasses(PassManager &pm) const {
}


} // end namespace lld
