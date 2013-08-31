//===- lib/Core/LinkingContext.cpp - Linker Context Object Interface ------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/LinkingContext.h"
#include "lld/Core/InputFiles.h"
#include "lld/ReaderWriter/Writer.h"
#include "lld/ReaderWriter/Simple.h"

#include "llvm/ADT/Triple.h"

namespace lld {

LinkingContext::LinkingContext()
    : Reader(*this), _deadStrip(false), _globalsAreDeadStripRoots(false),
      _searchArchivesToOverrideTentativeDefinitions(false),
      _searchSharedLibrariesToOverrideTentativeDefinitions(false),
      _warnIfCoalesableAtomsHaveDifferentCanBeNull(false),
      _warnIfCoalesableAtomsHaveDifferentLoadName(false),
      _forceLoadAllArchives(false), _printRemainingUndefines(true),
      _allowRemainingUndefines(false), _logInputFiles(false),
      _allowShlibUndefines(false) {}

LinkingContext::~LinkingContext() {}

bool LinkingContext::validate(raw_ostream &diagnostics) {
  _yamlReader = createReaderYAML(*this);
  return validateImpl(diagnostics);
}

error_code
LinkingContext::readFile(StringRef path,
                         std::vector<std::unique_ptr<File>> &result) const {
  OwningPtr<llvm::MemoryBuffer> opmb;
  if (error_code ec = llvm::MemoryBuffer::getFileOrSTDIN(path, opmb))
    return ec;

  std::unique_ptr<MemoryBuffer> mb(opmb.take());
  return this->parseFile(mb, result);
}

error_code LinkingContext::writeFile(const File &linkedFile) const {
  return this->writer().writeFile(linkedFile, _outputPath);
}

void LinkingContext::addImplicitFiles(InputFiles &inputs) const {
  this->writer().addFiles(inputs);
}

std::unique_ptr<File> LinkingContext::createEntrySymbolFile() {
  if (entrySymbolName().empty())
    return nullptr;
  std::unique_ptr<SimpleFile> entryFile(
      new SimpleFile(*this, "command line option -entry"));
  entryFile->addAtom(
      *(new (_allocator) SimpleUndefinedAtom(*entryFile, entrySymbolName())));
  return std::move(entryFile);
}

std::unique_ptr<File> LinkingContext::createUndefinedSymbolFile() {
  if (_initialUndefinedSymbols.empty())
    return nullptr;
  std::unique_ptr<SimpleFile> undefinedSymFile(
      new SimpleFile(*this, "command line option -u"));
  for (auto undefSymStr : _initialUndefinedSymbols)
    undefinedSymFile->addAtom(*(new (_allocator) SimpleUndefinedAtom(
                                   *undefinedSymFile, undefSymStr)));
  return std::move(undefinedSymFile);
}

std::vector<std::unique_ptr<File> > LinkingContext::createInternalFiles() {
  std::vector<std::unique_ptr<File> > result;
  std::unique_ptr<File> internalFile;
  internalFile = createEntrySymbolFile();
  if (internalFile)
    result.push_back(std::move(internalFile));
  internalFile = createUndefinedSymbolFile();
  if (internalFile)
    result.push_back(std::move(internalFile));
  return result;
}

void LinkingContext::addPasses(PassManager &pm) const {}

} // end namespace lld
