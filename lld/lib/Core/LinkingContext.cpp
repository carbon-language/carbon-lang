//===- lib/Core/LinkingContext.cpp - Linker Context Object Interface ------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/LinkingContext.h"
#include "lld/Core/Resolver.h"
#include "lld/ReaderWriter/Writer.h"
#include "lld/ReaderWriter/Simple.h"

#include "llvm/ADT/Triple.h"

namespace lld {

LinkingContext::LinkingContext()
    : _deadStrip(false), _globalsAreDeadStripRoots(false),
      _searchArchivesToOverrideTentativeDefinitions(false),
      _searchSharedLibrariesToOverrideTentativeDefinitions(false),
      _warnIfCoalesableAtomsHaveDifferentCanBeNull(false),
      _warnIfCoalesableAtomsHaveDifferentLoadName(false),
      _printRemainingUndefines(true), _allowRemainingUndefines(false),
      _logInputFiles(false), _allowShlibUndefines(false),
      _outputFileType(OutputFileType::Default), _currentInputElement(nullptr),
      _nextOrdinal(0) {}

LinkingContext::~LinkingContext() {}

bool LinkingContext::validate(raw_ostream &diagnostics) {
  return validateImpl(diagnostics);
}

error_code LinkingContext::writeFile(const File &linkedFile) const {
  return this->writer().writeFile(linkedFile, _outputPath);
}

bool LinkingContext::createImplicitFiles(
    std::vector<std::unique_ptr<File> > &result) const {
  return this->writer().createImplicitFiles(result);
}

std::unique_ptr<File> LinkingContext::createEntrySymbolFile() const {
  return createEntrySymbolFile("command line option -u");
}

std::unique_ptr<File>
LinkingContext::createEntrySymbolFile(StringRef filename) const {
  if (entrySymbolName().empty())
    return nullptr;
  std::unique_ptr<SimpleFile> entryFile(new SimpleFile(filename));
  entryFile->addAtom(
      *(new (_allocator) SimpleUndefinedAtom(*entryFile, entrySymbolName())));
  return std::move(entryFile);
}

std::unique_ptr<File> LinkingContext::createUndefinedSymbolFile() const {
  return createUndefinedSymbolFile("command line option -u");
}

std::unique_ptr<File>
LinkingContext::createUndefinedSymbolFile(StringRef filename) const {
  if (_initialUndefinedSymbols.empty())
    return nullptr;
  std::unique_ptr<SimpleFile> undefinedSymFile(new SimpleFile(filename));
  for (auto undefSymStr : _initialUndefinedSymbols)
    undefinedSymFile->addAtom(*(new (_allocator) SimpleUndefinedAtom(
                                   *undefinedSymFile, undefSymStr)));
  return std::move(undefinedSymFile);
}

bool LinkingContext::createInternalFiles(
    std::vector<std::unique_ptr<File> > &result) const {
  std::unique_ptr<File> internalFile;
  internalFile = createEntrySymbolFile();
  if (internalFile)
    result.push_back(std::move(internalFile));
  internalFile = createUndefinedSymbolFile();
  if (internalFile)
    result.push_back(std::move(internalFile));
  return true;
}

void LinkingContext::setResolverState(uint32_t state) {
  _currentInputElement->setResolveState(state);
}

ErrorOr<File &> LinkingContext::nextFile() {
  // When nextFile() is called for the first time, _currentInputElement is not
  // initialized. Initialize it with the first element of the input graph.
  if (_currentInputElement == nullptr) {
    ErrorOr<InputElement *> elem = inputGraph().getNextInputElement();
    if (error_code(elem) == InputGraphError::no_more_elements)
      return make_error_code(InputGraphError::no_more_files);
    _currentInputElement = *elem;
  }

  // Otherwise, try to get the next file of _currentInputElement. If the current
  // input element points to an archive file, and there's a file left in the
  // archive, it will succeed. If not, try to get the next file in the input
  // graph.
  for (;;) {
    ErrorOr<File &> nextFile = _currentInputElement->getNextFile();
    if (error_code(nextFile) != InputGraphError::no_more_files)
      return std::move(nextFile);

    ErrorOr<InputElement *> elem = inputGraph().getNextInputElement();
    if (error_code(elem) == InputGraphError::no_more_elements ||
        *elem == nullptr)
      return make_error_code(InputGraphError::no_more_files);
    _currentInputElement = *elem;
  }
}

void LinkingContext::addPasses(PassManager &pm) {}

} // end namespace lld
