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
      _outputFileType(OutputFileType::Default), _currentInputElement(nullptr) {}

LinkingContext::~LinkingContext() {}

bool LinkingContext::validate(raw_ostream &diagnostics) {
  _yamlReader = createReaderYAML(*this);
  _nativeReader = createReaderNative(*this);
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
  if (entrySymbolName().empty())
    return nullptr;
  std::unique_ptr<SimpleFile> entryFile(
      new SimpleFile(*this, "command line option -entry"));
  entryFile->addAtom(
      *(new (_allocator) SimpleUndefinedAtom(*entryFile, entrySymbolName())));
  return std::move(entryFile);
}

std::unique_ptr<File> LinkingContext::createUndefinedSymbolFile() const {
  if (_initialUndefinedSymbols.empty())
    return nullptr;
  std::unique_ptr<SimpleFile> undefinedSymFile(
      new SimpleFile(*this, "command line option -u"));
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

void LinkingContext::setResolverState(int32_t state) const {
  _currentInputElement->setResolverState(state);
}

ErrorOr<File &> LinkingContext::nextFile() const {
  if (_currentInputElement == nullptr) {
    ErrorOr<InputElement *> elem = inputGraph().getNextInputElement();
    if (error_code(elem) == input_graph_error::no_more_elements)
      return make_error_code(input_graph_error::no_more_files);
    _currentInputElement = *elem;
  }
  do {
    ErrorOr<File &> nextFile = _currentInputElement->getNextFile();
    if (error_code(nextFile) == input_graph_error::no_more_files) {
      ErrorOr<InputElement *> elem = inputGraph().getNextInputElement();
      if (error_code(elem) == input_graph_error::no_more_elements)
        return make_error_code(input_graph_error::no_more_files);
      _currentInputElement = *elem;
    } else {
      return std::move(nextFile);
    }
  } while (_currentInputElement != nullptr);
  return make_error_code(input_graph_error::no_more_files);
}

void LinkingContext::addPasses(PassManager &pm) const {}

} // end namespace lld
