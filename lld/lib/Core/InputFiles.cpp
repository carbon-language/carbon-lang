//===- Core/InputFiles.cpp - Manages list of Files ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/InputFiles.h"
#include "lld/Core/SharedLibraryFile.h"
#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/LLVM.h"

namespace lld {

InputFiles::InputFiles() {
}


InputFiles::~InputFiles() {
}

void InputFiles::forEachInitialAtom(InputFiles::Handler &handler) const {
  for ( const File *file : _files ) {
    this->handleFile(file, handler);
  }
}

void InputFiles::prependFile(const File &file) {
  _files.insert(_files.begin(), &file);
}

void InputFiles::appendFile(const File &file) {
  _files.push_back(&file);
}

void InputFiles::appendFiles(std::vector<std::unique_ptr<File>> &files) {
  for (std::unique_ptr<File> &f : files) {
    _files.push_back(f.release());
  }
}

void InputFiles::assignFileOrdinals() {
  uint64_t i = 0;
  for ( const File *file : _files ) {
    file->setOrdinalAndIncrement(i);
  }
}


bool InputFiles::searchLibraries(StringRef name, bool searchSharedLibs,
                               bool searchArchives, bool dataSymbolOnly,
                               InputFiles::Handler &handler) const {

  for ( const File *file : _files ) {
    if ( searchSharedLibs ) {
      if (const SharedLibraryFile *shlib = dyn_cast<SharedLibraryFile>(file)) {
        if ( const SharedLibraryAtom* shAtom = shlib->exports(name,
                                                            dataSymbolOnly) ) {
          handler.doSharedLibraryAtom(*shAtom);
          return true;
        }
      }
    }
    if ( searchArchives ) {
      if (const ArchiveLibraryFile *lib = dyn_cast<ArchiveLibraryFile>(file)) {
        if ( const File *member = lib->find(name, dataSymbolOnly) ) {
          this->handleFile(member, handler);
          return true;
        }
      }
    }
  }
  return false;
}


void InputFiles::handleFile(const File *file,
                            InputFiles::Handler &handler) const {
  handler.doFile(*file);
  for( const DefinedAtom *atom : file->defined() ) {
    handler.doDefinedAtom(*atom);
  }
  for( const UndefinedAtom *undefAtom : file->undefined() ) {
    handler.doUndefinedAtom(*undefAtom);
  }
  for( const SharedLibraryAtom *shlibAtom : file->sharedLibrary() ) {
    handler.doSharedLibraryAtom(*shlibAtom);
  }
  for( const AbsoluteAtom *absAtom : file->absolute() ) {
    handler.doAbsoluteAtom(*absAtom);
  }
}


} // namespace lld
