//===- lld/ReaderWriter/FileArchive.h - Archive Library File -----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_FILE_ARCHIVE_H
#define LLD_READER_WRITER_FILE_ARCHIVE_H

#include "lld/Core/ArchiveLibraryFile.h"

#include "llvm/Object/Archive.h"
#include "llvm/Support/MemoryBuffer.h"

#include <unordered_map>

namespace lld {

/// \brief The FileArchive class represents an Archive Library file
class FileArchive : public ArchiveLibraryFile {
public:
  FileArchive(const LinkingContext &context, std::unique_ptr<MemoryBuffer> mb,
              error_code &ec, bool isWholeArchive);
  virtual ~FileArchive() {}

  virtual const File *find(StringRef name, bool dataSymbolOnly) const;

  /// \brief Load all members of the archive?
  virtual bool isWholeArchive() const { return _isWholeArchive; }

  /// \brief parse each member
  virtual error_code
  parseAllMembers(std::vector<std::unique_ptr<File> > &result) const;

  virtual const atom_collection<DefinedAtom> &defined() const;
  virtual const atom_collection<UndefinedAtom> &undefined() const;
  virtual const atom_collection<SharedLibraryAtom> &sharedLibrary() const;
  virtual const atom_collection<AbsoluteAtom> &absolute() const;

protected:
  error_code isDataSymbol(MemoryBuffer *mb, StringRef symbol) const;

private:
  std::unique_ptr<llvm::object::Archive>    _archive;
  atom_collection_vector<DefinedAtom>       _definedAtoms;
  atom_collection_vector<UndefinedAtom>     _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom>      _absoluteAtoms;
  const bool _isWholeArchive;
  std::unordered_map<StringRef, llvm::object::Archive::child_iterator>
  _symbolMemberMap;
};

} // end namespace lld

#endif // LLD_READER_WRITER_FILE_ARCHIVE_H
