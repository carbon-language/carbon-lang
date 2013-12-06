//===- lld/ReaderWriter/FileArchive.cpp - Archive Library File -----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "lld/ReaderWriter/FileArchive.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/MemoryBuffer.h"

#include <unordered_map>

namespace lld {

/// \brief Check if any member of the archive contains an Atom with the
/// specified name and return the File object for that member, or nullptr.
const File *FileArchive::find(StringRef name, bool dataSymbolOnly) const {
  auto member = _symbolMemberMap.find(name);
  if (member == _symbolMemberMap.end())
    return nullptr;

  llvm::object::Archive::child_iterator ci = member->second;

  if (dataSymbolOnly) {
    OwningPtr<MemoryBuffer> buff;
    if (ci->getMemoryBuffer(buff, true))
      return nullptr;
    if (isDataSymbol(buff.take(), name))
      return nullptr;
  }

  std::vector<std::unique_ptr<File> > result;

  OwningPtr<MemoryBuffer> buff;
  if (ci->getMemoryBuffer(buff, true))
    return nullptr;
  if (_context.logInputFiles())
    llvm::outs() << buff->getBufferIdentifier() << "\n";
  std::unique_ptr<MemoryBuffer> mb(buff.take());
  if (_context.getDefaultReader().parseFile(mb, result))
    return nullptr;

  assert(result.size() == 1);

  // give up the pointer so that this object no longer manages it
  return result[0].release();
}

/// \brief parse each member
error_code FileArchive::parseAllMembers(
    std::vector<std::unique_ptr<File> > &result) const {
  for (auto mf = _archive->begin_children(), me = _archive->end_children();
       mf != me; ++mf) {
    OwningPtr<MemoryBuffer> buff;
    if (error_code ec = mf->getMemoryBuffer(buff, true))
      return ec;
    if (_context.logInputFiles())
      llvm::outs() << buff->getBufferIdentifier() << "\n";
    std::unique_ptr<MemoryBuffer> mbc(buff.take());
    if (error_code ec = _context.getDefaultReader().parseFile(mbc, result))
      return ec;
  }
  return error_code::success();
}

const ArchiveLibraryFile::atom_collection<DefinedAtom> &
FileArchive::defined() const {
  return _definedAtoms;
}

const ArchiveLibraryFile::atom_collection<UndefinedAtom> &
FileArchive::undefined() const {
  return _undefinedAtoms;
}

const ArchiveLibraryFile::atom_collection<SharedLibraryAtom> &
FileArchive::sharedLibrary() const {
  return _sharedLibraryAtoms;
}

const ArchiveLibraryFile::atom_collection<AbsoluteAtom> &
FileArchive::absolute() const {
  return _absoluteAtoms;
}

error_code FileArchive::isDataSymbol(MemoryBuffer *mb, StringRef symbol) const {
  std::unique_ptr<llvm::object::ObjectFile> obj(
      llvm::object::ObjectFile::createObjectFile(mb));
  error_code ec;
  llvm::object::SymbolRef::Type symtype;
  uint32_t symflags;
  llvm::object::symbol_iterator ibegin = obj->begin_symbols();
  llvm::object::symbol_iterator iend = obj->end_symbols();
  StringRef symbolname;

  for (llvm::object::symbol_iterator i = ibegin; i != iend; i.increment(ec)) {
    if (ec)
      return ec;

    // Get symbol name
    if (error_code ec = i->getName(symbolname))
      return ec;

    if (symbolname != symbol)
      continue;

    // Get symbol flags
    if (error_code ec = i->getFlags(symflags))
      return ec;

    if (symflags <= llvm::object::SymbolRef::SF_Undefined)
      continue;

    // Get Symbol Type
    if (error_code ec = i->getType(symtype))
      return ec;

    if (symtype == llvm::object::SymbolRef::ST_Data) {
      return error_code::success();
    }
  }
  return llvm::object::object_error::parse_failed;
}

FileArchive::FileArchive(const LinkingContext &context,
                         std::unique_ptr<MemoryBuffer> mb, error_code &ec,
                         bool isWholeArchive)
    : ArchiveLibraryFile(context, mb->getBufferIdentifier()),
      _isWholeArchive(isWholeArchive) {
  std::unique_ptr<llvm::object::Archive> archive_obj(
      new llvm::object::Archive(mb.release(), ec));
  if (ec)
    return;
  _archive.swap(archive_obj);

  // Cache symbols.
  for (auto i = _archive->begin_symbols(), e = _archive->end_symbols(); i != e;
       ++i) {
    StringRef name;
    llvm::object::Archive::child_iterator member;
    if ((ec = i->getName(name)))
      return;
    if ((ec = i->getMember(member)))
      return;
    _symbolMemberMap[name] = member;
  }
}

} // end namespace lld
