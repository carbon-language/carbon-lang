//===- lib/ReaderWriter/ReaderArchive.cpp - Archive Library Reader--------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "lld/ReaderWriter/ReaderArchive.h"

#include "lld/Core/ArchiveLibraryFile.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/Object/ObjectFile.h"

#include <unordered_map>

namespace lld {
/// \brief The FileArchive class represents an Archive Library file
class FileArchive : public ArchiveLibraryFile {
public:

  virtual ~FileArchive() { }

  /// \brief Check if any member of the archive contains an Atom with the
  /// specified name and return the File object for that member, or nullptr.
  virtual const File *find(StringRef name, bool dataSymbolOnly) const {
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

    std::vector<std::unique_ptr<File>> result;

    OwningPtr<MemoryBuffer> buff;
    if (ci->getMemoryBuffer(buff, true))
      return nullptr;
    std::unique_ptr<MemoryBuffer> mb(buff.take());
    if (_targetInfo.logInputFiles())
      llvm::outs() << buff->getBufferIdentifier() << "\n";
    if (_targetInfo.parseFile(mb, result))
      return nullptr;

    assert(result.size() == 1);

    result[0]->setOrdinalAndIncrement(_curChildOrd);

    // give up the pointer so that this object no longer manages it
    return result[0].release();
  }

  virtual void setOrdinalAndIncrement(uint64_t &ordinal) const {
    _ordinal = ordinal++;
    _curChildOrd = _ordinal;
    // Leave space in ordinal range for all children
    for (auto mf = _archive->begin_children(),
              me = _archive->end_children(); mf != me; ++mf) {
        ordinal++;
    }
  }

  virtual const atom_collection<DefinedAtom> &defined() const {
    return _definedAtoms;
  }

  virtual const atom_collection<UndefinedAtom> &undefined() const {
    return _undefinedAtoms;
  }

  virtual const atom_collection<SharedLibraryAtom> &sharedLibrary() const {
    return _sharedLibraryAtoms;
  }

  virtual const atom_collection<AbsoluteAtom> &absolute() const {
    return _absoluteAtoms;
  }

protected:
  error_code isDataSymbol(MemoryBuffer *mb, StringRef symbol) const {
    std::unique_ptr<llvm::object::ObjectFile>
                    obj(llvm::object::ObjectFile::createObjectFile(mb));
    error_code ec;
    llvm::object::SymbolRef::Type symtype;
    uint32_t symflags;
    llvm::object::symbol_iterator ibegin = obj->begin_symbols();
    llvm::object::symbol_iterator iend = obj->end_symbols();
    StringRef symbolname;

    for (llvm::object::symbol_iterator i = ibegin; i != iend; i.increment(ec)) {
      if (ec) return ec;

      // Get symbol name
      if ((ec = (i->getName(symbolname)))) return ec;

      if (symbolname != symbol)
          continue;

      // Get symbol flags
      if ((ec = (i->getFlags(symflags)))) return ec;

      if (symflags <= llvm::object::SymbolRef::SF_Undefined)
          continue;

      // Get Symbol Type
      if ((ec = (i->getType(symtype)))) return ec;

      if (symtype == llvm::object::SymbolRef::ST_Data) {
        return error_code::success();
      }
    }
    return llvm::object::object_error::parse_failed;
  }

private:
  std::unique_ptr<llvm::object::Archive>    _archive;
  atom_collection_vector<DefinedAtom>       _definedAtoms;
  atom_collection_vector<UndefinedAtom>     _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom>      _absoluteAtoms;
  mutable uint64_t _curChildOrd;

public:
  /// only subclasses of ArchiveLibraryFile can be instantiated
  FileArchive(const TargetInfo &ti,
              std::unique_ptr<llvm::MemoryBuffer> mb, error_code &ec)
      : ArchiveLibraryFile(ti, mb->getBufferIdentifier()) {
    std::unique_ptr<llvm::object::Archive> archive_obj(
        new llvm::object::Archive(mb.release(), ec));
    if (ec)
      return;
    _archive.swap(archive_obj);

    // Cache symbols.
    for (auto i = _archive->begin_symbols(), e = _archive->end_symbols();
              i != e; ++i) {
      StringRef name;
      llvm::object::Archive::child_iterator member;
      if ((ec = i->getName(name)))
        return;
      if ((ec = i->getMember(member)))
        return;
      _symbolMemberMap[name] = member;
    }
  }

  std::unordered_map<StringRef, 
                     llvm::object::Archive::child_iterator> _symbolMemberMap;
}; // class FileArchive

// Returns a vector of Files that are contained in the archive file
// pointed to by the MemoryBuffer
error_code ReaderArchive::parseFile(std::unique_ptr<llvm::MemoryBuffer> &mb,
                            std::vector<std::unique_ptr<File>> &result) const {
  error_code ec;

  if (_targetInfo.forceLoadAllArchives()) {
    _archive.reset(new llvm::object::Archive(mb.release(), ec));
    if (ec)
      return ec;

    for (auto mf = _archive->begin_children(),
              me = _archive->end_children(); mf != me; ++mf) {
      OwningPtr<MemoryBuffer> buff;
      if ((ec = mf->getMemoryBuffer(buff, true)))
        return ec;
      std::unique_ptr<MemoryBuffer> mbc(buff.take());
      if (_targetInfo.logInputFiles())
        llvm::outs() << buff->getBufferIdentifier() << "\n";
      if ((ec = _targetInfo.parseFile(mbc, result)))
        return ec;
    }
  } else {
    std::unique_ptr<File> f;
    f.reset(new FileArchive(_targetInfo, std::move(mb), ec));
    if (ec)
      return ec;

    result.push_back(std::move(f));
  }
  return llvm::error_code::success();
}
} // end namespace lld
