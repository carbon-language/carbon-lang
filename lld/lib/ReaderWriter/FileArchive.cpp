//===- lib/ReaderWriter/FileArchive.cpp -----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/LLVM.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"

#include <unordered_map>
#include <set>

using llvm::object::Archive;
using llvm::object::ObjectFile;
using llvm::object::SymbolRef;
using llvm::object::symbol_iterator;
using llvm::object::object_error;

namespace lld {

namespace {

/// \brief The FileArchive class represents an Archive Library file
class FileArchive : public lld::ArchiveLibraryFile {
public:
  virtual ~FileArchive() {}

  /// \brief Check if any member of the archive contains an Atom with the
  /// specified name and return the File object for that member, or nullptr.
  virtual const File *find(StringRef name, bool dataSymbolOnly) const {
    auto member = _symbolMemberMap.find(name);
    if (member == _symbolMemberMap.end())
      return nullptr;
    Archive::child_iterator ci = member->second;

    // Don't return a member already returned
    const char *memberStart = ci->getBuffer().data();
    if (_membersInstantiated.count(memberStart))
      return nullptr;

    if (dataSymbolOnly) {
      OwningPtr<MemoryBuffer> buff;
      if (ci->getMemoryBuffer(buff, true))
        return nullptr;
      if (isDataSymbol(buff.take(), name))
        return nullptr;
    }

    std::vector<std::unique_ptr<File>> result;
    if (instantiateMember(ci, result))
      return nullptr;
    assert(result.size() == 1);

    // give up the pointer so that this object no longer manages it
    return result[0].release();
  }

  /// \brief Load all members of the archive ?
  virtual bool isWholeArchive() const { return _isWholeArchive; }

  /// \brief parse each member
  virtual error_code
  parseAllMembers(std::vector<std::unique_ptr<File>> &result) const {
    for (auto mf = _archive->child_begin(), me = _archive->child_end();
         mf != me; ++mf) {
      if (error_code ec = instantiateMember(mf, result))
        return ec;
    }
    return error_code::success();
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
  error_code
  instantiateMember(Archive::child_iterator member,
                    std::vector<std::unique_ptr<File>> &result) const {
    OwningPtr<MemoryBuffer> buff;
    if (error_code ec = member->getMemoryBuffer(buff, true))
      return ec;
    if (_logLoading)
      llvm::outs() << buff->getBufferIdentifier() << "\n";
    std::unique_ptr<MemoryBuffer> mb(buff.take());
    _registry.parseFile(mb, result);
    const char *memberStart = member->getBuffer().data();
    _membersInstantiated.insert(memberStart);
    return error_code::success();
  }

  error_code isDataSymbol(MemoryBuffer *mb, StringRef symbol) const {
    auto objOrErr(ObjectFile::createObjectFile(mb));
    if (auto ec = objOrErr.getError())
      return ec;
    std::unique_ptr<ObjectFile> obj(objOrErr.get());
    SymbolRef::Type symtype;
    uint32_t symflags;
    symbol_iterator ibegin = obj->begin_symbols();
    symbol_iterator iend = obj->end_symbols();
    StringRef symbolname;

    for (symbol_iterator i = ibegin; i != iend; ++i) {
      error_code ec;

      // Get symbol name
      if ((ec = (i->getName(symbolname))))
        return ec;

      if (symbolname != symbol)
        continue;

      // Get symbol flags
      symflags = i->getFlags();

      if (symflags <= SymbolRef::SF_Undefined)
        continue;

      // Get Symbol Type
      if ((ec = (i->getType(symtype))))
        return ec;

      if (symtype == SymbolRef::ST_Data) {
        return error_code::success();
      }
    }
    return object_error::parse_failed;
  }

private:
  typedef std::unordered_map<StringRef, Archive::child_iterator> MemberMap;
  typedef std::set<const char *> InstantiatedSet;

  const Registry &_registry;
  std::unique_ptr<Archive> _archive;
  mutable MemberMap _symbolMemberMap;
  mutable InstantiatedSet _membersInstantiated;
  atom_collection_vector<DefinedAtom> _definedAtoms;
  atom_collection_vector<UndefinedAtom> _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom> _absoluteAtoms;
  bool _isWholeArchive;
  bool _logLoading;

public:
  /// only subclasses of ArchiveLibraryFile can be instantiated
  FileArchive(const Registry &registry, Archive *archive, StringRef path,
              bool isWholeArchive, bool logLoading)
      : ArchiveLibraryFile(path), _registry(registry),
        _archive(std::move(archive)), _isWholeArchive(isWholeArchive),
        _logLoading(logLoading) {}

  error_code buildTableOfContents() {
    DEBUG_WITH_TYPE("FileArchive", llvm::dbgs()
                                       << "Table of contents for archive '"
                                       << _archive->getFileName() << "':\n");
    for (auto i = _archive->symbol_begin(), e = _archive->symbol_end();
         i != e; ++i) {
      StringRef name;
      error_code ec;
      Archive::child_iterator member;
      if ((ec = i->getName(name)))
        return ec;
      if ((ec = i->getMember(member)))
        return ec;
      DEBUG_WITH_TYPE(
          "FileArchive",
          llvm::dbgs() << llvm::format("0x%08llX ", member->getBuffer().data())
                       << "'" << name << "'\n");
      _symbolMemberMap[name] = member;
    }
    return error_code::success();
  }

}; // class FileArchive

class ArchiveReader : public Reader {
public:
  ArchiveReader(bool logLoading) : _logLoading(logLoading) {}

  virtual bool canParse(file_magic magic, StringRef,
                        const MemoryBuffer &) const {
    return (magic == llvm::sys::fs::file_magic::archive);
  }

  virtual error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const Registry &reg,
            std::vector<std::unique_ptr<File>> &result) const {
    // Make Archive object which will be owned by FileArchive object.
    error_code ec;
    Archive *archive = new Archive(mb.get(), ec);
    if (ec)
      return ec;
    StringRef path = mb->getBufferIdentifier();
    // Construct FileArchive object.
    std::unique_ptr<FileArchive> file(
        new FileArchive(reg, archive, path, false, _logLoading));
    ec = file->buildTableOfContents();
    if (ec)
      return ec;

    // Transfer ownership of memory buffer to Archive object.
    mb.release();

    result.push_back(std::move(file));
    return error_code::success();
  }

private:
  bool _logLoading;
};

} // anonymous namespace

void Registry::addSupportArchives(bool logLoading) {
  add(std::unique_ptr<Reader>(new ArchiveReader(logLoading)));
}

} // end namespace lld


