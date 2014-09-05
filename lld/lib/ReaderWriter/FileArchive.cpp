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

#include <memory>
#include <set>
#include <unordered_map>

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
  FileArchive(const Registry &registry, Archive *archive, StringRef path,
              bool isWholeArchive, bool logLoading)
      : ArchiveLibraryFile(path), _registry(registry),
        _archive(std::move(archive)), _isWholeArchive(isWholeArchive),
        _logLoading(logLoading) {}

  virtual ~FileArchive() {}

  /// \brief Check if any member of the archive contains an Atom with the
  /// specified name and return the File object for that member, or nullptr.
  const File *find(StringRef name, bool dataSymbolOnly) const override {
    auto member = _symbolMemberMap.find(name);
    if (member == _symbolMemberMap.end())
      return nullptr;
    Archive::child_iterator ci = member->second;

    // Don't return a member already returned
    const char *memberStart = ci->getBuffer().data();
    if (_membersInstantiated.count(memberStart))
      return nullptr;
    if (dataSymbolOnly && !isDataSymbol(ci, name))
      return nullptr;

    std::vector<std::unique_ptr<File>> result;
    if (instantiateMember(ci, result))
      return nullptr;
    assert(result.size() == 1);

    // give up the pointer so that this object no longer manages it
    return result[0].release();
  }

  /// \brief Load all members of the archive?
  virtual bool isWholeArchive() const { return _isWholeArchive; }

  /// \brief parse each member
  std::error_code
  parseAllMembers(std::vector<std::unique_ptr<File>> &result) const override {
    for (auto mf = _archive->child_begin(), me = _archive->child_end();
         mf != me; ++mf) {
      if (std::error_code ec = instantiateMember(mf, result))
        return ec;
    }
    return std::error_code();
  }

  const atom_collection<DefinedAtom> &defined() const override {
    return _definedAtoms;
  }

  const atom_collection<UndefinedAtom> &undefined() const override {
    return _undefinedAtoms;
  }

  const atom_collection<SharedLibraryAtom> &sharedLibrary() const override {
    return _sharedLibraryAtoms;
  }

  const atom_collection<AbsoluteAtom> &absolute() const override {
    return _absoluteAtoms;
  }

  std::error_code buildTableOfContents() {
    DEBUG_WITH_TYPE("FileArchive", llvm::dbgs()
                                       << "Table of contents for archive '"
                                       << _archive->getFileName() << "':\n");
    for (auto i = _archive->symbol_begin(), e = _archive->symbol_end();
         i != e; ++i) {
      StringRef name = i->getName();
      ErrorOr<Archive::child_iterator> memberOrErr = i->getMember();
      if (std::error_code ec = memberOrErr.getError())
        return ec;
      Archive::child_iterator member = memberOrErr.get();
      DEBUG_WITH_TYPE(
          "FileArchive",
          llvm::dbgs() << llvm::format("0x%08llX ", member->getBuffer().data())
                       << "'" << name << "'\n");
      _symbolMemberMap[name] = member;
    }
    return std::error_code();
  }

  /// Returns a set of all defined symbols in the archive.
  std::set<StringRef> getDefinedSymbols() const override {
    std::set<StringRef> ret;
    for (const auto &e : _symbolMemberMap)
      ret.insert(e.first);
    return ret;
  }

protected:
  std::error_code
  instantiateMember(Archive::child_iterator member,
                    std::vector<std::unique_ptr<File>> &result) const {
    ErrorOr<llvm::MemoryBufferRef> mbOrErr = member->getMemoryBufferRef();
    if (std::error_code ec = mbOrErr.getError())
      return ec;
    llvm::MemoryBufferRef mb = mbOrErr.get();
    if (_logLoading)
      llvm::outs() << mb.getBufferIdentifier() << "\n";

    std::unique_ptr<MemoryBuffer> buf(MemoryBuffer::getMemBuffer(
        mb.getBuffer(), mb.getBufferIdentifier(), false));

    _registry.parseFile(buf, result);
    const char *memberStart = member->getBuffer().data();
    _membersInstantiated.insert(memberStart);
    return std::error_code();
  }

  // Parses the given memory buffer as an object file, and returns true
  // code if the given symbol is a data symbol. If the symbol is not a data
  // symbol or does not exist, returns false.
  bool isDataSymbol(Archive::child_iterator member, StringRef symbol) const {
    ErrorOr<llvm::MemoryBufferRef> buf = member->getMemoryBufferRef();
    if (buf.getError())
      return false;
    std::unique_ptr<MemoryBuffer> mb(MemoryBuffer::getMemBuffer(
        buf.get().getBuffer(), buf.get().getBufferIdentifier(), false));

    auto objOrErr(ObjectFile::createObjectFile(mb->getMemBufferRef()));
    if (objOrErr.getError())
      return false;
    std::unique_ptr<ObjectFile> obj = std::move(objOrErr.get());
    SymbolRef::Type symtype;
    uint32_t symflags;
    symbol_iterator ibegin = obj->symbol_begin();
    symbol_iterator iend = obj->symbol_end();
    StringRef symbolname;

    for (symbol_iterator i = ibegin; i != iend; ++i) {
      // Get symbol name
      if (i->getName(symbolname))
        return false;
      if (symbolname != symbol)
        continue;

      // Get symbol flags
      symflags = i->getFlags();

      if (symflags <= SymbolRef::SF_Undefined)
        continue;

      // Get Symbol Type
      if (i->getType(symtype))
        return false;

      if (symtype == SymbolRef::ST_Data)
        return true;
    }
    return false;
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
};

class ArchiveReader : public Reader {
public:
  ArchiveReader(bool logLoading) : _logLoading(logLoading) {}

  bool canParse(file_magic magic, StringRef,
                const MemoryBuffer &) const override {
    return (magic == llvm::sys::fs::file_magic::archive);
  }

  std::error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const Registry &reg,
            std::vector<std::unique_ptr<File>> &result) const override {
    MemoryBuffer &buff = *mb;
    // Make Archive object which will be owned by FileArchive object.
    std::error_code ec;
    Archive *archive = new Archive(mb->getMemBufferRef(), ec);
    if (ec)
      return ec;
    StringRef path = buff.getBufferIdentifier();
    // Construct FileArchive object.
    std::unique_ptr<FileArchive> file(
        new FileArchive(reg, archive, path, false, _logLoading));
    ec = file->buildTableOfContents();
    if (ec)
      return ec;

    result.push_back(std::move(file));
    return std::error_code();
  }

private:
  bool _logLoading;
};

} // anonymous namespace

void Registry::addSupportArchives(bool logLoading) {
  add(std::unique_ptr<Reader>(new ArchiveReader(logLoading)));
}

} // end namespace lld
