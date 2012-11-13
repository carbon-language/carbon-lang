//===- lib/ReaderWriter/ReaderArchive.cpp - Archive Library Reader--------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
#include "lld/ReaderWriter/ReaderArchive.h"

namespace lld 
{
// The FileArchive class represents an Archive Library file
class FileArchive : public ArchiveLibraryFile {
public:

  virtual ~FileArchive() { }

  /// Check if any member of the archive contains an Atom with the
  /// specified name and return the File object for that member, or nullptr.
  virtual const File *find(StringRef name, bool dataSymbolOnly) const {
    error_code ec;  
    llvm::object::Archive::child_iterator ci;

    ci = _archive.get()->findSym(name);
    if (ci == _archive->end_children()) 
      return nullptr;
    
    if (dataSymbolOnly && (ec = isDataSymbol(ci->getBuffer(), name)))
      return nullptr;
    
    std::vector<std::unique_ptr<File>> result;

    if ((ec = _options.reader()->parseFile(std::unique_ptr<MemoryBuffer>
                                           (ci->getBuffer()), result)))
      return nullptr;

    assert(result.size() == 1);

    // give up the pointer so that this object no longer manages it
    for (std::unique_ptr<File> &f : result) {
      return f.release();
    }

    return nullptr;
  }

  virtual void addAtom(const Atom&) {
    llvm_unreachable("cannot add atoms to archive files");
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
  error_code isDataSymbol(MemoryBuffer *mb, StringRef symbol) const
  {
    llvm::object::ObjectFile *obj = 
                  llvm::object::ObjectFile::createObjectFile(mb);
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
  llvm::MemoryBuffer *_mb;
  std::unique_ptr<llvm::object::Archive> _archive;
  const ReaderOptionsArchive _options;
  atom_collection_vector<DefinedAtom>       _definedAtoms;
  atom_collection_vector<UndefinedAtom>     _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom>      _absoluteAtoms;

public:
  /// only subclasses of ArchiveLibraryFile can be instantiated 
  explicit FileArchive(llvm::MemoryBuffer *mb, 
                       const ReaderOptionsArchive &options, 
                       error_code &ec)
                      :ArchiveLibraryFile(mb->getBufferIdentifier()),
                       _mb(mb),
                       _archive(nullptr),
                       _options(options) { 
    auto *archive_obj = new llvm::object::Archive(mb, ec);
    if (ec) 
      return;
    _archive.reset(archive_obj);
  }
}; // class FileArchive

// Returns a vector of Files that are contained in the archive file 
// pointed to by the MemoryBuffer
error_code ReaderArchive::parseFile(std::unique_ptr<llvm::MemoryBuffer> mb,
		std::vector<std::unique_ptr<File>> &result) {
  error_code ec;
  
  if (_options.isForceLoad())
  {
    _archive.reset(new llvm::object::Archive(mb.release(), ec));
    if (ec)
      return ec;
    
    for (auto mf = _archive->begin_children(), 
              me = _archive->end_children(); mf != me; ++mf)
    {
    	if ((ec = _options.reader()->parseFile(std::unique_ptr<MemoryBuffer>
                                             (mf->getBuffer()), result)))
        return ec;
    }
  } else {
    std::unique_ptr<File> f;
    f.reset(new FileArchive(mb.release(), _options, ec));
    if (ec)
      return ec;

    result.push_back(std::move(f));
  }
  return llvm::error_code::success();
}

} // namespace lld
