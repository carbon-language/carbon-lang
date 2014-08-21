//===- lib/ReaderWriter/MachO/File.h --------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_MACHO_FILE_H
#define LLD_READER_WRITER_MACHO_FILE_H

#include "Atoms.h"
#include "MachONormalizedFile.h"

#include "lld/Core/Simple.h"
#include "lld/Core/SharedLibraryFile.h"

#include "llvm/ADT/StringMap.h"

#include <unordered_map>

namespace lld {
namespace mach_o {

using lld::mach_o::normalized::Section;

class MachOFile : public SimpleFile {
public:
  MachOFile(StringRef path) : SimpleFile(path) {}

  void addDefinedAtom(StringRef name, Atom::Scope scope,
                      DefinedAtom::ContentType type, DefinedAtom::Merge merge,
                      uint64_t sectionOffset, uint64_t contentSize, bool thumb,
                      bool noDeadStrip, bool copyRefs,
                      const Section *inSection) {
    assert(sectionOffset+contentSize <= inSection->content.size());
    ArrayRef<uint8_t> content = inSection->content.slice(sectionOffset,
                                                        contentSize);
    if (copyRefs) {
      // Make a copy of the atom's name and content that is owned by this file.
      name = name.copy(_allocator);
      content = content.copy(_allocator);
    }
    MachODefinedAtom *atom =
        new (_allocator) MachODefinedAtom(*this, name, scope, type, merge,
                                          thumb, noDeadStrip, content);
    addAtomForSection(inSection, atom, sectionOffset);
  }

  void addDefinedAtomInCustomSection(StringRef name, Atom::Scope scope,
                      DefinedAtom::ContentType type, DefinedAtom::Merge merge,
                      bool thumb, bool noDeadStrip, uint64_t sectionOffset,
                      uint64_t contentSize, StringRef sectionName,
                      bool copyRefs, const Section *inSection) {
    assert(sectionOffset+contentSize <= inSection->content.size());
    ArrayRef<uint8_t> content = inSection->content.slice(sectionOffset,
                                                        contentSize);
   if (copyRefs) {
      // Make a copy of the atom's name and content that is owned by this file.
      name = name.copy(_allocator);
      content = content.copy(_allocator);
      sectionName = sectionName.copy(_allocator);
    }
    MachODefinedCustomSectionAtom *atom =
        new (_allocator) MachODefinedCustomSectionAtom(*this, name, scope, type,
                                                        merge, thumb,
                                                        noDeadStrip, content,
                                                        sectionName);
    addAtomForSection(inSection, atom, sectionOffset);
  }

  void addZeroFillDefinedAtom(StringRef name, Atom::Scope scope,
                              uint64_t sectionOffset, uint64_t size,
                              bool noDeadStrip, bool copyRefs,
                              const Section *inSection) {
    if (copyRefs) {
      // Make a copy of the atom's name and content that is owned by this file.
      name = name.copy(_allocator);
    }
    MachODefinedAtom *atom =
       new (_allocator) MachODefinedAtom(*this, name, scope, size, noDeadStrip);
    addAtomForSection(inSection, atom, sectionOffset);
  }

  void addUndefinedAtom(StringRef name, bool copyRefs) {
    if (copyRefs) {
      // Make a copy of the atom's name that is owned by this file.
      name = name.copy(_allocator);
    }
    SimpleUndefinedAtom *atom =
        new (_allocator) SimpleUndefinedAtom(*this, name);
    addAtom(*atom);
    _undefAtoms[name] = atom;
  }

  void addTentativeDefAtom(StringRef name, Atom::Scope scope, uint64_t size,
                           DefinedAtom::Alignment align, bool copyRefs) {
    if (copyRefs) {
      // Make a copy of the atom's name that is owned by this file.
      name = name.copy(_allocator);
    }
    MachOTentativeDefAtom *atom =
        new (_allocator) MachOTentativeDefAtom(*this, name, scope, size, align);
    addAtom(*atom);
    _undefAtoms[name] = atom;
  }
  
  /// Search this file for an the atom from 'section' that covers
  /// 'offsetInSect'.  Returns nullptr is no atom found.
  MachODefinedAtom *findAtomCoveringAddress(const Section &section,
                                            uint64_t offsetInSect,
                                            uint32_t *foundOffsetAtom=nullptr) {
    auto pos = _sectionAtoms.find(&section);
    if (pos == _sectionAtoms.end())
      return nullptr;
    auto vec = pos->second;
    assert(offsetInSect < section.content.size());
    // Vector of atoms for section are already sorted, so do binary search.
    auto atomPos = std::lower_bound(vec.begin(), vec.end(), offsetInSect, 
        [offsetInSect](const SectionOffsetAndAtom &ao, 
                       uint64_t targetAddr) -> bool {
          // Each atom has a start offset of its slice of the
          // section's content. This compare function must return true
          // iff the atom's range is before the offset being searched for.
          uint64_t atomsEndOffset = ao.offset+ao.atom->rawContent().size();
          return (atomsEndOffset <= offsetInSect);
        });
    if (atomPos == vec.end())
      return nullptr;
    if (foundOffsetAtom)
      *foundOffsetAtom = offsetInSect - atomPos->offset;
    return atomPos->atom;
  }
  
  /// Searches this file for an UndefinedAtom named 'name'. Returns
  /// nullptr is no such atom found.
  const lld::Atom *findUndefAtom(StringRef name) {
    auto pos = _undefAtoms.find(name);
    if (pos == _undefAtoms.end())
      return nullptr;
    return pos->second;
  }
  
  typedef std::function<void (MachODefinedAtom* atom)> DefinedAtomVisitor;

  void eachDefinedAtom(DefinedAtomVisitor vistor) {
    for (auto &sectAndAtoms : _sectionAtoms) {
      for (auto &offAndAtom : sectAndAtoms.second) {
        vistor(offAndAtom.atom);
      }
    }
  }

  llvm::BumpPtrAllocator &allocator() { return _allocator; }
  
private:
  struct SectionOffsetAndAtom { uint64_t offset;  MachODefinedAtom *atom; };

  void addAtomForSection(const Section *inSection, MachODefinedAtom* atom, 
                         uint64_t sectionOffset) {
    SectionOffsetAndAtom offAndAtom;
    offAndAtom.offset = sectionOffset;
    offAndAtom.atom   = atom;
     _sectionAtoms[inSection].push_back(offAndAtom);
    addAtom(*atom);
  }

  
  typedef llvm::DenseMap<const normalized::Section *, 
                         std::vector<SectionOffsetAndAtom>>  SectionToAtoms;
  typedef llvm::StringMap<const lld::Atom *> NameToAtom;

  llvm::BumpPtrAllocator  _allocator;
  SectionToAtoms          _sectionAtoms;
  NameToAtom              _undefAtoms;
};

class MachODylibFile : public SharedLibraryFile {
public:
  MachODylibFile(StringRef path, StringRef installName)
      : SharedLibraryFile(path), _installName(installName) {
  }

  virtual const SharedLibraryAtom *exports(StringRef name, bool isData) const {
    // Pass down _installName and _allocator so that if this requested symbol
    // is re-exported through this dylib, the SharedLibraryAtom's loadName()
    // is this dylib installName and not the implementation dylib's.
    // NOTE: isData is not needed for dylibs (it matters for static libs).
    return exports(name, _installName, _allocator);
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

  /// Adds symbol name that this dylib exports. The corresponding
  /// SharedLibraryAtom is created lazily (since most symbols are not used).
  void addExportedSymbol(StringRef name, bool weakDef, bool copyRefs) {
    if (copyRefs) {
      name = name.copy(_allocator);
    }
    AtomAndFlags info(weakDef);
    _nameToAtom[name] = info;
  }

  void addReExportedDylib(StringRef dylibPath) {
    _reExportedDylibs.emplace_back(dylibPath);
  }

  StringRef installName() { return _installName; }

  typedef std::function<MachODylibFile *(StringRef)> FindDylib;

  void loadReExportedDylibs(FindDylib find) {
    for (ReExportedDylib &entry : _reExportedDylibs) {
      entry.file = find(entry.path);
    }
  }

private:
  const SharedLibraryAtom *exports(StringRef name, StringRef installName,
                                   llvm::BumpPtrAllocator &allocator) const {
    // First, check if requested symbol is directly implemented by this dylib.
    auto entry = _nameToAtom.find(name);
    if (entry != _nameToAtom.end()) {
      if (!entry->second.atom) {
        // Lazily create SharedLibraryAtom.
        entry->second.atom =
          new (allocator) MachOSharedLibraryAtom(*this, name, installName,
                                                 entry->second.weakDef);
      }
      return entry->second.atom;
    }

    // Next, check if symbol is implemented in some re-exported dylib.
    for (const ReExportedDylib &dylib : _reExportedDylibs) {
      assert(dylib.file);
      auto atom = dylib.file->exports(name, installName, allocator);
      if (atom)
        return atom;
    }

    // Symbol not exported or re-exported by this dylib.
    return nullptr;
  }


  struct ReExportedDylib {
    ReExportedDylib(StringRef p) : path(p), file(nullptr) { }
    StringRef       path;
    MachODylibFile *file;
  };

  struct AtomAndFlags {
    AtomAndFlags() : atom(nullptr), weakDef(false) { }
    AtomAndFlags(bool weak) : atom(nullptr), weakDef(weak) { }
    const SharedLibraryAtom  *atom;
    bool                      weakDef;
  };

  StringRef _installName;
  atom_collection_vector<DefinedAtom>        _definedAtoms;
  atom_collection_vector<UndefinedAtom>      _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom>  _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom>       _absoluteAtoms;
  std::vector<ReExportedDylib>               _reExportedDylibs;
  mutable std::unordered_map<StringRef, AtomAndFlags> _nameToAtom;
  mutable llvm::BumpPtrAllocator _allocator;
};

} // end namespace mach_o
} // end namespace lld

#endif
