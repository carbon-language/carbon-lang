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

#include "llvm/ADT/StringMap.h"

#include "Atoms.h"

#include "lld/Core/Simple.h"

namespace lld {
namespace mach_o {

using lld::mach_o::normalized::Section;

class MachOFile : public SimpleFile {
public:
  MachOFile(StringRef path) : SimpleFile(path) {}

  void addDefinedAtom(StringRef name, Atom::Scope scope,
                      DefinedAtom::ContentType type, DefinedAtom::Merge merge,
                      uint64_t sectionOffset, uint64_t contentSize, 
                      bool copyRefs, const Section *inSection) {
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
                                          content);
    addAtomForSection(inSection, atom, sectionOffset);
  }

  void addDefinedAtomInCustomSection(StringRef name, Atom::Scope scope,
                      DefinedAtom::ContentType type, DefinedAtom::Merge merge,
                      uint64_t sectionOffset, uint64_t contentSize, 
                      StringRef sectionName, bool copyRefs, 
                      const Section *inSection) {
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
                                                  merge, content, sectionName);
    addAtomForSection(inSection, atom, sectionOffset);
  }

  void addZeroFillDefinedAtom(StringRef name, Atom::Scope scope,
                              uint64_t sectionOffset, uint64_t size,
                              bool copyRefs, const Section *inSection) {
    if (copyRefs) {
      // Make a copy of the atom's name and content that is owned by this file.
      name = name.copy(_allocator);
    }
    MachODefinedAtom *atom =
        new (_allocator) MachODefinedAtom(*this, name, scope, size);
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



} // end namespace mach_o
} // end namespace lld

#endif
