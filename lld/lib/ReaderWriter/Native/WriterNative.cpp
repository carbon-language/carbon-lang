//===- lib/ReaderWriter/Native/WriterNative.cpp ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/Writer.h"
#include "lld/Core/File.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

#include "NativeFileFormat.h"

#include <vector>

namespace lld {
namespace native {

///
/// Class for writing native object files.
///
class Writer : public lld::Writer {
public:
  Writer(const TargetInfo &ti) {}

  virtual error_code writeFile(const lld::File &file, StringRef outPath) {
    // reserve first byte for unnamed atoms
    _stringPool.push_back('\0');
    // visit all atoms
    for ( const DefinedAtom *defAtom : file.defined() ) {
      this->addIVarsForDefinedAtom(*defAtom);
    }
    for ( const UndefinedAtom *undefAtom : file.undefined() ) {
      this->addIVarsForUndefinedAtom(*undefAtom);
    }
    for ( const SharedLibraryAtom *shlibAtom : file.sharedLibrary() ) {
      this->addIVarsForSharedLibraryAtom(*shlibAtom);
    }
    for ( const AbsoluteAtom *absAtom : file.absolute() ) {
      this->addIVarsForAbsoluteAtom(*absAtom);
    }

    // construct file header based on atom information accumulated
    this->makeHeader();

    std::string errorInfo;
    llvm::raw_fd_ostream out(outPath.data(), errorInfo,
                              llvm::raw_fd_ostream::F_Binary);
    if (!errorInfo.empty())
      return error_code::success(); // FIXME

    this->write(out);

    return error_code::success();
  }

  virtual ~Writer() {
  }

private:

  // write the lld::File in native format to the specified stream
  void write(raw_ostream &out) {
    assert( out.tell() == 0 );
    out.write((char*)_headerBuffer, _headerBufferSize);

    if (!_definedAtomIvars.empty()) {
      assert( out.tell() == findChunk(NCS_DefinedAtomsV1).fileOffset );
      out.write((char*)&_definedAtomIvars[0],
                _definedAtomIvars.size()*sizeof(NativeDefinedAtomIvarsV1));
    }

    if (!_attributes.empty()) {
      assert( out.tell() == findChunk(NCS_AttributesArrayV1).fileOffset );
      out.write((char*)&_attributes[0],
                _attributes.size()*sizeof(NativeAtomAttributesV1));
    }

    if ( !_undefinedAtomIvars.empty() ) {
      assert( out.tell() == findChunk(NCS_UndefinedAtomsV1).fileOffset );
      out.write((char*)&_undefinedAtomIvars[0],
              _undefinedAtomIvars.size()*sizeof(NativeUndefinedAtomIvarsV1));
    }

     if ( !_sharedLibraryAtomIvars.empty() ) {
      assert( out.tell() == findChunk(NCS_SharedLibraryAtomsV1).fileOffset );
      out.write((char*)&_sharedLibraryAtomIvars[0],
              _sharedLibraryAtomIvars.size()
              * sizeof(NativeSharedLibraryAtomIvarsV1));
    }

    if ( !_absoluteAtomIvars.empty() ) {
      assert( out.tell() == findChunk(NCS_AbsoluteAtomsV1).fileOffset );
      out.write((char*)&_absoluteAtomIvars[0],
              _absoluteAtomIvars.size()
              * sizeof(NativeAbsoluteAtomIvarsV1));
    }
    if (!_absAttributes.empty()) {
      assert( out.tell() == findChunk(NCS_AbsoluteAttributesV1).fileOffset );
      out.write((char*)&_absAttributes[0],
                _absAttributes.size()*sizeof(NativeAtomAttributesV1));
    }

    if (!_stringPool.empty()) {
      assert( out.tell() == findChunk(NCS_Strings).fileOffset );
      out.write(&_stringPool[0], _stringPool.size());
    }

    if ( !_references.empty() ) {
      assert( out.tell() == findChunk(NCS_ReferencesArrayV1).fileOffset );
      out.write((char*)&_references[0],
              _references.size()*sizeof(NativeReferenceIvarsV1));
    }

    if ( !_targetsTableIndex.empty() ) {
      assert( out.tell() == findChunk(NCS_TargetsTable).fileOffset );
      writeTargetTable(out);
    }

    if ( !_addendsTableIndex.empty() ) {
      assert( out.tell() == findChunk(NCS_AddendsTable).fileOffset );
      writeAddendTable(out);
    }

    if (!_contentPool.empty()) {
      assert( out.tell() == findChunk(NCS_Content).fileOffset );
      out.write((char*)&_contentPool[0], _contentPool.size());
    }
  }

  void addIVarsForDefinedAtom(const DefinedAtom& atom) {
    _definedAtomIndex[&atom] = _definedAtomIvars.size();
    NativeDefinedAtomIvarsV1 ivar;
    unsigned refsCount;
    ivar.nameOffset = getNameOffset(atom);
    ivar.attributesOffset = getAttributeOffset(atom);
    ivar.referencesStartIndex = getReferencesIndex(atom, refsCount);
    ivar.referencesCount = refsCount;
    ivar.contentOffset = getContentOffset(atom);
    ivar.contentSize = atom.size();
    _definedAtomIvars.push_back(ivar);
  }

  void addIVarsForUndefinedAtom(const UndefinedAtom& atom) {
    _undefinedAtomIndex[&atom] = _undefinedAtomIvars.size();
    NativeUndefinedAtomIvarsV1 ivar;
    ivar.nameOffset = getNameOffset(atom);
    ivar.flags = (atom.canBeNull() & 0x03);
    _undefinedAtomIvars.push_back(ivar);
  }

  void addIVarsForSharedLibraryAtom(const SharedLibraryAtom& atom) {
    _sharedLibraryAtomIndex[&atom] = _sharedLibraryAtomIvars.size();
    NativeSharedLibraryAtomIvarsV1 ivar;
    ivar.nameOffset = getNameOffset(atom);
    ivar.loadNameOffset = getSharedLibraryNameOffset(atom.loadName());
    ivar.flags = atom.canBeNullAtRuntime();
    _sharedLibraryAtomIvars.push_back(ivar);
  }

  void addIVarsForAbsoluteAtom(const AbsoluteAtom& atom) {
    _absoluteAtomIndex[&atom] = _absoluteAtomIvars.size();
    NativeAbsoluteAtomIvarsV1 ivar;
    ivar.nameOffset = getNameOffset(atom);
    ivar.attributesOffset = getAttributeOffset(atom);
    ivar.reserved = 0;
    ivar.value = atom.value();
    _absoluteAtomIvars.push_back(ivar);
  }

  // fill out native file header and chunk directory
  void makeHeader() {
    const bool hasDefines = !_definedAtomIvars.empty();
    const bool hasUndefines = !_undefinedAtomIvars.empty();
    const bool hasSharedLibraries = !_sharedLibraryAtomIvars.empty();
    const bool hasAbsolutes = !_absoluteAtomIvars.empty();
    const bool hasReferences = !_references.empty();
    const bool hasTargetsTable = !_targetsTableIndex.empty();
    const bool hasAddendTable = !_addendsTableIndex.empty();
    const bool hasContent = !_contentPool.empty();

    int chunkCount = 1; // always have string pool chunk
    if ( hasDefines ) chunkCount += 2;
    if ( hasUndefines ) ++chunkCount;
    if ( hasSharedLibraries ) ++chunkCount;
    if ( hasAbsolutes ) chunkCount += 2;
    if ( hasReferences ) ++chunkCount;
    if ( hasTargetsTable ) ++chunkCount;
    if ( hasAddendTable ) ++chunkCount;
    if ( hasContent ) ++chunkCount;

    _headerBufferSize = sizeof(NativeFileHeader)
                         + chunkCount*sizeof(NativeChunk);
    _headerBuffer = reinterpret_cast<NativeFileHeader*>
                               (operator new(_headerBufferSize, std::nothrow));
    NativeChunk *chunks =
      reinterpret_cast<NativeChunk*>(reinterpret_cast<char*>(_headerBuffer)
                                     + sizeof(NativeFileHeader));
    memcpy(_headerBuffer->magic, NATIVE_FILE_HEADER_MAGIC, 16);
    _headerBuffer->endian = NFH_LittleEndian;
    _headerBuffer->architecture = 0;
    _headerBuffer->fileSize = 0;
    _headerBuffer->chunkCount = chunkCount;

    // create chunk for defined atom ivar array
    int nextIndex = 0;
    uint32_t nextFileOffset = _headerBufferSize;
    if ( hasDefines ) {
      NativeChunk& chd = chunks[nextIndex++];
      chd.signature = NCS_DefinedAtomsV1;
      chd.fileOffset = nextFileOffset;
      chd.fileSize = _definedAtomIvars.size()*sizeof(NativeDefinedAtomIvarsV1);
      chd.elementCount = _definedAtomIvars.size();
      nextFileOffset = chd.fileOffset + chd.fileSize;

      // create chunk for attributes
      NativeChunk& cha = chunks[nextIndex++];
      cha.signature = NCS_AttributesArrayV1;
      cha.fileOffset = nextFileOffset;
      cha.fileSize = _attributes.size()*sizeof(NativeAtomAttributesV1);
      cha.elementCount = _attributes.size();
      nextFileOffset = cha.fileOffset + cha.fileSize;
    }

    // create chunk for undefined atom array
    if ( hasUndefines ) {
      NativeChunk& chu = chunks[nextIndex++];
      chu.signature = NCS_UndefinedAtomsV1;
      chu.fileOffset = nextFileOffset;
      chu.fileSize = _undefinedAtomIvars.size() *
                                            sizeof(NativeUndefinedAtomIvarsV1);
      chu.elementCount = _undefinedAtomIvars.size();
      nextFileOffset = chu.fileOffset + chu.fileSize;
    }

    // create chunk for shared library atom array
    if ( hasSharedLibraries ) {
      NativeChunk& chsl = chunks[nextIndex++];
      chsl.signature = NCS_SharedLibraryAtomsV1;
      chsl.fileOffset = nextFileOffset;
      chsl.fileSize = _sharedLibraryAtomIvars.size() *
                                        sizeof(NativeSharedLibraryAtomIvarsV1);
      chsl.elementCount = _sharedLibraryAtomIvars.size();
      nextFileOffset = chsl.fileOffset + chsl.fileSize;
    }

     // create chunk for shared library atom array
    if ( hasAbsolutes ) {
      NativeChunk& chabs = chunks[nextIndex++];
      chabs.signature = NCS_AbsoluteAtomsV1;
      chabs.fileOffset = nextFileOffset;
      chabs.fileSize = _absoluteAtomIvars.size() *
                                        sizeof(NativeAbsoluteAtomIvarsV1);
      chabs.elementCount = _absoluteAtomIvars.size();
      nextFileOffset = chabs.fileOffset + chabs.fileSize;

      // create chunk for attributes
      NativeChunk& cha = chunks[nextIndex++];
      cha.signature = NCS_AbsoluteAttributesV1;
      cha.fileOffset = nextFileOffset;
      cha.fileSize = _absAttributes.size()*sizeof(NativeAtomAttributesV1);
      cha.elementCount = _absAttributes.size();
      nextFileOffset = cha.fileOffset + cha.fileSize;
    }

    // create chunk for symbol strings
    // pad end of string pool to 4-bytes
    while ( (_stringPool.size() % 4) != 0 )
      _stringPool.push_back('\0');
    NativeChunk& chs = chunks[nextIndex++];
    chs.signature = NCS_Strings;
    chs.fileOffset = nextFileOffset;
    chs.fileSize = _stringPool.size();
    chs.elementCount = _stringPool.size();
    nextFileOffset = chs.fileOffset + chs.fileSize;

    // create chunk for references
    if ( hasReferences ) {
      NativeChunk& chr = chunks[nextIndex++];
      chr.signature = NCS_ReferencesArrayV1;
      chr.fileOffset = nextFileOffset;
      chr.fileSize = _references.size() * sizeof(NativeReferenceIvarsV1);
      chr.elementCount = _references.size();
      nextFileOffset = chr.fileOffset + chr.fileSize;
    }

    // create chunk for target table
    if ( hasTargetsTable ) {
      NativeChunk& cht = chunks[nextIndex++];
      cht.signature = NCS_TargetsTable;
      cht.fileOffset = nextFileOffset;
      cht.fileSize = _targetsTableIndex.size() * sizeof(uint32_t);
      cht.elementCount = _targetsTableIndex.size();
      nextFileOffset = cht.fileOffset + cht.fileSize;
    }

    // create chunk for addend table
    if ( hasAddendTable ) {
      NativeChunk& chad = chunks[nextIndex++];
      chad.signature = NCS_AddendsTable;
      chad.fileOffset = nextFileOffset;
      chad.fileSize = _addendsTableIndex.size() * sizeof(Reference::Addend);
      chad.elementCount = _addendsTableIndex.size();
      nextFileOffset = chad.fileOffset + chad.fileSize;
    }

    // create chunk for content
    if ( hasContent ) {
      NativeChunk& chc = chunks[nextIndex++];
      chc.signature = NCS_Content;
      chc.fileOffset = nextFileOffset;
      chc.fileSize = _contentPool.size();
      chc.elementCount = _contentPool.size();
      nextFileOffset = chc.fileOffset + chc.fileSize;
    }

    _headerBuffer->fileSize = nextFileOffset;
  }

  // scan header to find particular chunk
  NativeChunk& findChunk(uint32_t signature) {
    const uint32_t chunkCount = _headerBuffer->chunkCount;
    NativeChunk* chunks =
      reinterpret_cast<NativeChunk*>(reinterpret_cast<char*>(_headerBuffer)
                                     + sizeof(NativeFileHeader));
    for (uint32_t i=0; i < chunkCount; ++i) {
      if ( chunks[i].signature == signature )
        return chunks[i];
    }
    assert(0 && "findChunk() signature not found");
    static NativeChunk x; return x; // suppress warning
  }

  // append atom name to string pool and return offset
  uint32_t getNameOffset(const Atom& atom) {
    return this->getNameOffset(atom.name());
  }

  // check if name is already in pool or append and return offset
  uint32_t getSharedLibraryNameOffset(StringRef name) {
    assert( ! name.empty() );
    // look to see if this library name was used by another atom
    for(NameToOffsetVector::iterator it = _sharedLibraryNames.begin();
                                    it != _sharedLibraryNames.end(); ++it) {
      if ( name.equals(it->first) )
        return it->second;
    }
    // first use of this library name
    uint32_t result = this->getNameOffset(name);
    _sharedLibraryNames.push_back(std::make_pair(name, result));
    return result;
  }

  // append atom name to string pool and return offset
  uint32_t getNameOffset(StringRef name) {
    if ( name.empty() )
      return 0;
    uint32_t result = _stringPool.size();
    _stringPool.insert(_stringPool.end(), name.begin(), name.end());
    _stringPool.push_back(0);
    return result;
  }

  // append atom cotent to content pool and return offset
  uint32_t getContentOffset(const class DefinedAtom& atom) {
    if ((atom.contentType() == DefinedAtom::typeZeroFill ) ||
        (atom.contentType() == DefinedAtom::typeZeroFillFast))
      return 0;
    uint32_t result = _contentPool.size();
    ArrayRef<uint8_t> cont = atom.rawContent();
    _contentPool.insert(_contentPool.end(), cont.begin(), cont.end());
    return result;
  }

  // reuse existing attributes entry or create a new one and return offet
  uint32_t getAttributeOffset(const class DefinedAtom& atom) {
    NativeAtomAttributesV1 attrs;
    computeAttributesV1(atom, attrs);
    for(unsigned int i=0; i < _attributes.size(); ++i) {
      if ( !memcmp(&_attributes[i], &attrs, sizeof(NativeAtomAttributesV1)) ) {
        // found that this set of attributes already used, so re-use
        return i * sizeof(NativeAtomAttributesV1);
      }
    }
    // append new attribute set to end
    uint32_t result = _attributes.size() * sizeof(NativeAtomAttributesV1);
    _attributes.push_back(attrs);
    return result;
  }

  uint32_t getAttributeOffset(const class AbsoluteAtom& atom) {
    NativeAtomAttributesV1 attrs;
    computeAbsoluteAttributes(atom, attrs);
    for(unsigned int i=0; i < _absAttributes.size(); ++i) {
      if ( !memcmp(&_absAttributes[i], &attrs, sizeof(NativeAtomAttributesV1)) ) {
        // found that this set of attributes already used, so re-use
        return i * sizeof(NativeAtomAttributesV1);
      }
    }
    // append new attribute set to end
    uint32_t result = _absAttributes.size() * sizeof(NativeAtomAttributesV1);
    _absAttributes.push_back(attrs);
    return result;
  }

  uint32_t sectionNameOffset(const class DefinedAtom& atom) {
    // if section based on content, then no custom section name available
    if ( atom.sectionChoice() == DefinedAtom::sectionBasedOnContent )
      return 0;
    StringRef name = atom.customSectionName();
    assert( ! name.empty() );
    // look to see if this section name was used by another atom
    for(NameToOffsetVector::iterator it=_sectionNames.begin();
                                            it != _sectionNames.end(); ++it) {
      if ( name.equals(it->first) )
        return it->second;
    }
    // first use of this section name
    uint32_t result = this->getNameOffset(name);
    _sectionNames.push_back(std::make_pair(name, result));
    return result;
  }

  void computeAttributesV1(const class DefinedAtom& atom,
                                                NativeAtomAttributesV1& attrs) {
    attrs.sectionNameOffset = sectionNameOffset(atom);
    attrs.align2            = atom.alignment().powerOf2;
    attrs.alignModulus      = atom.alignment().modulus;
    attrs.scope             = atom.scope();
    attrs.interposable      = atom.interposable();
    attrs.merge             = atom.merge();
    attrs.contentType       = atom.contentType();
    attrs.sectionChoiceAndPosition
                          = atom.sectionChoice() << 4 | atom.sectionPosition();
    attrs.deadStrip         = atom.deadStrip();
    attrs.permissions       = atom.permissions();
    attrs.alias             = atom.isAlias();
  }

  void computeAbsoluteAttributes(const class AbsoluteAtom& atom,
                                                NativeAtomAttributesV1& attrs) {
    attrs.scope       = atom.scope();
  }

  // add references for this atom in a contiguous block in NCS_ReferencesArrayV1
  uint32_t getReferencesIndex(const DefinedAtom& atom, unsigned& count) {
    count = 0;
    size_t startRefSize = _references.size();
    uint32_t result = startRefSize;
    for (const Reference *ref : atom) {
      NativeReferenceIvarsV1 nref;
      nref.offsetInAtom = ref->offsetInAtom();
      nref.kind = ref->kind();
      nref.targetIndex = this->getTargetIndex(ref->target());
      nref.addendIndex = this->getAddendIndex(ref->addend());
      _references.push_back(nref);
    }
    count = _references.size() - startRefSize;
    if ( count == 0 )
      return 0;
    else
      return result;
  }

  uint32_t getTargetIndex(const Atom* target) {
    if ( target == nullptr )
      return NativeReferenceIvarsV1::noTarget;
    TargetToIndex::const_iterator pos = _targetsTableIndex.find(target);
    if ( pos != _targetsTableIndex.end() ) {
      return pos->second;
    }
    uint32_t result = _targetsTableIndex.size();
    _targetsTableIndex[target] = result;
    return result;
  }

  void writeTargetTable(raw_ostream &out) {
    // Build table of target indexes
    uint32_t maxTargetIndex = _targetsTableIndex.size();
    assert(maxTargetIndex > 0);
    std::vector<uint32_t> targetIndexes(maxTargetIndex);
    for (TargetToIndex::iterator it = _targetsTableIndex.begin();
                                 it != _targetsTableIndex.end(); ++it) {
      const Atom* atom = it->first;
      uint32_t targetIndex = it->second;
      assert(targetIndex < maxTargetIndex);
      uint32_t atomIndex = 0;
      TargetToIndex::iterator pos = _definedAtomIndex.find(atom);
      if ( pos != _definedAtomIndex.end() ) {
        atomIndex = pos->second;
      }
      else {
        pos = _undefinedAtomIndex.find(atom);
        if ( pos != _undefinedAtomIndex.end() ) {
          atomIndex = pos->second + _definedAtomIvars.size();
        }
        else {
          pos = _sharedLibraryAtomIndex.find(atom);
          if ( pos != _sharedLibraryAtomIndex.end() ) {
            assert(pos != _sharedLibraryAtomIndex.end());
            atomIndex = pos->second
                      + _definedAtomIvars.size()
                      + _undefinedAtomIndex.size();
          }
          else {
            pos = _absoluteAtomIndex.find(atom);
            assert(pos != _absoluteAtomIndex.end());
            atomIndex = pos->second
                      + _definedAtomIvars.size()
                      + _undefinedAtomIndex.size()
                      + _sharedLibraryAtomIndex.size();
         }
        }
      }
      targetIndexes[targetIndex] = atomIndex;
    }
    // write table
    out.write((char*)&targetIndexes[0], maxTargetIndex*sizeof(uint32_t));
  }

  uint32_t getAddendIndex(Reference::Addend addend) {
    if ( addend == 0 )
      return 0; // addend index zero is used to mean "no addend"
    AddendToIndex::const_iterator pos = _addendsTableIndex.find(addend);
    if ( pos != _addendsTableIndex.end() ) {
      return pos->second;
    }
    uint32_t result = _addendsTableIndex.size() + 1; // one-based index
    _addendsTableIndex[addend] = result;
    return result;
  }

  void writeAddendTable(raw_ostream &out) {
    // Build table of addends
    uint32_t maxAddendIndex = _addendsTableIndex.size();
    std::vector<Reference::Addend> addends(maxAddendIndex);
    for (AddendToIndex::iterator it = _addendsTableIndex.begin();
                                 it != _addendsTableIndex.end(); ++it) {
      Reference::Addend addend = it->first;
      uint32_t index = it->second;
      assert(index <= maxAddendIndex);
      addends[index-1] = addend;
    }
    // write table
    out.write((char*)&addends[0], maxAddendIndex*sizeof(Reference::Addend));
  }

  typedef std::vector<std::pair<StringRef, uint32_t> > NameToOffsetVector;

  typedef llvm::DenseMap<const Atom*, uint32_t> TargetToIndex;
  typedef llvm::DenseMap<Reference::Addend, uint32_t> AddendToIndex;

  NativeFileHeader*                       _headerBuffer;
  size_t                                  _headerBufferSize;
  std::vector<char>                       _stringPool;
  std::vector<uint8_t>                    _contentPool;
  std::vector<NativeDefinedAtomIvarsV1>   _definedAtomIvars;
  std::vector<NativeAtomAttributesV1>     _attributes;
  std::vector<NativeAtomAttributesV1>     _absAttributes;
  std::vector<NativeUndefinedAtomIvarsV1> _undefinedAtomIvars;
  std::vector<NativeSharedLibraryAtomIvarsV1> _sharedLibraryAtomIvars;
  std::vector<NativeAbsoluteAtomIvarsV1>  _absoluteAtomIvars;
  std::vector<NativeReferenceIvarsV1>     _references;
  TargetToIndex                           _targetsTableIndex;
  TargetToIndex                           _definedAtomIndex;
  TargetToIndex                           _undefinedAtomIndex;
  TargetToIndex                           _sharedLibraryAtomIndex;
  TargetToIndex                           _absoluteAtomIndex;
  AddendToIndex                           _addendsTableIndex;
  NameToOffsetVector                      _sectionNames;
  NameToOffsetVector                      _sharedLibraryNames;
};
} // end namespace native

std::unique_ptr<Writer> createWriterNative(const TargetInfo &ti) {
  return std::unique_ptr<Writer>(new native::Writer(ti));
}
} // end namespace lld
