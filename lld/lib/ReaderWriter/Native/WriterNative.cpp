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

#include <cstdint>
#include <set>
#include <vector>

namespace lld {
namespace native {

///
/// Class for writing native object files.
///
class Writer : public lld::Writer {
public:
  Writer(const LinkingContext &context) {}

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

    maybeConvertReferencesToV1();

    // construct file header based on atom information accumulated
    this->makeHeader();

    std::string errorInfo;
    llvm::raw_fd_ostream out(outPath.data(), errorInfo,
                             llvm::sys::fs::F_None);
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
    assert(out.tell() == 0);
    out.write((char*)_headerBuffer, _headerBufferSize);

    writeChunk(out, _definedAtomIvars, NCS_DefinedAtomsV1);
    writeChunk(out, _attributes, NCS_AttributesArrayV1);
    writeChunk(out, _undefinedAtomIvars, NCS_UndefinedAtomsV1);
    writeChunk(out, _sharedLibraryAtomIvars, NCS_SharedLibraryAtomsV1);
    writeChunk(out, _absoluteAtomIvars, NCS_AbsoluteAtomsV1);
    writeChunk(out, _absAttributes, NCS_AbsoluteAttributesV1);
    writeChunk(out, _stringPool, NCS_Strings);
    writeChunk(out, _referencesV1, NCS_ReferencesArrayV1);
    writeChunk(out, _referencesV2, NCS_ReferencesArrayV2);

    if (!_targetsTableIndex.empty()) {
      assert(out.tell() == findChunk(NCS_TargetsTable).fileOffset);
      writeTargetTable(out);
    }

    if (!_addendsTableIndex.empty()) {
      assert(out.tell() == findChunk(NCS_AddendsTable).fileOffset);
      writeAddendTable(out);
    }

    writeChunk(out, _contentPool, NCS_Content);
  }

  template<class T>
  void writeChunk(raw_ostream &out, std::vector<T> &vector, uint32_t signature) {
    if (vector.empty())
      return;
    assert(out.tell() == findChunk(signature).fileOffset);
    out.write((char*)&vector[0], vector.size() * sizeof(T));
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
    ivar.fallbackNameOffset = 0;
    if (atom.fallback())
      ivar.fallbackNameOffset = getNameOffset(*atom.fallback());
    _undefinedAtomIvars.push_back(ivar);
  }

  void addIVarsForSharedLibraryAtom(const SharedLibraryAtom& atom) {
    _sharedLibraryAtomIndex[&atom] = _sharedLibraryAtomIvars.size();
    NativeSharedLibraryAtomIvarsV1 ivar;
    ivar.size = atom.size();
    ivar.nameOffset = getNameOffset(atom);
    ivar.loadNameOffset = getSharedLibraryNameOffset(atom.loadName());
    ivar.type = (uint32_t)atom.type();
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

  void convertReferencesToV1() {
    for (const NativeReferenceIvarsV2 &v2 : _referencesV2) {
      NativeReferenceIvarsV1 v1;
      v1.offsetInAtom = v2.offsetInAtom;
      v1.kindNamespace = v2.kindNamespace;
      v1.kindArch = v2.kindArch;
      v1.kindValue = v2.kindValue;
      v1.targetIndex = (v2.targetIndex == NativeReferenceIvarsV2::noTarget) ?
          NativeReferenceIvarsV1::noTarget : v2.targetIndex;
      v1.addendIndex = this->getAddendIndex(v2.addend);
      _referencesV1.push_back(v1);
    }
    _referencesV2.clear();
  }

  bool canConvertReferenceToV1(const NativeReferenceIvarsV2 &ref) {
    bool validOffset = (ref.offsetInAtom == NativeReferenceIvarsV2::noTarget) ||
        ref.offsetInAtom < NativeReferenceIvarsV1::noTarget;
    return validOffset && ref.targetIndex < UINT16_MAX;
  }

  // Convert vector of NativeReferenceIvarsV2 to NativeReferenceIvarsV1 if
  // possible.
  void maybeConvertReferencesToV1() {
    std::set<int64_t> addends;
    for (const NativeReferenceIvarsV2 &ref : _referencesV2) {
      if (!canConvertReferenceToV1(ref))
        return;
      addends.insert(ref.addend);
      if (addends.size() >= UINT16_MAX)
        return;
    }
    convertReferencesToV1();
  }

  // fill out native file header and chunk directory
  void makeHeader() {
    const bool hasDefines = !_definedAtomIvars.empty();
    const bool hasUndefines = !_undefinedAtomIvars.empty();
    const bool hasSharedLibraries = !_sharedLibraryAtomIvars.empty();
    const bool hasAbsolutes = !_absoluteAtomIvars.empty();
    const bool hasReferencesV1 = !_referencesV1.empty();
    const bool hasReferencesV2 = !_referencesV2.empty();
    const bool hasTargetsTable = !_targetsTableIndex.empty();
    const bool hasAddendTable = !_addendsTableIndex.empty();
    const bool hasContent = !_contentPool.empty();

    int chunkCount = 1; // always have string pool chunk
    if ( hasDefines ) chunkCount += 2;
    if ( hasUndefines ) ++chunkCount;
    if ( hasSharedLibraries ) ++chunkCount;
    if ( hasAbsolutes ) chunkCount += 2;
    if ( hasReferencesV1 ) ++chunkCount;
    if ( hasReferencesV2 ) ++chunkCount;
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
    memcpy(_headerBuffer->magic, NATIVE_FILE_HEADER_MAGIC,
           sizeof(_headerBuffer->magic));
    _headerBuffer->endian = NFH_LittleEndian;
    _headerBuffer->architecture = 0;
    _headerBuffer->fileSize = 0;
    _headerBuffer->chunkCount = chunkCount;

    // create chunk for defined atom ivar array
    int nextIndex = 0;
    uint32_t nextFileOffset = _headerBufferSize;
    if (hasDefines) {
      fillChunkHeader(chunks[nextIndex++], nextFileOffset, _definedAtomIvars,
                      NCS_DefinedAtomsV1);

      // create chunk for attributes
      fillChunkHeader(chunks[nextIndex++], nextFileOffset, _attributes,
                      NCS_AttributesArrayV1);
    }

    // create chunk for undefined atom array
    if (hasUndefines)
      fillChunkHeader(chunks[nextIndex++], nextFileOffset, _undefinedAtomIvars,
                      NCS_UndefinedAtomsV1);

    // create chunk for shared library atom array
    if (hasSharedLibraries)
      fillChunkHeader(chunks[nextIndex++], nextFileOffset,
                      _sharedLibraryAtomIvars, NCS_SharedLibraryAtomsV1);

     // create chunk for shared library atom array
    if (hasAbsolutes) {
      fillChunkHeader(chunks[nextIndex++], nextFileOffset, _absoluteAtomIvars,
                      NCS_AbsoluteAtomsV1);

      // create chunk for attributes
      fillChunkHeader(chunks[nextIndex++], nextFileOffset, _absAttributes,
                      NCS_AbsoluteAttributesV1);
    }

    // create chunk for symbol strings
    // pad end of string pool to 4-bytes
    while ((_stringPool.size() % 4) != 0)
      _stringPool.push_back('\0');
    fillChunkHeader(chunks[nextIndex++], nextFileOffset, _stringPool,
                    NCS_Strings);

    // create chunk for referencesV2
    if (hasReferencesV1)
      fillChunkHeader(chunks[nextIndex++], nextFileOffset, _referencesV1,
                      NCS_ReferencesArrayV1);

    // create chunk for referencesV2
    if (hasReferencesV2)
      fillChunkHeader(chunks[nextIndex++], nextFileOffset, _referencesV2,
                      NCS_ReferencesArrayV2);

    // create chunk for target table
    if (hasTargetsTable) {
      NativeChunk& cht = chunks[nextIndex++];
      cht.signature = NCS_TargetsTable;
      cht.fileOffset = nextFileOffset;
      cht.fileSize = _targetsTableIndex.size() * sizeof(uint32_t);
      cht.elementCount = _targetsTableIndex.size();
      nextFileOffset = cht.fileOffset + cht.fileSize;
    }

    // create chunk for addend table
    if (hasAddendTable) {
      NativeChunk& chad = chunks[nextIndex++];
      chad.signature = NCS_AddendsTable;
      chad.fileOffset = nextFileOffset;
      chad.fileSize = _addendsTableIndex.size() * sizeof(Reference::Addend);
      chad.elementCount = _addendsTableIndex.size();
      nextFileOffset = chad.fileOffset + chad.fileSize;
    }

    // create chunk for content
    if (hasContent)
      fillChunkHeader(chunks[nextIndex++], nextFileOffset, _contentPool,
                      NCS_Content);

    _headerBuffer->fileSize = nextFileOffset;
  }

  template<class T>
  void fillChunkHeader(NativeChunk &chunk, uint32_t &nextFileOffset,
                       const std::vector<T> &data, uint32_t signature) {
    chunk.signature = signature;
    chunk.fileOffset = nextFileOffset;
    chunk.fileSize = data.size() * sizeof(T);
    chunk.elementCount = data.size();
    nextFileOffset = chunk.fileOffset + chunk.fileSize;
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
    llvm_unreachable("findChunk() signature not found");
  }

  // append atom name to string pool and return offset
  uint32_t getNameOffset(const Atom& atom) {
    return this->getNameOffset(atom.name());
  }

  // check if name is already in pool or append and return offset
  uint32_t getSharedLibraryNameOffset(StringRef name) {
    assert(!name.empty());
    // look to see if this library name was used by another atom
    for (auto &it : _sharedLibraryNames)
      if (name.equals(it.first))
        return it.second;
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
  uint32_t getContentOffset(const DefinedAtom& atom) {
    if (!atom.occupiesDiskSpace())
      return 0;
    uint32_t result = _contentPool.size();
    ArrayRef<uint8_t> cont = atom.rawContent();
    _contentPool.insert(_contentPool.end(), cont.begin(), cont.end());
    return result;
  }

  // reuse existing attributes entry or create a new one and return offet
  uint32_t getAttributeOffset(const DefinedAtom& atom) {
    NativeAtomAttributesV1 attrs = computeAttributesV1(atom);
    return getOrPushAttribute(_attributes, attrs);
  }

  uint32_t getAttributeOffset(const AbsoluteAtom& atom) {
    NativeAtomAttributesV1 attrs = computeAbsoluteAttributes(atom);
    return getOrPushAttribute(_absAttributes, attrs);
  }

  uint32_t getOrPushAttribute(std::vector<NativeAtomAttributesV1> &dest,
                              const NativeAtomAttributesV1 &attrs) {
    for (size_t i = 0, e = dest.size(); i < e; ++i) {
      if (!memcmp(&dest[i], &attrs, sizeof(attrs))) {
        // found that this set of attributes already used, so re-use
        return i * sizeof(attrs);
      }
    }
    // append new attribute set to end
    uint32_t result = dest.size() * sizeof(attrs);
    dest.push_back(attrs);
    return result;
  }

  uint32_t sectionNameOffset(const DefinedAtom& atom) {
    // if section based on content, then no custom section name available
    if (atom.sectionChoice() == DefinedAtom::sectionBasedOnContent)
      return 0;
    StringRef name = atom.customSectionName();
    assert(!name.empty());
    // look to see if this section name was used by another atom
    for (auto &it : _sectionNames)
      if (name.equals(it.first))
        return it.second;
    // first use of this section name
    uint32_t result = this->getNameOffset(name);
    _sectionNames.push_back(std::make_pair(name, result));
    return result;
  }

  NativeAtomAttributesV1 computeAttributesV1(const DefinedAtom& atom) {
    NativeAtomAttributesV1 attrs;
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
    attrs.dynamicExport     = atom.dynamicExport();
    attrs.permissions       = atom.permissions();
    attrs.alias             = atom.isAlias();
    return attrs;
  }

  NativeAtomAttributesV1 computeAbsoluteAttributes(const AbsoluteAtom& atom) {
    NativeAtomAttributesV1 attrs;
    attrs.scope = atom.scope();
    return attrs;
  }

  // add references for this atom in a contiguous block in NCS_ReferencesArrayV2
  uint32_t getReferencesIndex(const DefinedAtom& atom, unsigned& refsCount) {
    size_t startRefSize = _referencesV2.size();
    uint32_t result = startRefSize;
    for (const Reference *ref : atom) {
      NativeReferenceIvarsV2 nref;
      nref.offsetInAtom = ref->offsetInAtom();
      nref.kindNamespace = (uint8_t)ref->kindNamespace();
      nref.kindArch = (uint8_t)ref->kindArch();
      nref.kindValue = ref->kindValue();
      nref.targetIndex = this->getTargetIndex(ref->target());
      nref.addend = ref->addend();
      _referencesV2.push_back(nref);
    }
    refsCount = _referencesV2.size() - startRefSize;
    return (refsCount == 0) ? 0 : result;
  }

  uint32_t getTargetIndex(const Atom* target) {
    if ( target == nullptr )
      return NativeReferenceIvarsV2::noTarget;
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
    for (auto &it : _targetsTableIndex) {
      const Atom* atom = it.first;
      uint32_t targetIndex = it.second;
      assert(targetIndex < maxTargetIndex);

      TargetToIndex::iterator pos = _definedAtomIndex.find(atom);
      if (pos != _definedAtomIndex.end()) {
        targetIndexes[targetIndex] = pos->second;
        continue;
      }
      uint32_t base = _definedAtomIvars.size();

      pos = _undefinedAtomIndex.find(atom);
      if (pos != _undefinedAtomIndex.end()) {
        targetIndexes[targetIndex] = pos->second + base;
        continue;
      }
      base += _undefinedAtomIndex.size();

      pos = _sharedLibraryAtomIndex.find(atom);
      if (pos != _sharedLibraryAtomIndex.end()) {
        targetIndexes[targetIndex] = pos->second + base;
        continue;
      }
      base += _sharedLibraryAtomIndex.size();

      pos = _absoluteAtomIndex.find(atom);
      assert(pos != _absoluteAtomIndex.end());
      targetIndexes[targetIndex] = pos->second + base;
    }
    // write table
    out.write((char*)&targetIndexes[0], maxTargetIndex * sizeof(uint32_t));
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
    for (auto &it : _addendsTableIndex) {
      Reference::Addend addend = it.first;
      uint32_t index = it.second;
      assert(index <= maxAddendIndex);
      addends[index-1] = addend;
    }
    // write table
    out.write((char*)&addends[0], maxAddendIndex*sizeof(Reference::Addend));
  }

  typedef std::vector<std::pair<StringRef, uint32_t>> NameToOffsetVector;

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
  std::vector<NativeReferenceIvarsV1>     _referencesV1;
  std::vector<NativeReferenceIvarsV2>     _referencesV2;
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

std::unique_ptr<Writer> createWriterNative(const LinkingContext &context) {
  return std::unique_ptr<Writer>(new native::Writer(context));
}
} // end namespace lld
