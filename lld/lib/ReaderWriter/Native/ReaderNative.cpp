//===- lib/ReaderWriter/Native/ReaderNative.cpp ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NativeFileFormat.h"
#include "lld/Core/Atom.h"
#include "lld/Core/Error.h"
#include "lld/Core/File.h"
#include "lld/Core/Reader.h"
#include "lld/Core/Simple.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <vector>

namespace lld {
namespace native {

// forward reference
class File;

//
// An object of this class is instantied for each NativeDefinedAtomIvarsV1
// struct in the NCS_DefinedAtomsV1 chunk.
//
class NativeDefinedAtomV1 : public DefinedAtom {
public:
      NativeDefinedAtomV1(const File& f,
                          const NativeDefinedAtomIvarsV1* ivarData)
        : _file(&f), _ivarData(ivarData) { }

  const lld::File& file() const override;

  uint64_t ordinal() const override;

  StringRef name() const override;

  uint64_t size() const override { return _ivarData->contentSize; }

  uint64_t sectionSize() const override { return _ivarData->sectionSize; }

  DefinedAtom::Scope scope() const override {
    return (DefinedAtom::Scope)(attributes().scope);
  }

  DefinedAtom::Interposable interposable() const override {
    return (DefinedAtom::Interposable)(attributes().interposable);
  }

  DefinedAtom::Merge merge() const override {
    return (DefinedAtom::Merge)(attributes().merge);
  }

  DefinedAtom::ContentType contentType() const override {
    const NativeAtomAttributesV1& attr = attributes();
    return (DefinedAtom::ContentType)(attr.contentType);
  }

  DefinedAtom::Alignment alignment() const override {
    return DefinedAtom::Alignment(attributes().align,
                                  attributes().alignModulus);
  }

  DefinedAtom::SectionChoice sectionChoice() const override {
    return (DefinedAtom::SectionChoice)(attributes().sectionChoice);
  }

  StringRef customSectionName() const override;

  DefinedAtom::DeadStripKind deadStrip() const override {
     return (DefinedAtom::DeadStripKind)(attributes().deadStrip);
  }

  DynamicExport dynamicExport() const override {
    return (DynamicExport)attributes().dynamicExport;
  }

  DefinedAtom::CodeModel codeModel() const override {
    return DefinedAtom::CodeModel(attributes().codeModel);
  }

  DefinedAtom::ContentPermissions permissions() const override {
     return (DefinedAtom::ContentPermissions)(attributes().permissions);
  }

  ArrayRef<uint8_t> rawContent() const override;

  reference_iterator begin() const override;

  reference_iterator end() const override;

  const Reference* derefIterator(const void*) const override;

  void incrementIterator(const void*& it) const override;

private:
  const NativeAtomAttributesV1& attributes() const;

  const File                     *_file;
  const NativeDefinedAtomIvarsV1 *_ivarData;
};



//
// An object of this class is instantied for each NativeUndefinedAtomIvarsV1
// struct in the NCS_UndefinedAtomsV1 chunk.
//
class NativeUndefinedAtomV1 : public UndefinedAtom {
public:
       NativeUndefinedAtomV1(const File& f,
                             const NativeUndefinedAtomIvarsV1* ivarData)
        : _file(&f), _ivarData(ivarData) { }

  const lld::File& file() const override;
  StringRef name() const override;

  CanBeNull canBeNull() const override {
    return (CanBeNull)(_ivarData->flags & 0x3);
  }

  const UndefinedAtom *fallback() const override;

private:
  const File                        *_file;
  const NativeUndefinedAtomIvarsV1  *_ivarData;
  mutable std::unique_ptr<const SimpleUndefinedAtom> _fallback;
};


//
// An object of this class is instantied for each NativeUndefinedAtomIvarsV1
// struct in the NCS_SharedLibraryAtomsV1 chunk.
//
class NativeSharedLibraryAtomV1 : public SharedLibraryAtom {
public:
       NativeSharedLibraryAtomV1(const File& f,
                             const NativeSharedLibraryAtomIvarsV1* ivarData)
        : _file(&f), _ivarData(ivarData) { }

  const lld::File& file() const override;
  StringRef name() const override;
  StringRef loadName() const override;

  bool canBeNullAtRuntime() const override {
    return (_ivarData->flags & 0x1);
  }

  Type type() const override {
    return (Type)_ivarData->type;
  }

  uint64_t size() const override {
    return _ivarData->size;
  }

private:
  const File                           *_file;
  const NativeSharedLibraryAtomIvarsV1 *_ivarData;
};


//
// An object of this class is instantied for each NativeAbsoluteAtomIvarsV1
// struct in the NCS_AbsoluteAtomsV1 chunk.
//
class NativeAbsoluteAtomV1 : public AbsoluteAtom {
public:
       NativeAbsoluteAtomV1(const File& f,
                             const NativeAbsoluteAtomIvarsV1* ivarData)
        : _file(&f), _ivarData(ivarData) { }

  const lld::File& file() const override;
  StringRef name() const override;
  Scope scope() const override {
    const NativeAtomAttributesV1& attr = absAttributes();
    return (Scope)(attr.scope);
  }
  uint64_t value() const override {
    return _ivarData->value;
  }

private:
  const NativeAtomAttributesV1& absAttributes() const;
  const File                      *_file;
  const NativeAbsoluteAtomIvarsV1 *_ivarData;
};


//
// An object of this class is instantied for each NativeReferenceIvarsV1
// struct in the NCS_ReferencesArrayV1 chunk.
//
class NativeReferenceV1 : public Reference {
public:
  NativeReferenceV1(const File &f, const NativeReferenceIvarsV1 *ivarData)
      : Reference((KindNamespace)ivarData->kindNamespace,
                  (KindArch)ivarData->kindArch, ivarData->kindValue),
        _file(&f), _ivarData(ivarData) {}

  uint64_t offsetInAtom() const override {
    return _ivarData->offsetInAtom;
  }

  const Atom* target() const override;
  Addend addend() const override;
  void setTarget(const Atom* newAtom) override;
  void setAddend(Addend a) override;

private:
  const File                    *_file;
  const NativeReferenceIvarsV1  *_ivarData;
};


//
// An object of this class is instantied for each NativeReferenceIvarsV1
// struct in the NCS_ReferencesArrayV1 chunk.
//
class NativeReferenceV2 : public Reference {
public:
  NativeReferenceV2(const File &f, const NativeReferenceIvarsV2 *ivarData)
      : Reference((KindNamespace)ivarData->kindNamespace,
                  (KindArch)ivarData->kindArch, ivarData->kindValue),
        _file(&f), _ivarData(ivarData) {}

  uint64_t offsetInAtom() const override {
    return _ivarData->offsetInAtom;
  }

  const Atom* target() const override;
  Addend addend() const override;
  void setTarget(const Atom* newAtom) override;
  void setAddend(Addend a) override;
  uint32_t tag() const override;

private:
  const File                    *_file;
  const NativeReferenceIvarsV2  *_ivarData;
};


//
// lld::File object for native llvm object file
//
class File : public lld::File {
public:
  File(std::unique_ptr<MemoryBuffer> mb)
      : lld::File(mb->getBufferIdentifier(), kindObject),
        _mb(std::move(mb)), // Reader now takes ownership of buffer
        _header(nullptr), _targetsTable(nullptr), _targetsTableCount(0),
        _strings(nullptr), _stringsMaxOffset(0), _addends(nullptr),
        _addendsMaxIndex(0), _contentStart(nullptr), _contentEnd(nullptr) {
    _header =
        reinterpret_cast<const NativeFileHeader *>(_mb->getBufferStart());
  }

  /// Parses a File object from a native object file.
  std::error_code doParse() override {
    const uint8_t *const base =
        reinterpret_cast<const uint8_t *>(_mb->getBufferStart());
    StringRef path(_mb->getBufferIdentifier());
    const NativeFileHeader *const header =
        reinterpret_cast<const NativeFileHeader *>(base);
    const NativeChunk *const chunks =
        reinterpret_cast<const NativeChunk *>(base + sizeof(NativeFileHeader));
    // make sure magic matches
    if (memcmp(header->magic, NATIVE_FILE_HEADER_MAGIC,
               sizeof(header->magic)) != 0)
      return make_error_code(NativeReaderError::unknown_file_format);

    // make sure mapped file contains all needed data
    const size_t fileSize = _mb->getBufferSize();
    if (header->fileSize > fileSize)
      return make_error_code(NativeReaderError::file_too_short);

    DEBUG_WITH_TYPE("ReaderNative",
                    llvm::dbgs() << " Native File Header:" << " fileSize="
                                 << header->fileSize << " chunkCount="
                                 << header->chunkCount << "\n");

    // process each chunk
    for (uint32_t i = 0; i < header->chunkCount; ++i) {
      std::error_code ec;
      const NativeChunk* chunk = &chunks[i];
      // sanity check chunk is within file
      if ( chunk->fileOffset > fileSize )
        return make_error_code(NativeReaderError::file_malformed);
      if ( (chunk->fileOffset + chunk->fileSize) > fileSize)
        return make_error_code(NativeReaderError::file_malformed);
      // process chunk, based on signature
      switch ( chunk->signature ) {
        case NCS_DefinedAtomsV1:
          ec = processDefinedAtomsV1(base, chunk);
          break;
        case NCS_AttributesArrayV1:
          ec = processAttributesV1(base, chunk);
          break;
        case NCS_UndefinedAtomsV1:
          ec = processUndefinedAtomsV1(base, chunk);
          break;
        case NCS_SharedLibraryAtomsV1:
          ec = processSharedLibraryAtomsV1(base, chunk);
          break;
        case NCS_AbsoluteAtomsV1:
          ec = processAbsoluteAtomsV1(base, chunk);
          break;
        case NCS_AbsoluteAttributesV1:
          ec = processAbsoluteAttributesV1(base, chunk);
          break;
        case NCS_ReferencesArrayV1:
          ec = processReferencesV1(base, chunk);
          break;
        case NCS_ReferencesArrayV2:
          ec = processReferencesV2(base, chunk);
          break;
        case NCS_TargetsTable:
          ec = processTargetsTable(base, chunk);
          break;
        case NCS_AddendsTable:
          ec = processAddendsTable(base, chunk);
          break;
        case NCS_Content:
          ec = processContent(base, chunk);
          break;
        case NCS_Strings:
          ec = processStrings(base, chunk);
          break;
        default:
          return make_error_code(NativeReaderError::unknown_chunk_type);
      }
      if ( ec ) {
        return ec;
      }
    }
    // TO DO: validate enough chunks were used

    DEBUG_WITH_TYPE("ReaderNative", {
      llvm::dbgs() << " ReaderNative DefinedAtoms:\n";
      for (const DefinedAtom *a : defined()) {
        llvm::dbgs() << llvm::format("    0x%09lX", a)
                     << ", name=" << a->name()
                     << ", size=" << a->size() << "\n";
        for (const Reference *r : *a) {
          llvm::dbgs() << "        offset="
                       << llvm::format("0x%03X", r->offsetInAtom())
                       << ", kind=" << r->kindValue()
                       << ", target=" << r->target() << "\n";
        }
      }
    });
    return make_error_code(NativeReaderError::success);
  }

  virtual ~File() {
    // _mb is automatically deleted because of std::unique_ptr<>

    // All other ivar pointers are pointers into the MemoryBuffer, except
    // the _definedAtoms array which was allocated to contain an array
    // of Atom objects.  The atoms have empty destructors, so it is ok
    // to just delete the memory.
    delete _referencesV1.arrayStart;
    delete _referencesV2.arrayStart;
    delete [] _targetsTable;
  }

  const atom_collection<DefinedAtom>&  defined() const override {
    return _definedAtoms;
  }
  const atom_collection<UndefinedAtom>& undefined() const override {
      return _undefinedAtoms;
  }
  const atom_collection<SharedLibraryAtom>& sharedLibrary() const override {
      return _sharedLibraryAtoms;
  }
  const atom_collection<AbsoluteAtom> &absolute() const override {
    return _absoluteAtoms;
  }

private:
  friend NativeDefinedAtomV1;
  friend NativeUndefinedAtomV1;
  friend NativeSharedLibraryAtomV1;
  friend NativeAbsoluteAtomV1;
  friend NativeReferenceV1;
  friend NativeReferenceV2;
  template <typename T> class AtomArray;

  // instantiate array of BASeT from IvarsT data in file
  template <typename BaseT, typename AtomT, typename IvarsT>
  std::error_code processAtoms(atom_collection_vector<BaseT> &result,
                               const uint8_t *base, const NativeChunk *chunk) {
    std::vector<const BaseT *> vec(chunk->elementCount);
    const size_t ivarElementSize = chunk->fileSize / chunk->elementCount;
    if (ivarElementSize != sizeof(IvarsT))
      return make_error_code(NativeReaderError::file_malformed);
    auto *ivar = reinterpret_cast<const IvarsT *>(base + chunk->fileOffset);
    for (size_t i = 0; i < chunk->elementCount; ++i)
      vec[i] = new (_alloc) AtomT(*this, ivar++);
    result._atoms = std::move(vec);
    return make_error_code(NativeReaderError::success);
  }

  // instantiate array of DefinedAtoms from v1 ivar data in file
  std::error_code processDefinedAtomsV1(const uint8_t *base,
                                        const NativeChunk *chunk) {
    return processAtoms<DefinedAtom, NativeDefinedAtomV1,
                        NativeDefinedAtomIvarsV1>(this->_definedAtoms, base,
                                                  chunk);
  }

  // set up pointers to attributes array
  std::error_code processAttributesV1(const uint8_t *base,
                                      const NativeChunk *chunk) {
    this->_attributes = base + chunk->fileOffset;
    this->_attributesMaxOffset = chunk->fileSize;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk AttributesV1:        "
                    << " count=" << chunk->elementCount
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(NativeReaderError::success);
  }

  // set up pointers to attributes array
  std::error_code processAbsoluteAttributesV1(const uint8_t *base,
                                              const NativeChunk *chunk) {
    this->_absAttributes = base + chunk->fileOffset;
    this->_absAbsoluteMaxOffset = chunk->fileSize;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk AbsoluteAttributesV1:        "
                    << " count=" << chunk->elementCount
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(NativeReaderError::success);
  }

  // instantiate array of UndefinedAtoms from v1 ivar data in file
  std::error_code processUndefinedAtomsV1(const uint8_t *base,
                                          const NativeChunk *chunk) {
    return processAtoms<UndefinedAtom, NativeUndefinedAtomV1,
                        NativeUndefinedAtomIvarsV1>(this->_undefinedAtoms, base,
                                                    chunk);
  }


  // instantiate array of ShareLibraryAtoms from v1 ivar data in file
  std::error_code processSharedLibraryAtomsV1(const uint8_t *base,
                                              const NativeChunk *chunk) {
    return processAtoms<SharedLibraryAtom, NativeSharedLibraryAtomV1,
                        NativeSharedLibraryAtomIvarsV1>(
        this->_sharedLibraryAtoms, base, chunk);
  }


   // instantiate array of AbsoluteAtoms from v1 ivar data in file
  std::error_code processAbsoluteAtomsV1(const uint8_t *base,
                                         const NativeChunk *chunk) {
    return processAtoms<AbsoluteAtom, NativeAbsoluteAtomV1,
                        NativeAbsoluteAtomIvarsV1>(this->_absoluteAtoms, base,
                                                   chunk);
  }

  template <class T, class U>
  std::error_code
  processReferences(const uint8_t *base, const NativeChunk *chunk,
                    uint8_t *&refsStart, uint8_t *&refsEnd) const {
    if (chunk->elementCount == 0)
      return make_error_code(NativeReaderError::success);
    size_t refsArraySize = chunk->elementCount * sizeof(T);
    refsStart = reinterpret_cast<uint8_t *>(
        operator new(refsArraySize, std::nothrow));
    if (refsStart == nullptr)
      return make_error_code(NativeReaderError::memory_error);
    const size_t ivarElementSize = chunk->fileSize / chunk->elementCount;
    if (ivarElementSize != sizeof(U))
      return make_error_code(NativeReaderError::file_malformed);
    refsEnd = refsStart + refsArraySize;
    const U* ivarData = reinterpret_cast<const U *>(base + chunk->fileOffset);
    for (uint8_t *s = refsStart; s != refsEnd; s += sizeof(T), ++ivarData) {
      T *atomAllocSpace = reinterpret_cast<T *>(s);
      new (atomAllocSpace) T(*this, ivarData);
    }
    return make_error_code(NativeReaderError::success);
  }

  // instantiate array of References from v1 ivar data in file
  std::error_code processReferencesV1(const uint8_t *base,
                                      const NativeChunk *chunk) {
    uint8_t *refsStart, *refsEnd;
    if (std::error_code ec =
            processReferences<NativeReferenceV1, NativeReferenceIvarsV1>(
                base, chunk, refsStart, refsEnd))
      return ec;
    this->_referencesV1.arrayStart = refsStart;
    this->_referencesV1.arrayEnd = refsEnd;
    this->_referencesV1.elementSize = sizeof(NativeReferenceV1);
    this->_referencesV1.elementCount = chunk->elementCount;
    DEBUG_WITH_TYPE("ReaderNative", {
      llvm::dbgs() << " chunk ReferencesV1:        "
                   << " count=" << chunk->elementCount
                   << " chunkSize=" << chunk->fileSize << "\n";
    });
    return make_error_code(NativeReaderError::success);
  }

  // instantiate array of References from v2 ivar data in file
  std::error_code processReferencesV2(const uint8_t *base,
                                      const NativeChunk *chunk) {
    uint8_t *refsStart, *refsEnd;
    if (std::error_code ec =
            processReferences<NativeReferenceV2, NativeReferenceIvarsV2>(
                base, chunk, refsStart, refsEnd))
      return ec;
    this->_referencesV2.arrayStart = refsStart;
    this->_referencesV2.arrayEnd = refsEnd;
    this->_referencesV2.elementSize = sizeof(NativeReferenceV2);
    this->_referencesV2.elementCount = chunk->elementCount;
    DEBUG_WITH_TYPE("ReaderNative", {
      llvm::dbgs() << " chunk ReferencesV2:        "
                   << " count=" << chunk->elementCount
                   << " chunkSize=" << chunk->fileSize << "\n";
    });
    return make_error_code(NativeReaderError::success);
  }

  // set up pointers to target table
  std::error_code processTargetsTable(const uint8_t *base,
                                      const NativeChunk *chunk) {
    const uint32_t* targetIndexes = reinterpret_cast<const uint32_t*>
                                                  (base + chunk->fileOffset);
    this->_targetsTableCount = chunk->elementCount;
    this->_targetsTable = new const Atom*[chunk->elementCount];
    for (uint32_t i=0; i < chunk->elementCount; ++i) {
      const uint32_t index = targetIndexes[i];
      if (index < _definedAtoms.size()) {
        this->_targetsTable[i] = _definedAtoms._atoms[index];
        continue;
      }
      const uint32_t undefIndex = index - _definedAtoms.size();
      if (undefIndex < _undefinedAtoms.size()) {
        this->_targetsTable[i] = _undefinedAtoms._atoms[index];
        continue;
      }
      const uint32_t slIndex = undefIndex - _undefinedAtoms.size();
      if (slIndex < _sharedLibraryAtoms.size()) {
        this->_targetsTable[i] = _sharedLibraryAtoms._atoms[slIndex];
        continue;
      }
      const uint32_t abIndex = slIndex - _sharedLibraryAtoms.size();
      if (abIndex < _absoluteAtoms.size()) {
        this->_targetsTable[i] = _absoluteAtoms._atoms[abIndex];
        continue;
      }
     return make_error_code(NativeReaderError::file_malformed);
    }
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk Targets Table:       "
                    << " count=" << chunk->elementCount
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(NativeReaderError::success);
  }


  // set up pointers to addend pool in file
  std::error_code processAddendsTable(const uint8_t *base,
                                      const NativeChunk *chunk) {
    this->_addends = reinterpret_cast<const Reference::Addend*>
                                                  (base + chunk->fileOffset);
    this->_addendsMaxIndex = chunk->elementCount;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk Addends:             "
                    << " count=" << chunk->elementCount
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(NativeReaderError::success);
  }

  // set up pointers to string pool in file
  std::error_code processStrings(const uint8_t *base,
                                 const NativeChunk *chunk) {
    this->_strings = reinterpret_cast<const char*>(base + chunk->fileOffset);
    this->_stringsMaxOffset = chunk->fileSize;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk Strings:             "
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(NativeReaderError::success);
  }

  // set up pointers to content area in file
  std::error_code processContent(const uint8_t *base,
                                 const NativeChunk *chunk) {
    this->_contentStart = base + chunk->fileOffset;
    this->_contentEnd = base + chunk->fileOffset + chunk->fileSize;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk content:             "
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(NativeReaderError::success);
  }

  StringRef string(uint32_t offset) const {
    assert(offset < _stringsMaxOffset);
    return StringRef(&_strings[offset]);
  }

  Reference::Addend addend(uint32_t index) const {
    if ( index == 0 )
      return 0; // addend index zero is used to mean "no addend"
    assert(index <= _addendsMaxIndex);
    return _addends[index-1]; // one-based indexing
  }

  const NativeAtomAttributesV1& attribute(uint32_t off) const {
    assert(off < _attributesMaxOffset);
    return *reinterpret_cast<const NativeAtomAttributesV1*>(_attributes + off);
  }

  const NativeAtomAttributesV1& absAttribute(uint32_t off) const {
    assert(off < _absAbsoluteMaxOffset);
    return *reinterpret_cast<const NativeAtomAttributesV1*>(_absAttributes + off);
  }

  const uint8_t* content(uint32_t offset, uint32_t size) const {
    const uint8_t* result = _contentStart + offset;
    assert((result+size) <= _contentEnd);
    return result;
  }

  const Reference* referenceByIndex(uintptr_t index) const {
    if (index < _referencesV1.elementCount) {
      return reinterpret_cast<const NativeReferenceV1*>(
          _referencesV1.arrayStart + index * _referencesV1.elementSize);
    }
    assert(index < _referencesV2.elementCount);
    return reinterpret_cast<const NativeReferenceV2*>(
        _referencesV2.arrayStart + index * _referencesV2.elementSize);
  }

  const Atom* targetV1(uint16_t index) const {
    if ( index == NativeReferenceIvarsV1::noTarget )
      return nullptr;
    assert(index < _targetsTableCount);
    return _targetsTable[index];
  }

  void setTargetV1(uint16_t index, const Atom* newAtom) const {
    assert(index != NativeReferenceIvarsV1::noTarget);
    assert(index > _targetsTableCount);
    _targetsTable[index] = newAtom;
  }

  const Atom* targetV2(uint32_t index) const {
    if (index == NativeReferenceIvarsV2::noTarget)
      return nullptr;
    assert(index < _targetsTableCount);
    return _targetsTable[index];
  }

  void setTargetV2(uint32_t index, const Atom* newAtom) const {
    assert(index != NativeReferenceIvarsV2::noTarget);
    assert(index > _targetsTableCount);
    _targetsTable[index] = newAtom;
  }

  struct IvarArray {
                      IvarArray() :
                        arrayStart(nullptr),
                        arrayEnd(nullptr),
                        elementSize(0),
                        elementCount(0) { }

    const uint8_t*     arrayStart;
    const uint8_t*     arrayEnd;
    uint32_t           elementSize;
    uint32_t           elementCount;
  };

  std::unique_ptr<MemoryBuffer>   _mb;
  const NativeFileHeader*         _header;
  atom_collection_vector<DefinedAtom> _definedAtoms;
  atom_collection_vector<UndefinedAtom> _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom> _absoluteAtoms;
  const uint8_t*                  _absAttributes;
  uint32_t                        _absAbsoluteMaxOffset;
  const uint8_t*                  _attributes;
  uint32_t                        _attributesMaxOffset;
  IvarArray                       _referencesV1;
  IvarArray                       _referencesV2;
  const Atom**                    _targetsTable;
  uint32_t                        _targetsTableCount;
  const char*                     _strings;
  uint32_t                        _stringsMaxOffset;
  const Reference::Addend*        _addends;
  uint32_t                        _addendsMaxIndex;
  const uint8_t                  *_contentStart;
  const uint8_t                  *_contentEnd;
  llvm::BumpPtrAllocator _alloc;
};

inline const lld::File &NativeDefinedAtomV1::file() const {
  return *_file;
}

inline uint64_t NativeDefinedAtomV1::ordinal() const {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(_ivarData);
  auto *start = reinterpret_cast<const NativeDefinedAtomV1 *>(
      _file->_definedAtoms._atoms[0]);
  const uint8_t *startp = reinterpret_cast<const uint8_t *>(start->_ivarData);
  return p - startp;
}

inline StringRef NativeDefinedAtomV1::name() const {
  return _file->string(_ivarData->nameOffset);
}

inline const NativeAtomAttributesV1& NativeDefinedAtomV1::attributes() const {
  return _file->attribute(_ivarData->attributesOffset);
}

inline ArrayRef<uint8_t> NativeDefinedAtomV1::rawContent() const {
  if (!occupiesDiskSpace())
    return ArrayRef<uint8_t>();
  const uint8_t* p = _file->content(_ivarData->contentOffset,
                                    _ivarData->contentSize);
  return ArrayRef<uint8_t>(p, _ivarData->contentSize);
}

inline StringRef NativeDefinedAtomV1::customSectionName() const {
  uint32_t offset = attributes().sectionNameOffset;
  return _file->string(offset);
}

DefinedAtom::reference_iterator NativeDefinedAtomV1::begin() const {
  uintptr_t index = _ivarData->referencesStartIndex;
  const void* it = reinterpret_cast<const void*>(index);
  return reference_iterator(*this, it);
}

DefinedAtom::reference_iterator NativeDefinedAtomV1::end() const {
  uintptr_t index = _ivarData->referencesStartIndex+_ivarData->referencesCount;
  const void* it = reinterpret_cast<const void*>(index);
  return reference_iterator(*this, it);
}

const Reference* NativeDefinedAtomV1::derefIterator(const void* it) const {
  uintptr_t index = reinterpret_cast<uintptr_t>(it);
  return _file->referenceByIndex(index);
}

void NativeDefinedAtomV1::incrementIterator(const void*& it) const {
  uintptr_t index = reinterpret_cast<uintptr_t>(it);
  ++index;
  it = reinterpret_cast<const void*>(index);
}

inline const lld::File& NativeUndefinedAtomV1::file() const {
  return *_file;
}

inline StringRef NativeUndefinedAtomV1::name() const {
  return _file->string(_ivarData->nameOffset);
}

inline const UndefinedAtom *NativeUndefinedAtomV1::fallback() const {
  if (!_ivarData->fallbackNameOffset)
    return nullptr;
  if (!_fallback)
    _fallback.reset(new SimpleUndefinedAtom(
        *_file, _file->string(_ivarData->fallbackNameOffset)));
  return _fallback.get();
}

inline const lld::File& NativeSharedLibraryAtomV1::file() const {
  return *_file;
}

inline StringRef NativeSharedLibraryAtomV1::name() const {
  return _file->string(_ivarData->nameOffset);
}

inline StringRef NativeSharedLibraryAtomV1::loadName() const {
  return _file->string(_ivarData->loadNameOffset);
}



inline const lld::File& NativeAbsoluteAtomV1::file() const {
  return *_file;
}

inline StringRef NativeAbsoluteAtomV1::name() const {
  return _file->string(_ivarData->nameOffset);
}

inline const NativeAtomAttributesV1& NativeAbsoluteAtomV1::absAttributes() const {
  return _file->absAttribute(_ivarData->attributesOffset);
}

inline const Atom* NativeReferenceV1::target() const {
  return _file->targetV1(_ivarData->targetIndex);
}

inline Reference::Addend NativeReferenceV1::addend() const {
  return _file->addend(_ivarData->addendIndex);
}

inline void NativeReferenceV1::setTarget(const Atom* newAtom) {
  return _file->setTargetV1(_ivarData->targetIndex, newAtom);
}

inline void NativeReferenceV1::setAddend(Addend a) {
  // Do nothing if addend value is not being changed.
  if (addend() == a)
    return;
  llvm_unreachable("setAddend() not supported");
}

inline const Atom* NativeReferenceV2::target() const {
  return _file->targetV2(_ivarData->targetIndex);
}

inline Reference::Addend NativeReferenceV2::addend() const {
  return _ivarData->addend;
}

inline void NativeReferenceV2::setTarget(const Atom* newAtom) {
  return _file->setTargetV2(_ivarData->targetIndex, newAtom);
}

inline void NativeReferenceV2::setAddend(Addend a) {
  // Do nothing if addend value is not being changed.
  if (addend() == a)
    return;
  llvm_unreachable("setAddend() not supported");
}

uint32_t NativeReferenceV2::tag() const { return _ivarData->tag; }

} // end namespace native

namespace {

class NativeReader : public Reader {
public:
  bool canParse(file_magic magic, const MemoryBuffer &mb) const override {
    const NativeFileHeader *const header =
        reinterpret_cast<const NativeFileHeader *>(mb.getBufferStart());
    return (memcmp(header->magic, NATIVE_FILE_HEADER_MAGIC,
                   sizeof(header->magic)) == 0);
  }

  virtual std::error_code
  loadFile(std::unique_ptr<MemoryBuffer> mb, const class Registry &,
           std::vector<std::unique_ptr<File>> &result) const override {
    auto *file = new lld::native::File(std::move(mb));
    result.push_back(std::unique_ptr<File>(file));
    return std::error_code();
  }
};

}

void Registry::addSupportNativeObjects() {
  add(std::unique_ptr<Reader>(new NativeReader()));
}

} // end namespace lld
