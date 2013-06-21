//===- lib/ReaderWriter/Native/ReaderNative.cpp ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/Reader.h"

#include "lld/Core/Atom.h"
#include "lld/Core/Error.h"
#include "lld/Core/File.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "NativeFileFormat.h"

#include <vector>
#include <memory>

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

  virtual const lld::File& file() const;

  virtual uint64_t ordinal() const;

  virtual StringRef name() const;

  virtual uint64_t size() const {
    return _ivarData->contentSize;
  }

  virtual DefinedAtom::Scope scope() const {
    return (DefinedAtom::Scope)(attributes().scope);
  }

  virtual DefinedAtom::Interposable interposable() const {
    return (DefinedAtom::Interposable)(attributes().interposable);
  }

  virtual DefinedAtom::Merge merge() const {
    return (DefinedAtom::Merge)(attributes().merge);
  }

  virtual DefinedAtom::ContentType contentType() const {
    const NativeAtomAttributesV1& attr = attributes();
    return (DefinedAtom::ContentType)(attr.contentType);
  }

  virtual DefinedAtom::Alignment alignment() const {
    return DefinedAtom::Alignment(attributes().align2, attributes().alignModulus);
  }

  virtual DefinedAtom::SectionChoice sectionChoice() const {
    return (DefinedAtom::SectionChoice)(
        attributes().sectionChoiceAndPosition >> 4);
  }

  virtual StringRef customSectionName() const;

  virtual SectionPosition sectionPosition() const {
     return (DefinedAtom::SectionPosition)(
        attributes().sectionChoiceAndPosition & 0xF);
  }

  virtual DefinedAtom::DeadStripKind deadStrip() const {
     return (DefinedAtom::DeadStripKind)(attributes().deadStrip);
  }

  virtual DefinedAtom::ContentPermissions permissions() const {
     return (DefinedAtom::ContentPermissions)(attributes().permissions);
  }

  virtual bool isAlias() const {
     return (attributes().alias != 0);
  }

  virtual ArrayRef<uint8_t> rawContent() const;

  virtual reference_iterator begin() const;

  virtual reference_iterator end() const;

  virtual const Reference* derefIterator(const void*) const;

  virtual void incrementIterator(const void*& it) const;

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

  virtual const lld::File& file() const;
  virtual StringRef name() const;

  virtual CanBeNull canBeNull() const {
    return (CanBeNull)(_ivarData->flags & 0x3);
  }


private:
  const File                        *_file;
  const NativeUndefinedAtomIvarsV1  *_ivarData;
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

  virtual const lld::File& file() const;
  virtual StringRef name() const;
  virtual StringRef loadName() const;

  virtual bool canBeNullAtRuntime() const {
    return (_ivarData->flags & 0x1);
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

  virtual const lld::File& file() const;
  virtual StringRef name() const;
  virtual Scope scope() const {
    const NativeAtomAttributesV1& attr = absAttributes();
    return (Scope)(attr.scope);
  }
  virtual uint64_t value() const {
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
  NativeReferenceV1(const File& f, const NativeReferenceIvarsV1* ivarData)
      : _file(&f), _ivarData(ivarData) {
    setKind(ivarData->kind);
  }

  virtual uint64_t offsetInAtom() const {
    return _ivarData->offsetInAtom;
  }

  virtual const Atom* target() const;
  virtual Addend addend() const;
  virtual void setTarget(const Atom* newAtom);
  virtual void setAddend(Addend a);

private:
  // Used in rare cases when Reference is modified,
  // since ivar data is mapped read-only.
  void cloneIvarData() {
    // TODO: do nothing on second call
   NativeReferenceIvarsV1* niv = reinterpret_cast<NativeReferenceIvarsV1*>
                                (operator new(sizeof(NativeReferenceIvarsV1),
                                                                std::nothrow));
    memcpy(niv, _ivarData, sizeof(NativeReferenceIvarsV1));
  }

  const File                    *_file;
  const NativeReferenceIvarsV1  *_ivarData;
};



//
// lld::File object for native llvm object file
//
class File : public lld::File {
public:

  /// Instantiates a File object from a native object file.  Ownership
  /// of the MemoryBuffer is transfered to the resulting File object.
  static error_code make(
      const TargetInfo &ti, std::unique_ptr<llvm::MemoryBuffer> &mb,
      StringRef path, std::vector<std::unique_ptr<lld::File> > &result) {
    const uint8_t *const base =
        reinterpret_cast<const uint8_t *>(mb->getBufferStart());
    const NativeFileHeader* const header =
                       reinterpret_cast<const NativeFileHeader*>(base);
    const NativeChunk *const chunks =
      reinterpret_cast<const NativeChunk*>(base + sizeof(NativeFileHeader));
    // make sure magic matches
    if ( memcmp(header->magic, NATIVE_FILE_HEADER_MAGIC, 16) != 0 )
      return make_error_code(native_reader_error::unknown_file_format);

    // make sure mapped file contains all needed data
    const size_t fileSize = mb->getBufferSize();
    if ( header->fileSize > fileSize )
      return make_error_code(native_reader_error::file_too_short);

    DEBUG_WITH_TYPE("ReaderNative",
                    llvm::dbgs() << " Native File Header:" << " fileSize="
                                 << header->fileSize << " chunkCount="
                                 << header->chunkCount << "\n");

    // instantiate NativeFile object and add values to it as found
    std::unique_ptr<File> file(new File(ti, std::move(mb), path));

    // process each chunk
    for (uint32_t i = 0; i < header->chunkCount; ++i) {
      error_code ec;
      const NativeChunk* chunk = &chunks[i];
      // sanity check chunk is within file
      if ( chunk->fileOffset > fileSize )
        return make_error_code(native_reader_error::file_malformed);
      if ( (chunk->fileOffset + chunk->fileSize) > fileSize)
        return make_error_code(native_reader_error::file_malformed);
      // process chunk, based on signature
      switch ( chunk->signature ) {
        case NCS_DefinedAtomsV1:
          ec = file->processDefinedAtomsV1(base, chunk);
          break;
        case NCS_AttributesArrayV1:
          ec = file->processAttributesV1(base, chunk);
          break;
        case NCS_UndefinedAtomsV1:
          ec = file->processUndefinedAtomsV1(base, chunk);
          break;
        case NCS_SharedLibraryAtomsV1:
          ec = file->processSharedLibraryAtomsV1(base, chunk);
          break;
        case NCS_AbsoluteAtomsV1:
          ec = file->processAbsoluteAtomsV1(base, chunk);
          break;
        case NCS_AbsoluteAttributesV1:
          ec = file->processAbsoluteAttributesV1(base, chunk);
          break;
        case NCS_ReferencesArrayV1:
          ec = file->processReferencesV1(base, chunk);
          break;
        case NCS_TargetsTable:
          ec = file->processTargetsTable(base, chunk);
          break;
        case NCS_AddendsTable:
          ec = file->processAddendsTable(base, chunk);
          break;
        case NCS_Content:
          ec = file->processContent(base, chunk);
          break;
        case NCS_Strings:
          ec = file->processStrings(base, chunk);
          break;
        default:
          return make_error_code(native_reader_error::unknown_chunk_type);
      }
      if ( ec ) {
        return ec;
      }
    }
    // TO DO: validate enough chunks were used

    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                  << " ReaderNative DefinedAtoms:\n");
    for (const DefinedAtom *a : file->defined() ) {
      DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << llvm::format("    0x%09lX", a)
                    << ", name=" << a->name()
                    << ", size=" << a->size()
                    << "\n");
      for (const Reference *r : *a ) {
        (void)r;
        DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << "        offset="
                    << llvm::format("0x%03X", r->offsetInAtom())
                    << ", kind=" << r->kind()
                    << ", target=" << r->target()
                    << "\n");
      }
    }

    result.push_back(std::move(file));
    return make_error_code(native_reader_error::success);
  }

  virtual ~File() {
    // _buffer is automatically deleted because of OwningPtr<>

    // All other ivar pointers are pointers into the MemoryBuffer, except
    // the _definedAtoms array which was allocated to contain an array
    // of Atom objects.  The atoms have empty destructors, so it is ok
    // to just delete the memory.
    delete _definedAtoms._arrayStart;
    delete _undefinedAtoms._arrayStart;
    delete _sharedLibraryAtoms._arrayStart;
    delete _absoluteAtoms._arrayStart;
    delete _references.arrayStart;
    delete [] _targetsTable;
  }

  virtual const atom_collection<DefinedAtom>&  defined() const {
    return _definedAtoms;
  }
  virtual const atom_collection<UndefinedAtom>& undefined() const {
      return _undefinedAtoms;
  }
  virtual const atom_collection<SharedLibraryAtom>& sharedLibrary() const {
      return _sharedLibraryAtoms;
  }
  virtual const atom_collection<AbsoluteAtom> &absolute() const {
    return _absoluteAtoms;
  }
  virtual const TargetInfo &getTargetInfo() const { return _targetInfo; }

private:
  friend NativeDefinedAtomV1;
  friend NativeUndefinedAtomV1;
  friend NativeSharedLibraryAtomV1;
  friend NativeAbsoluteAtomV1;
  friend NativeReferenceV1;

  // instantiate array of DefinedAtoms from v1 ivar data in file
  error_code processDefinedAtomsV1(const uint8_t *base,
                                   const NativeChunk *chunk) {
    const size_t atomSize = sizeof(NativeDefinedAtomV1);
    size_t atomsArraySize = chunk->elementCount * atomSize;
    uint8_t* atomsStart = reinterpret_cast<uint8_t*>
                                (operator new(atomsArraySize, std::nothrow));
    if (atomsStart == nullptr)
      return make_error_code(native_reader_error::memory_error);
    const size_t ivarElementSize = chunk->fileSize
                                          / chunk->elementCount;
    if ( ivarElementSize != sizeof(NativeDefinedAtomIvarsV1) )
      return make_error_code(native_reader_error::file_malformed);
    uint8_t* atomsEnd = atomsStart + atomsArraySize;
    const NativeDefinedAtomIvarsV1* ivarData =
                             reinterpret_cast<const NativeDefinedAtomIvarsV1*>
                                                  (base + chunk->fileOffset);
    for(uint8_t* s = atomsStart; s != atomsEnd; s += atomSize) {
      NativeDefinedAtomV1* atomAllocSpace =
                  reinterpret_cast<NativeDefinedAtomV1*>(s);
      new (atomAllocSpace) NativeDefinedAtomV1(*this, ivarData);
      ++ivarData;
    }
    this->_definedAtoms._arrayStart = atomsStart;
    this->_definedAtoms._arrayEnd = atomsEnd;
    this->_definedAtoms._elementSize = atomSize;
    this->_definedAtoms._elementCount = chunk->elementCount;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk DefinedAtomsV1:      "
                    << " count=" << chunk->elementCount
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(native_reader_error::success);
  }



  // set up pointers to attributes array
  error_code processAttributesV1(const uint8_t *base,
                                 const NativeChunk *chunk) {
    this->_attributes = base + chunk->fileOffset;
    this->_attributesMaxOffset = chunk->fileSize;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk AttributesV1:        "
                    << " count=" << chunk->elementCount
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(native_reader_error::success);
  }

  // set up pointers to attributes array
  error_code processAbsoluteAttributesV1(const uint8_t *base,
                                 const NativeChunk *chunk) {
    this->_absAttributes = base + chunk->fileOffset;
    this->_absAbsoluteMaxOffset = chunk->fileSize;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk AbsoluteAttributesV1:        "
                    << " count=" << chunk->elementCount
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(native_reader_error::success);
  }

  // instantiate array of UndefinedAtoms from v1 ivar data in file
  error_code processUndefinedAtomsV1(const uint8_t *base,
                                     const NativeChunk *chunk) {
    const size_t atomSize = sizeof(NativeUndefinedAtomV1);
    size_t atomsArraySize = chunk->elementCount * atomSize;
    uint8_t* atomsStart = reinterpret_cast<uint8_t*>
                                (operator new(atomsArraySize, std::nothrow));
    if (atomsStart == nullptr)
      return make_error_code(native_reader_error::memory_error);
    const size_t ivarElementSize = chunk->fileSize
                                          / chunk->elementCount;
    if ( ivarElementSize != sizeof(NativeUndefinedAtomIvarsV1) )
      return make_error_code(native_reader_error::file_malformed);
    uint8_t* atomsEnd = atomsStart + atomsArraySize;
    const NativeUndefinedAtomIvarsV1* ivarData =
                            reinterpret_cast<const NativeUndefinedAtomIvarsV1*>
                                                  (base + chunk->fileOffset);
    for(uint8_t* s = atomsStart; s != atomsEnd; s += atomSize) {
      NativeUndefinedAtomV1* atomAllocSpace =
                  reinterpret_cast<NativeUndefinedAtomV1*>(s);
      new (atomAllocSpace) NativeUndefinedAtomV1(*this, ivarData);
      ++ivarData;
    }
    this->_undefinedAtoms._arrayStart = atomsStart;
    this->_undefinedAtoms._arrayEnd = atomsEnd;
    this->_undefinedAtoms._elementSize = atomSize;
    this->_undefinedAtoms._elementCount = chunk->elementCount;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk UndefinedAtomsV1:"
                    << " count=" << chunk->elementCount
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(native_reader_error::success);
  }


  // instantiate array of ShareLibraryAtoms from v1 ivar data in file
  error_code processSharedLibraryAtomsV1(const uint8_t *base,
                                         const NativeChunk *chunk) {
    const size_t atomSize = sizeof(NativeSharedLibraryAtomV1);
    size_t atomsArraySize = chunk->elementCount * atomSize;
    uint8_t* atomsStart = reinterpret_cast<uint8_t*>
                                (operator new(atomsArraySize, std::nothrow));
    if (atomsStart == nullptr)
      return make_error_code(native_reader_error::memory_error);
    const size_t ivarElementSize = chunk->fileSize
                                          / chunk->elementCount;
    if ( ivarElementSize != sizeof(NativeSharedLibraryAtomIvarsV1) )
      return make_error_code(native_reader_error::file_malformed);
    uint8_t* atomsEnd = atomsStart + atomsArraySize;
    const NativeSharedLibraryAtomIvarsV1* ivarData =
                      reinterpret_cast<const NativeSharedLibraryAtomIvarsV1*>
                                                  (base + chunk->fileOffset);
    for(uint8_t* s = atomsStart; s != atomsEnd; s += atomSize) {
      NativeSharedLibraryAtomV1* atomAllocSpace =
                  reinterpret_cast<NativeSharedLibraryAtomV1*>(s);
      new (atomAllocSpace) NativeSharedLibraryAtomV1(*this, ivarData);
      ++ivarData;
    }
    this->_sharedLibraryAtoms._arrayStart = atomsStart;
    this->_sharedLibraryAtoms._arrayEnd = atomsEnd;
    this->_sharedLibraryAtoms._elementSize = atomSize;
    this->_sharedLibraryAtoms._elementCount = chunk->elementCount;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk SharedLibraryAtomsV1:"
                    << " count=" << chunk->elementCount
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(native_reader_error::success);
  }


   // instantiate array of AbsoluteAtoms from v1 ivar data in file
  error_code processAbsoluteAtomsV1(const uint8_t *base,
                                    const NativeChunk *chunk) {
    const size_t atomSize = sizeof(NativeAbsoluteAtomV1);
    size_t atomsArraySize = chunk->elementCount * atomSize;
    uint8_t* atomsStart = reinterpret_cast<uint8_t*>
                                (operator new(atomsArraySize, std::nothrow));
    if (atomsStart == nullptr)
      return make_error_code(native_reader_error::memory_error);
    const size_t ivarElementSize = chunk->fileSize
                                          / chunk->elementCount;
    if ( ivarElementSize != sizeof(NativeAbsoluteAtomIvarsV1) )
      return make_error_code(native_reader_error::file_malformed);
    uint8_t* atomsEnd = atomsStart + atomsArraySize;
    const NativeAbsoluteAtomIvarsV1* ivarData =
                      reinterpret_cast<const NativeAbsoluteAtomIvarsV1*>
                                                  (base + chunk->fileOffset);
    for(uint8_t* s = atomsStart; s != atomsEnd; s += atomSize) {
      NativeAbsoluteAtomV1* atomAllocSpace =
                  reinterpret_cast<NativeAbsoluteAtomV1*>(s);
      new (atomAllocSpace) NativeAbsoluteAtomV1(*this, ivarData);
      ++ivarData;
    }
    this->_absoluteAtoms._arrayStart = atomsStart;
    this->_absoluteAtoms._arrayEnd = atomsEnd;
    this->_absoluteAtoms._elementSize = atomSize;
    this->_absoluteAtoms._elementCount = chunk->elementCount;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk AbsoluteAtomsV1:     "
                    << " count=" << chunk->elementCount
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(native_reader_error::success);
  }




  // instantiate array of Referemces from v1 ivar data in file
  error_code processReferencesV1(const uint8_t *base,
                                 const NativeChunk *chunk) {
    if ( chunk->elementCount == 0 )
      return make_error_code(native_reader_error::success);
    const size_t refSize = sizeof(NativeReferenceV1);
    size_t refsArraySize = chunk->elementCount * refSize;
    uint8_t* refsStart = reinterpret_cast<uint8_t*>
                                (operator new(refsArraySize, std::nothrow));
    if (refsStart == nullptr)
      return make_error_code(native_reader_error::memory_error);
    const size_t ivarElementSize = chunk->fileSize
                                          / chunk->elementCount;
    if ( ivarElementSize != sizeof(NativeReferenceIvarsV1) )
      return make_error_code(native_reader_error::file_malformed);
    uint8_t* refsEnd = refsStart + refsArraySize;
    const NativeReferenceIvarsV1* ivarData =
                             reinterpret_cast<const NativeReferenceIvarsV1*>
                                                  (base + chunk->fileOffset);
    for(uint8_t* s = refsStart; s != refsEnd; s += refSize) {
      NativeReferenceV1* atomAllocSpace =
                  reinterpret_cast<NativeReferenceV1*>(s);
      new (atomAllocSpace) NativeReferenceV1(*this, ivarData);
      ++ivarData;
    }
    this->_references.arrayStart = refsStart;
    this->_references.arrayEnd = refsEnd;
    this->_references.elementSize = refSize;
    this->_references.elementCount = chunk->elementCount;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk ReferencesV1:        "
                    << " count=" << chunk->elementCount
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(native_reader_error::success);
  }

  // set up pointers to target table
  error_code processTargetsTable(const uint8_t *base,
                                 const NativeChunk *chunk) {
    const uint32_t* targetIndexes = reinterpret_cast<const uint32_t*>
                                                  (base + chunk->fileOffset);
    this->_targetsTableCount = chunk->elementCount;
    this->_targetsTable = new const Atom*[chunk->elementCount];
    for (uint32_t i=0; i < chunk->elementCount; ++i) {
      const uint32_t index = targetIndexes[i];
      if ( index < _definedAtoms._elementCount ) {
        const uint8_t* p = _definedAtoms._arrayStart
                                    + index * _definedAtoms._elementSize;
        this->_targetsTable[i] = reinterpret_cast<const DefinedAtom*>(p);
        continue;
      }
      const uint32_t undefIndex = index - _definedAtoms._elementCount;
      if ( undefIndex < _undefinedAtoms._elementCount ) {
        const uint8_t* p = _undefinedAtoms._arrayStart
                                    + undefIndex * _undefinedAtoms._elementSize;
        this->_targetsTable[i] = reinterpret_cast<const UndefinedAtom*>(p);
        continue;
      }
      const uint32_t slIndex = index - _definedAtoms._elementCount
                                     - _undefinedAtoms._elementCount;
      if ( slIndex < _sharedLibraryAtoms._elementCount ) {
        const uint8_t* p = _sharedLibraryAtoms._arrayStart
                                  + slIndex * _sharedLibraryAtoms._elementSize;
        this->_targetsTable[i] = reinterpret_cast<const SharedLibraryAtom*>(p);
        continue;
      }
      const uint32_t abIndex = index - _definedAtoms._elementCount
                                     - _undefinedAtoms._elementCount
                                     - _sharedLibraryAtoms._elementCount;
      if ( abIndex < _absoluteAtoms._elementCount ) {
        const uint8_t* p = _absoluteAtoms._arrayStart
                                  + slIndex * _absoluteAtoms._elementSize;
        this->_targetsTable[i] = reinterpret_cast<const AbsoluteAtom*>(p);
        continue;
      }
     return make_error_code(native_reader_error::file_malformed);
    }
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk Targets Table:       "
                    << " count=" << chunk->elementCount
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(native_reader_error::success);
  }


  // set up pointers to addend pool in file
  error_code processAddendsTable(const uint8_t *base,
                                 const NativeChunk *chunk) {
    this->_addends = reinterpret_cast<const Reference::Addend*>
                                                  (base + chunk->fileOffset);
    this->_addendsMaxIndex = chunk->elementCount;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk Addends:             "
                    << " count=" << chunk->elementCount
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(native_reader_error::success);
  }

  // set up pointers to string pool in file
  error_code processStrings(const uint8_t *base,
                            const NativeChunk *chunk) {
    this->_strings = reinterpret_cast<const char*>(base + chunk->fileOffset);
    this->_stringsMaxOffset = chunk->fileSize;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk Strings:             "
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(native_reader_error::success);
  }

  // set up pointers to content area in file
  error_code processContent(const uint8_t *base,
                            const NativeChunk *chunk) {
    this->_contentStart = base + chunk->fileOffset;
    this->_contentEnd = base + chunk->fileOffset + chunk->fileSize;
    DEBUG_WITH_TYPE("ReaderNative", llvm::dbgs()
                    << " chunk content:             "
                    << " chunkSize=" << chunk->fileSize
                    << "\n");
    return make_error_code(native_reader_error::success);
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
    assert(index < _references.elementCount);
    const uint8_t* p = _references.arrayStart + index * _references.elementSize;
    return reinterpret_cast<const NativeReferenceV1*>(p);
  }

  const Atom* target(uint16_t index) const {
    if ( index == NativeReferenceIvarsV1::noTarget )
      return nullptr;
    assert(index < _targetsTableCount);
    return _targetsTable[index];
  }

  void setTarget(uint16_t index, const Atom* newAtom) const {
    assert(index != NativeReferenceIvarsV1::noTarget);
    assert(index > _targetsTableCount);
    _targetsTable[index] = newAtom;
  }

  // private constructor, only called by make()
  File(const TargetInfo &ti, std::unique_ptr<llvm::MemoryBuffer> mb,
       StringRef path)
      : lld::File(path, kindObject),
        _buffer(std::move(mb)), // Reader now takes ownership of buffer
        _header(nullptr), _targetsTable(nullptr), _targetsTableCount(0),
        _strings(nullptr), _stringsMaxOffset(0), _addends(nullptr),
        _addendsMaxIndex(0), _contentStart(nullptr), _contentEnd(nullptr),
        _targetInfo(ti) {
    _header =
        reinterpret_cast<const NativeFileHeader *>(_buffer->getBufferStart());
  }

  template <typename T>
  class AtomArray : public File::atom_collection<T> {
  public:
     AtomArray() : _arrayStart(nullptr), _arrayEnd(nullptr),
                   _elementSize(0), _elementCount(0) { }

    virtual atom_iterator<T> begin() const {
      return atom_iterator<T>(*this, reinterpret_cast<const void*>(_arrayStart));
    }
    virtual atom_iterator<T> end() const{
      return atom_iterator<T>(*this, reinterpret_cast<const void*>(_arrayEnd));
    }
    virtual const T* deref(const void* it) const {
      return reinterpret_cast<const T*>(it);
    }
    virtual void next(const void*& it) const {
      const uint8_t* p = reinterpret_cast<const uint8_t*>(it);
      p += _elementSize;
      it = reinterpret_cast<const void*>(p);
    }
    virtual uint64_t size() const { return _elementCount; }
    const uint8_t *_arrayStart;
    const uint8_t *_arrayEnd;
    uint32_t           _elementSize;
    uint32_t           _elementCount;
  };

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


  std::unique_ptr<llvm::MemoryBuffer> _buffer;
  const NativeFileHeader*         _header;
  AtomArray<DefinedAtom>          _definedAtoms;
  AtomArray<UndefinedAtom>        _undefinedAtoms;
  AtomArray<SharedLibraryAtom>    _sharedLibraryAtoms;
  AtomArray<AbsoluteAtom>         _absoluteAtoms;
  const uint8_t*                  _absAttributes;
  uint32_t                        _absAbsoluteMaxOffset;
  const uint8_t*                  _attributes;
  uint32_t                        _attributesMaxOffset;
  IvarArray                       _references;
  const Atom**                    _targetsTable;
  uint32_t                        _targetsTableCount;
  const char*                     _strings;
  uint32_t                        _stringsMaxOffset;
  const Reference::Addend*        _addends;
  uint32_t _addendsMaxIndex;
  const uint8_t *_contentStart;
  const uint8_t *_contentEnd;
  const TargetInfo &_targetInfo;
};

inline const lld::File &NativeDefinedAtomV1::file() const {
  return *_file;
}

inline uint64_t NativeDefinedAtomV1:: ordinal() const {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(_ivarData);
  return p - _file->_definedAtoms._arrayStart;
}

inline StringRef NativeDefinedAtomV1::name() const {
  return _file->string(_ivarData->nameOffset);
}

inline const NativeAtomAttributesV1& NativeDefinedAtomV1::attributes() const {
  return _file->attribute(_ivarData->attributesOffset);
}

inline ArrayRef<uint8_t> NativeDefinedAtomV1::rawContent() const {
  if (( this->contentType() == DefinedAtom::typeZeroFill ) ||
      ( this->contentType() == DefinedAtom::typeZeroFillFast))
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
  return _file->target(_ivarData->targetIndex);
}

inline Reference::Addend NativeReferenceV1::addend() const {
  return _file->addend(_ivarData->addendIndex);
}

inline void NativeReferenceV1::setTarget(const Atom* newAtom) {
  return _file->setTarget(_ivarData->targetIndex, newAtom);
}

inline void NativeReferenceV1::setAddend(Addend a) {
  // Do nothing if addend value is not being changed.
  if (addend() == a)
    return;
  llvm_unreachable("setAddend() not supported");
}

class Reader : public lld::Reader {
public:
  Reader(const TargetInfo &ti)
   : lld::Reader(ti) {}

  virtual error_code parseFile(
      std::unique_ptr<MemoryBuffer> &mb,
      std::vector<std::unique_ptr<lld::File> > &result) const {
    return File::make(_targetInfo, mb, mb->getBufferIdentifier(), result);
  }
};
} // end namespace native

std::unique_ptr<Reader> createReaderNative(const TargetInfo &ti) {
  return std::unique_ptr<Reader>(new lld::native::Reader(ti));
}
} // end namespace lld
