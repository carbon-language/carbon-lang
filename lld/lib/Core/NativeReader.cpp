//===- Core/NativeReader.cpp - reads native object file  ------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <vector>

#include <assert.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"

#include "lld/Core/Error.h"
#include "lld/Core/File.h"
#include "lld/Core/Atom.h"

#include "NativeFileFormat.h"

namespace lld {

// forward reference
class NativeFile;

//
// An object of this class is instantied for each NativeDefinedAtomIvarsV1
// struct in the NCS_DefinedAtomsV1 chunk.
//
class NativeDefinedAtomV1 : public DefinedAtom {
public:
      NativeDefinedAtomV1(const NativeFile& f, 
                          const NativeDefinedAtomIvarsV1* ivarData)
        : _file(&f), _ivarData(ivarData) { } 
  
  virtual const class File& file() const;
  
  virtual uint64_t ordinal() const; 
  
  virtual llvm::StringRef name() const;
  
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
    return (DefinedAtom::SectionChoice)(attributes().sectionChoice);
  }

  virtual llvm::StringRef customSectionName() const;
      
  virtual DefinedAtom::DeadStripKind deadStrip() const {
     return (DefinedAtom::DeadStripKind)(attributes().deadStrip);
 }
    
  virtual DefinedAtom::ContentPermissions permissions() const {
     return (DefinedAtom::ContentPermissions)(attributes().permissions);
  }
  
  virtual bool isThumb() const {
     return false; //(attributes().thumb != 0);
  }
    
  virtual bool isAlias() const {
     return (attributes().alias != 0);
  }
  
  virtual llvm::ArrayRef<uint8_t> rawContent() const;
    
  virtual void forEachReference(ReferenceHandler&) const;
   
private:
  const NativeAtomAttributesV1& attributes() const;
  
  const NativeFile*               _file;
  const NativeDefinedAtomIvarsV1* _ivarData;
};



//
// An object of this class is instantied for each NativeUndefinedAtomIvarsV1
// struct in the NCS_UndefinedAtomsV1 chunk.
//
class NativeUndefinedAtomV1 : public UndefinedAtom {
public:
       NativeUndefinedAtomV1(const NativeFile& f, 
                             const NativeUndefinedAtomIvarsV1* ivarData)
        : _file(&f), _ivarData(ivarData) { } 

  virtual const File& file() const;
  virtual llvm::StringRef name() const;
  
  virtual bool weakImport() const {
    return (_ivarData->flags & 0x1);
  }
  
private:
  const NativeFile*                 _file;
  const NativeUndefinedAtomIvarsV1* _ivarData;
};



//
// An object of this class is instantied for each NativeReferenceIvarsV1
// struct in the NCS_ReferencesArrayV1 chunk.
//
class NativeReferenceV1 : public Reference {
public:
       NativeReferenceV1(const NativeFile& f, 
                             const NativeReferenceIvarsV1* ivarData)
        : _file(&f), _ivarData(ivarData) { } 

  virtual uint64_t offsetInAtom() const {
    return _ivarData->offsetInAtom;
  }
  
  virtual Kind kind() const {
    return _ivarData->kind;
  }
  
  virtual const Atom* target() const;
  virtual Addend addend() const;
  virtual void setTarget(const Atom* newAtom);
   
private:
  const NativeFile*                 _file;
  const NativeReferenceIvarsV1*     _ivarData;
};



//
// lld::File object for native llvm object file
//
class NativeFile : public File {
public: 

  /// Instantiates a File object from a native object file.  Ownership
  /// of the MemoryBuffer is transfered to the resulting File object.
  static llvm::error_code make(llvm::OwningPtr<llvm::MemoryBuffer>& mb, 
                               llvm::StringRef path, 
                               llvm::OwningPtr<File>& result) {
    const uint8_t* const base = 
                       reinterpret_cast<const uint8_t*>(mb->getBufferStart());
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

    // instantiate NativeFile object and add values to it as found
    NativeFile* file = new NativeFile(mb, path);
    
    // process each chunk
    for(uint32_t i=0; i < header->chunkCount; ++i) {
      llvm::error_code ec;
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
        delete file;
        return ec;
      }
      
      // TO DO: validate enough chunks were used
      
      result.reset(file);
    }


    return make_error_code(native_reader_error::success);
  }
  
  virtual ~NativeFile() {
    // _buffer is automatically deleted because of OwningPtr<>
    
    // All other ivar pointers are pointers into the MemoryBuffer, except
    // the _definedAtoms array which was allocated to contain an array
    // of Atom objects.  The atoms have empty destructors, so it is ok
    // to just delete the memory.
    delete _definedAtoms.arrayStart;
    delete _undefinedAtoms.arrayStart;
    delete _references.arrayStart;
    delete _targetsTable;
  }
  
  // visits each atom in the file
  virtual bool forEachAtom(AtomHandler& handler) const {
    for(const uint8_t* p=_definedAtoms.arrayStart; p != _definedAtoms.arrayEnd; 
          p += _definedAtoms.elementSize) {
      const DefinedAtom* atom = reinterpret_cast<const DefinedAtom*>(p);
      handler.doDefinedAtom(*atom);
    }
    for(const uint8_t* p=_undefinedAtoms.arrayStart; p != _undefinedAtoms.arrayEnd; 
          p += _undefinedAtoms.elementSize) {
      const UndefinedAtom* atom = reinterpret_cast<const UndefinedAtom*>(p);
      handler.doUndefinedAtom(*atom);
    }
    return (_definedAtoms.arrayStart != _definedAtoms.arrayEnd);
  }
  
  // not used
  virtual bool justInTimeforEachAtom(llvm::StringRef name,
                                              AtomHandler &) const {
    return false;
  }
  
private:
  friend class NativeDefinedAtomV1;
  friend class NativeUndefinedAtomV1;
  friend class NativeReferenceV1;
  
  // instantiate array of DefinedAtoms from v1 ivar data in file
  llvm::error_code processDefinedAtomsV1(const uint8_t* base, 
                                                const NativeChunk* chunk) {
    const size_t atomSize = sizeof(NativeDefinedAtomV1);
    size_t atomsArraySize = chunk->elementCount * atomSize;
    uint8_t* atomsStart = reinterpret_cast<uint8_t*>
                                (operator new(atomsArraySize, std::nothrow));
    if (atomsStart == NULL )
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
    this->_definedAtoms.arrayStart = atomsStart;
    this->_definedAtoms.arrayEnd = atomsEnd;
    this->_definedAtoms.elementSize = atomSize;
    this->_definedAtoms.elementCount = chunk->elementCount;
    return make_error_code(native_reader_error::success);
  }
  
  // set up pointers to attributes array
  llvm::error_code processAttributesV1(const uint8_t* base, const NativeChunk* chunk) {
    this->_attributes = base + chunk->fileOffset;
    this->_attributesMaxOffset = chunk->fileSize;
    return make_error_code(native_reader_error::success);
  }
  
  llvm::error_code processUndefinedAtomsV1(const uint8_t* base, 
                                                const NativeChunk* chunk) {
    const size_t atomSize = sizeof(NativeUndefinedAtomV1);
    size_t atomsArraySize = chunk->elementCount * atomSize;
    uint8_t* atomsStart = reinterpret_cast<uint8_t*>
                                (operator new(atomsArraySize, std::nothrow));
    if (atomsStart == NULL )
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
    this->_undefinedAtoms.arrayStart = atomsStart;
    this->_undefinedAtoms.arrayEnd = atomsEnd;
    this->_undefinedAtoms.elementSize = atomSize;
    this->_undefinedAtoms.elementCount = chunk->elementCount;
    return make_error_code(native_reader_error::success);
  }
  
  
  // instantiate array of Referemces from v1 ivar data in file
  llvm::error_code processReferencesV1(const uint8_t* base, 
                                                const NativeChunk* chunk) {
    if ( chunk->elementCount == 0 )
      return make_error_code(native_reader_error::success);
    const size_t refSize = sizeof(NativeReferenceV1);
    size_t refsArraySize = chunk->elementCount * refSize;
    uint8_t* refsStart = reinterpret_cast<uint8_t*>
                                (operator new(refsArraySize, std::nothrow));
    if (refsStart == NULL )
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
    return make_error_code(native_reader_error::success);
  }
  
  // set up pointers to target table
  llvm::error_code processTargetsTable(const uint8_t* base, 
                                                const NativeChunk* chunk) {
    const uint32_t* targetIndexes = reinterpret_cast<const uint32_t*>
                                                  (base + chunk->fileOffset);
    this->_targetsTableCount = chunk->elementCount;
    this->_targetsTable = new const Atom*[chunk->elementCount];
    for (uint32_t i=0; i < chunk->elementCount; ++i) {
      const uint32_t index = targetIndexes[i];
      if ( index < _definedAtoms.elementCount ) {
        const uint8_t* p = _definedAtoms.arrayStart 
                                    + index * _definedAtoms.elementSize;
        this->_targetsTable[i] = reinterpret_cast<const DefinedAtom*>(p);
        continue;
      }
      const uint32_t undefIndex = index - _definedAtoms.elementCount;
      if ( undefIndex < _undefinedAtoms.elementCount ) {
        const uint8_t* p = _undefinedAtoms.arrayStart 
                                    + undefIndex * _undefinedAtoms.elementSize;
        this->_targetsTable[i] = reinterpret_cast<const UndefinedAtom*>(p);
        continue;
      }
      return make_error_code(native_reader_error::file_malformed);
    }
    return make_error_code(native_reader_error::success);
  }
  
  
  // set up pointers to addend pool in file
  llvm::error_code processAddendsTable(const uint8_t* base, 
                                                const NativeChunk* chunk) {
    this->_addends = reinterpret_cast<const Reference::Addend*>
                                                  (base + chunk->fileOffset);
    this->_addendsMaxIndex = chunk->elementCount;
    return make_error_code(native_reader_error::success);
  }
  
  // set up pointers to string pool in file
  llvm::error_code processStrings(const uint8_t* base, 
                                                const NativeChunk* chunk) {
    this->_strings = reinterpret_cast<const char*>(base + chunk->fileOffset);
    this->_stringsMaxOffset = chunk->fileSize;
    return make_error_code(native_reader_error::success);
  }
  
  // set up pointers to content area in file
  llvm::error_code processContent(const uint8_t* base, 
                                                const NativeChunk* chunk) {
    this->_contentStart = base + chunk->fileOffset;
    this->_contentEnd = base + chunk->fileOffset + chunk->fileSize;
    return make_error_code(native_reader_error::success);
  }
  
  llvm::StringRef string(uint32_t offset) const {
    assert(offset < _stringsMaxOffset);
    return llvm::StringRef(&_strings[offset]);
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

  const uint8_t* content(uint32_t offset, uint32_t size) const {
    const uint8_t* result = _contentStart + offset;
    assert((result+size) <= _contentEnd);
    return result;
  }

  void forEachReference(DefinedAtom::ReferenceHandler& handler, 
                                        uint32_t start, uint32_t count) const {
    assert(start < _references.elementCount);
    assert(start+count <= _references.elementCount);
    const uint8_t* arrStart = _references.arrayStart 
                                   + start * _references.elementSize;
    const uint8_t* arrEnd = arrStart + count * _references.elementSize;
    for(const uint8_t* p=arrStart; p != arrEnd; p += _references.elementSize) {
      const NativeReferenceV1* ref 
                              = reinterpret_cast<const NativeReferenceV1*>(p);
      handler.doReference(*ref);
    }
  }
  
  const Atom* target(uint32_t index) const {
    assert(index < _targetsTableCount);
    return _targetsTable[index];
  }
  
  void setTarget(uint32_t index, const Atom* newAtom) const {
    assert(index > _targetsTableCount);
    _targetsTable[index] = newAtom;
  }
 
  
  // private constructor, only called by make()
  NativeFile(llvm::OwningPtr<llvm::MemoryBuffer>& mb, llvm::StringRef path) :
    lld::File(path), 
    _buffer(mb.take()),  // NativeFile now takes ownership of buffer
    _header(NULL), 
    _targetsTable(NULL), 
    _targetsTableCount(0),
    _strings(NULL), 
    _stringsMaxOffset(0),
    _addends(NULL), 
    _addendsMaxIndex(0),
    _contentStart(NULL), 
    _contentEnd(NULL)
  {
    _header = reinterpret_cast<const NativeFileHeader*>(_buffer->getBufferStart());
  }

  struct IvarArray {
                      IvarArray() : arrayStart(NULL), arrayEnd(NULL), 
                                    elementSize(0), elementCount(0) { }
    const uint8_t*     arrayStart;
    const uint8_t*     arrayEnd;
    uint32_t           elementSize;
    uint32_t           elementCount;
 };

  llvm::OwningPtr<llvm::MemoryBuffer>  _buffer;
  const NativeFileHeader*         _header;
  IvarArray                       _definedAtoms;
  IvarArray                       _undefinedAtoms;
  const uint8_t*                  _attributes;
  uint32_t                        _attributesMaxOffset;
  IvarArray                       _references;
  const Atom**                    _targetsTable;
  uint32_t                        _targetsTableCount;
  const char*                     _strings;
  uint32_t                        _stringsMaxOffset;
  const Reference::Addend*        _addends;
  uint32_t                        _addendsMaxIndex;
  const uint8_t*                  _contentStart;
  const uint8_t*                  _contentEnd;
};

 
inline const class File& NativeDefinedAtomV1::file() const {
  return *_file;
}

inline uint64_t NativeDefinedAtomV1:: ordinal() const {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(_ivarData);
  return p - _file->_definedAtoms.arrayStart;
}

inline llvm::StringRef NativeDefinedAtomV1::name() const {
  return _file->string(_ivarData->nameOffset);
}

inline const NativeAtomAttributesV1& NativeDefinedAtomV1::attributes() const {
  return _file->attribute(_ivarData->attributesOffset);
}

inline llvm::ArrayRef<uint8_t> NativeDefinedAtomV1::rawContent() const {
  if ( this->contentType() == DefinedAtom::typeZeroFill )
    return llvm::ArrayRef<uint8_t>();
  const uint8_t* p = _file->content(_ivarData->contentOffset,
                                    _ivarData->contentSize);
   return llvm::ArrayRef<uint8_t>(p, _ivarData->contentSize);
}

inline llvm::StringRef NativeDefinedAtomV1::customSectionName() const {
  uint32_t offset = attributes().sectionNameOffset;
  return _file->string(offset);
}

inline void NativeDefinedAtomV1::forEachReference(ReferenceHandler& hnd) const {
  if ( _ivarData->referencesCount == 0 )
    return;
  _file->forEachReference(hnd, _ivarData->referencesStartIndex, 
                               _ivarData->referencesCount);
}

inline const class File& NativeUndefinedAtomV1::file() const {
  return *_file;
}

inline llvm::StringRef NativeUndefinedAtomV1::name() const {
  return _file->string(_ivarData->nameOffset);
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


//
// Instantiate an lld::File from the given native object file buffer
//
llvm::error_code parseNativeObjectFile(llvm::OwningPtr<llvm::MemoryBuffer>& mb, 
                                       llvm::StringRef path, 
                                       llvm::OwningPtr<File>& result) {
  return NativeFile::make(mb, path, result);
}



//
// Instantiate an lld::File from the given native object file path
//
llvm::error_code parseNativeObjectFileOrSTDIN(llvm::StringRef path, 
                                              llvm::OwningPtr<File>& result) {
  llvm::OwningPtr<llvm::MemoryBuffer> mb;
  llvm::error_code ec = llvm::MemoryBuffer::getFileOrSTDIN(path, mb);
  if ( ec ) 
      return ec;

  return parseNativeObjectFile(mb, path, result);
}



} // namespace lld
