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
#include "llvm/Support/system_error.h"

#include "lld/Core/File.h"
#include "lld/Core/Atom.h"

#include "NativeFileFormat.h"

namespace lld {

// forward reference
class NativeFile;


enum native_reader_errors {
  success = 0,
  unknown_file_format,
  file_too_short,
  file_malformed,
  unknown_chunk_type,
  memory_error,
};

class reader_error_category : public llvm::_do_message {
public:
  virtual const char* name() const {
    return "lld.native.reader";
  }
  virtual std::string message(int ev) const;
};

const reader_error_category reader_error_category_singleton;

std::string reader_error_category::message(int ev) const {
  switch (ev) {
  case success: 
    return "Success";
  case unknown_file_format:
    return "Unknown file foramt";
  case file_too_short: 
    return "file truncated";
  case file_malformed: 
    return "file malformed";
  case memory_error: 
    return "out of memory";
  case unknown_chunk_type:
    return "unknown chunk type";
  default:
    llvm_unreachable("An enumerator of native_reader_errors does not have a "
                     "message defined.");
  }
}

inline llvm::error_code make_error_code(native_reader_errors e) {
  return llvm::error_code(static_cast<int>(e), reader_error_category_singleton);
}




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
  
  virtual bool internalName() const {
    return attributes().internalName;
  }
  
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
    return (DefinedAtom::ContentType)(attributes().contentType);
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
     return (attributes().thumb != 0);
  }
    
  virtual bool isAlias() const {
     return (attributes().alias != 0);
  }
  
  llvm::ArrayRef<uint8_t> rawContent() const;
    
  virtual Reference::iterator referencesBegin() const {
     return 0;
  }
  
  virtual Reference::iterator referencesEnd() const {
    return 0;
  }

private:
  const NativeAtomAttributesV1& attributes() const;
  
  const NativeFile*               _file;
  const NativeDefinedAtomIvarsV1* _ivarData;
};




//
// lld::File object for native llvm object file
//
class NativeFile : public File {
public: 

  /// Instantiates a File object from a native object file.  Ownership
  /// of the MemoryBuffer is transfered to the resulting File object.
  static llvm::error_code make(llvm::MemoryBuffer* mb, llvm::StringRef path, 
                                                                File*& result) {
    const uint8_t* const base = 
                       reinterpret_cast<const uint8_t*>(mb->getBufferStart());
    const NativeFileHeader* const header = 
                       reinterpret_cast<const NativeFileHeader*>(base);
    // make sure magic matches
    if ( memcmp(header->magic, NATIVE_FILE_HEADER_MAGIC, 16) != 0 )
      return make_error_code(unknown_file_format);
    
    // make sure mapped file contains all needed data
    const size_t fileSize = mb->getBufferSize();
    if ( header->fileSize > fileSize )
      return make_error_code(file_too_short);
    
    // instantiate NativeFile object and add values to it as found
    NativeFile* file = new NativeFile(mb, path);
    
    // process each chunk
    for(uint32_t i=0; i < header->chunkCount; ++i) {
      llvm::error_code ec;
      const NativeChunk* chunk = &header->chunks[i];
      // sanity check chunk is within file
      if ( chunk->fileOffset > fileSize ) 
        return make_error_code(file_malformed);
      if ( (chunk->fileOffset + chunk->fileSize) > fileSize) 
        return make_error_code(file_malformed);
      // process chunk, based on signature
      switch ( chunk->signature ) {
        case NCS_DefinedAtomsV1:
          ec = file->processDefinedAtomsV1(base, chunk);
          break;
        case NCS_AttributesArrayV1:
          ec = file->processAttributesV1(base, chunk);
          break;
        case NCS_Content:
          ec = file->processContent(base, chunk);
          break;
        case NCS_Strings:
          ec = file->processStrings(base, chunk);
          break;
        default:
          return make_error_code(unknown_chunk_type);
      }
      if ( ec ) {
        delete file;
        return ec;
      }
      
      // TO DO: validate enough chunks were used
      
      result = file;
    }
    

    return make_error_code(success);
  }
  
  virtual ~NativeFile() {
    // The NativeFile owns the MemoryBuffer and must not delete it.
    delete _buffer;
    // All other ivar pointers are pointers into the MemoryBuffer, except
    // the _definedAtoms array which was allocated to contain an array
    // of Atom objects.  The atoms have empty destructors, so it is ok
    // to just delete the memory.
    delete _definedAtoms.arrayStart;
  }
  
  // visits each atom in the file
  virtual bool forEachAtom(AtomHandler& handler) const {
    for(const uint8_t* p=_definedAtoms.arrayStart; p != _definedAtoms.arrayEnd; 
          p += _definedAtoms.elementSize) {
      const DefinedAtom* atom = reinterpret_cast<const DefinedAtom*>(p);
      handler.doDefinedAtom(*atom);
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
  
  // instantiate array of DefinedAtoms from v1 ivar data in file
  llvm::error_code processDefinedAtomsV1(const uint8_t* base, 
                                                const NativeChunk* chunk) {
    const size_t atomSize = sizeof(NativeDefinedAtomV1);
    size_t atomsArraySize = chunk->elementCount * atomSize;
    uint8_t* atomsStart = reinterpret_cast<uint8_t*>
                                (operator new(atomsArraySize, std::nothrow));
    if (atomsStart == NULL )
      return make_error_code(memory_error);
    const size_t ivarElementSize = chunk->fileSize 
                                          / chunk->elementCount;
    if ( ivarElementSize != sizeof(NativeDefinedAtomIvarsV1) )
      return make_error_code(file_malformed);
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
    return make_error_code(success);
  }
  
  // set up pointers to attributes array
  llvm::error_code processAttributesV1(const uint8_t* base, const NativeChunk* chunk) {
    this->_attributes = base + chunk->fileOffset;
    this->_attributesMaxOffset = chunk->fileSize;
    return make_error_code(success);
  }
  
  // set up pointers to string pool in file
  llvm::error_code processStrings(const uint8_t* base, 
                                                const NativeChunk* chunk) {
    this->_strings = reinterpret_cast<const char*>(base + chunk->fileOffset);
    this->_stringsMaxOffset = chunk->fileSize;
    return make_error_code(success);
  }
  
  // set up pointers to content area in file
  llvm::error_code processContent(const uint8_t* base, 
                                                const NativeChunk* chunk) {
    this->_contentStart = base + chunk->fileOffset;
    this->_contentEnd = base + chunk->fileOffset + chunk->fileSize;
    return make_error_code(success);
  }
  
  llvm::StringRef string(uint32_t offset) const {
    assert(offset < _stringsMaxOffset);
    return llvm::StringRef(&_strings[offset]);
  }
  
  const NativeAtomAttributesV1& attribute(uint32_t offset) const {
    assert(offset < _attributesMaxOffset);
    return *reinterpret_cast<const NativeAtomAttributesV1*>(_attributes + offset);
  }

  const uint8_t* content(uint32_t offset, uint32_t size) const {
    const uint8_t* result = _contentStart + offset;
    assert((result+size) <= _contentEnd);
    return result;
  }


  // private constructor, only called by make()
  NativeFile(llvm::MemoryBuffer* mb, llvm::StringRef path) :
    lld::File(path), _buffer(mb), _header(NULL), 
    _strings(NULL), _stringsMaxOffset(0),
    _contentStart(NULL), _contentEnd(NULL)
  {
    _header = reinterpret_cast<const NativeFileHeader*>(mb->getBufferStart());
  }

  struct AtomArray {
                      AtomArray() : arrayStart(NULL), arrayEnd(NULL), 
                                    elementSize(0) { }
    const uint8_t*     arrayStart;
    const uint8_t*     arrayEnd;
    uint32_t           elementSize;
  };

  llvm::MemoryBuffer*             _buffer;
  const NativeFileHeader*         _header;
  AtomArray                       _definedAtoms;
  const uint8_t*                  _attributes;
  uint32_t                        _attributesMaxOffset;
  const char*                     _strings;
  uint32_t                        _stringsMaxOffset;
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



//
// Instantiate an lld::File from the given native object file buffer
//
llvm::error_code parseNativeObjectFile(llvm::MemoryBuffer* mb, 
                                       llvm::StringRef path, File*& result) {
  return NativeFile::make(mb, path, result);
}



//
// Instantiate an lld::File from the given native object file path
//
llvm::error_code parseNativeObjectFileOrSTDIN(llvm::StringRef path, 
                                              File*& result) {
  llvm::OwningPtr<llvm::MemoryBuffer> mb;
  llvm::error_code ec = llvm::MemoryBuffer::getFileOrSTDIN(path, mb);
  if ( ec ) 
      return ec;

  return parseNativeObjectFile(mb.get(), path, result);
}



} // namespace lld
