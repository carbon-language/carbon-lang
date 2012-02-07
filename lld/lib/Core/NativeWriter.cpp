//===- Core/NativeWriter.cpp - Creates a native object file ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <vector>
#include <map>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ArrayRef.h"

#include "lld/Core/File.h"
#include "lld/Core/NativeWriter.h"

#include "NativeFileFormat.h"


namespace lld {


///
/// Class for writing native object files.
///
class NativeWriter : public File::AtomHandler {
public:
  /// construct writer for an lld::File object
  NativeWriter(const lld::File& file) : _file(file) { 
    // visit all atoms
    _file.forEachAtom(*this);
    // construct file header based on atom information accumulated
    makeHeader();
  }

  // write the lld::File in native format to the specified stream
  void write(llvm::raw_ostream& out) {
    out.write((char*)_headerBuffer, _headerBufferSize);
    if (!_definedAtomIvars.empty())
      out.write((char*)&_definedAtomIvars[0],
                _definedAtomIvars.size()*sizeof(NativeDefinedAtomIvarsV1));
    if (!_attributes.empty())
      out.write((char*)&_attributes[0],
                _attributes.size()*sizeof(NativeAtomAttributesV1));
    if ( !_undefinedAtomIvars.empty() ) 
      out.write((char*)&_undefinedAtomIvars[0], 
              _undefinedAtomIvars.size()*sizeof(NativeUndefinedAtomIvarsV1));
    if (!_stringPool.empty())
      out.write(&_stringPool[0], _stringPool.size());
    if (!_contentPool.empty())
      out.write((char*)&_contentPool[0], _contentPool.size());
  }

private:

  // visitor routine called by forEachAtom() 
  virtual void doDefinedAtom(const class DefinedAtom& atom) {
    NativeDefinedAtomIvarsV1 ivar;
    ivar.nameOffset = getNameOffset(atom);
    ivar.attributesOffset = getAttributeOffset(atom);
    ivar.contentOffset = getContentOffset(atom);
    ivar.contentSize = atom.size();
    _definedAtomIvars.push_back(ivar);
  }
  
  // visitor routine called by forEachAtom() 
  virtual void doUndefinedAtom(const class UndefinedAtom& atom) {
    NativeUndefinedAtomIvarsV1 ivar;
    ivar.nameOffset = getNameOffset(atom);
    ivar.flags = (atom.weakImport() ? 1 : 0);
    _undefinedAtomIvars.push_back(ivar);
  }
  
  // visitor routine called by forEachAtom() 
  virtual void doFile(const class File &) {
  }

  // fill out native file header and chunk directory
  void makeHeader() {
    const bool hasUndefines = !_undefinedAtomIvars.empty();
    const int chunkCount = hasUndefines ? 5 : 4;
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
    
    
    // create chunk for atom ivar array
    int nextIndex = 0;
    NativeChunk& chd = chunks[nextIndex++];
    chd.signature = NCS_DefinedAtomsV1;
    chd.fileOffset = _headerBufferSize;
    chd.fileSize = _definedAtomIvars.size()*sizeof(NativeDefinedAtomIvarsV1);
    chd.elementCount = _definedAtomIvars.size();
    uint32_t nextFileOffset = chd.fileOffset + chd.fileSize;

    // create chunk for attributes 
    NativeChunk& cha = chunks[nextIndex++];
    cha.signature = NCS_AttributesArrayV1;
    cha.fileOffset = nextFileOffset;
    cha.fileSize = _attributes.size()*sizeof(NativeAtomAttributesV1);
    cha.elementCount = _attributes.size();
    nextFileOffset = cha.fileOffset + cha.fileSize;
    
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
    
    // create chunk for symbol strings
    NativeChunk& chs = chunks[nextIndex++];
    chs.signature = NCS_Strings;
    chs.fileOffset = nextFileOffset;
    chs.fileSize = _stringPool.size();
    chs.elementCount = _stringPool.size();
    nextFileOffset = chs.fileOffset + chs.fileSize;
    
    // create chunk for content 
    NativeChunk& chc = chunks[nextIndex++];
    chc.signature = NCS_Content;
    chc.fileOffset = nextFileOffset;
    chc.fileSize = _contentPool.size();
    chc.elementCount = _contentPool.size();
    nextFileOffset = chc.fileOffset + chc.fileSize;

    _headerBuffer->fileSize = nextFileOffset;
  }


  // append atom name to string pool and return offset
  uint32_t getNameOffset(const Atom& atom) {
    return this->getNameOffset(atom.name());
  }
  
 // append atom name to string pool and return offset
  uint32_t getNameOffset(llvm::StringRef name) {
    uint32_t result = _stringPool.size();
    _stringPool.insert(_stringPool.end(), name.size()+1, 0);
    strcpy(&_stringPool[result], name.data());
    return result;
  }

  // append atom cotent to content pool and return offset
  uint32_t getContentOffset(const class DefinedAtom& atom) {
    if ( atom.contentType() == DefinedAtom::typeZeroFill ) 
      return 0;
    uint32_t result = _contentPool.size();
    llvm::ArrayRef<uint8_t> cont = atom.rawContent();
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
  
  uint32_t sectionNameOffset(const class DefinedAtom& atom) {
    // if section based on content, then no custom section name available
    if ( atom.sectionChoice() == DefinedAtom::sectionBasedOnContent )
      return 0;
    llvm::StringRef name = atom.customSectionName();
    assert( ! name.empty() );
    // look to see if this section name was used by another atom
    for(NameToOffsetVector::iterator it=_sectionNames.begin(); 
                                            it != _sectionNames.end(); ++it) {
      if ( name.equals(it->first) )
        return it->second;
    }
    // first use of this section name
    uint32_t result = this->getNameOffset(name);
    _sectionNames.push_back(
                    std::make_pair<llvm::StringRef, uint32_t>(name, result));
    return result;
  }
  
  void computeAttributesV1(const class DefinedAtom& atom, 
                                                NativeAtomAttributesV1& attrs) {
    attrs.sectionNameOffset = sectionNameOffset(atom);
    attrs.align2            = atom.alignment().powerOf2;
    attrs.alignModulus      = atom.alignment().modulus;
    attrs.internalName      = atom.internalName(); 
    attrs.scope             = atom.scope(); 
    attrs.interposable      = atom.interposable(); 
    attrs.merge             = atom.merge(); 
    attrs.contentType       = atom.contentType(); 
    attrs.sectionChoice     = atom.sectionChoice(); 
    attrs.deadStrip         = atom.deadStrip(); 
    attrs.permissions       = atom.permissions(); 
    attrs.thumb             = atom.isThumb(); 
    attrs.alias             = atom.isAlias(); 
  }

  typedef std::vector<std::pair<llvm::StringRef, uint32_t> > NameToOffsetVector;

  const lld::File&                        _file;
  NativeFileHeader*                       _headerBuffer;
  size_t                                  _headerBufferSize;
  std::vector<char>                       _stringPool;
  std::vector<uint8_t>                    _contentPool;
  std::vector<NativeDefinedAtomIvarsV1>   _definedAtomIvars;
  std::vector<NativeAtomAttributesV1>     _attributes;
  std::vector<NativeUndefinedAtomIvarsV1> _undefinedAtomIvars;
  NameToOffsetVector                      _sectionNames;
};





/// writeNativeObjectFile - writes the lld::File object in native object
/// file format to the specified stream.
int writeNativeObjectFile(const lld::File &file, llvm::raw_ostream &out) {
  NativeWriter  writer(file);
  writer.write(out);
  return 0;
}

/// writeNativeObjectFile - writes the lld::File object in native object
/// file format to the specified file path.
int writeNativeObjectFile(const lld::File& file, llvm::StringRef path) {
  std::string errorInfo;
  llvm::raw_fd_ostream out(path.data(), errorInfo, llvm::raw_fd_ostream::F_Binary);
  if ( !errorInfo.empty() )
    return -1;
  return writeNativeObjectFile(file, out);
}

} // namespace lld
