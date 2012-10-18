//===- lib/ReaderWriter/Native/NativeFileFormat.h -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_NATIVE_FILE_FORMAT_H_
#define LLD_READER_WRITER_NATIVE_FILE_FORMAT_H_

#include "llvm/Support/DataTypes.h"

namespace lld {


//
// Overview:
//
// The number one design goal of this file format is enable the linker to
// read object files into in-memory Atom objects extremely quickly.
// The second design goal is to enable future modifications to the
// Atom attribute model.
//
// The llvm native object file format is not like traditional object file
// formats (e.g. ELF, COFF, mach-o).  There is no symbol table and no
// sections.  Instead the file is essentially an array of archived Atoms.
// It is *not* serialized Atoms which would require deserialization into
// in memory objects.  Instead it is an array of read-only info about each
// Atom.  The NativeReader bulk creates in-memory Atoms which just have
// an ivar which points to the read-only info for that Atom. No additional
// processing is done to construct the in-memory Atoms. All Atom attribute
// getter methods are virtual calls which dig up the info they need from the
// ivar data.
//
// To support the gradual evolution of Atom attributes, the Atom read-only
// data is versioned. The NativeReader chooses which in-memory Atom class
// to use based on the version. What this means is that if new attributes
// are added (or changed) in the Atom model, a new native atom class and
// read-only atom info struct needs to be defined.  Then, all the existing
// native reader atom classes need to be modified to do their best effort
// to map their old style read-only data to the new Atom model.  At some point
// some classes to support old versions may be dropped.
//
//
// Details:
//
// The native object file format consists of a header that specifies the
// endianness of the file and the architecture along with a list of "chunks"
// in the file.  A Chunk is simply a tagged range of the file.  There is
// one chunk for the array of atom infos.  There is another chunk for the
// string pool, and another for the content pool.
//
// It turns out there most atoms have very similar sets of attributes, only
// the name and content attribute vary. To exploit this fact to reduce the file
// size, the atom read-only info contains just the name and content info plus
// a reference to which attribute set it uses.  The attribute sets are stored
// in another chunk.
//


//
// An entry in the NativeFileHeader that describes one chunk of the file.
//
struct NativeChunk {
  uint32_t    signature;
  uint32_t    fileOffset;
  uint32_t    fileSize;
  uint32_t    elementCount;
};


//
// The header in a native object file
//
struct NativeFileHeader {
  uint8_t     magic[16];
  uint32_t    endian;
  uint32_t    architecture;
  uint32_t    fileSize;
  uint32_t    chunkCount;
  // NativeChunk chunks[]
};

//
// Possible values for NativeChunk.signature field
//
enum NativeChunkSignatures {
  NCS_DefinedAtomsV1 = 1,
  NCS_AttributesArrayV1 = 2,
  NCS_AbsoluteAttributesV1 = 12,
  NCS_UndefinedAtomsV1 = 3,
  NCS_SharedLibraryAtomsV1 = 4,
  NCS_AbsoluteAtomsV1 = 5,
  NCS_Strings = 6,
  NCS_ReferencesArrayV1 = 7,
  NCS_ReferencesArrayV2 = 8,
  NCS_TargetsTable = 9,
  NCS_AddendsTable = 10,
  NCS_Content = 11,
};

//
// The 16-bytes at the start of a native object file
//
#define NATIVE_FILE_HEADER_MAGIC "llvm nat obj v1 "

//
// Possible values for the NativeFileHeader.endian field
//
enum {
  NFH_BigEndian     = 0x42696745,
  NFH_LittleEndian  = 0x4574696c
};


//
// Possible values for the NativeFileHeader.architecture field
//
enum {
  NFA_x86     = 1,
  NFA_x86_64  = 2,
  NFA_armv6   = 3,
  NFA_armv7   = 4,
};


//
// The NCS_DefinedAtomsV1 chunk contains an array of these structs
//
struct NativeDefinedAtomIvarsV1 {
  uint32_t  nameOffset;
  uint32_t  attributesOffset;
  uint32_t  referencesStartIndex;
  uint32_t  referencesCount;
  uint32_t  contentOffset;
  uint32_t  contentSize;
};


//
// The NCS_AttributesArrayV1 chunk contains an array of these structs
//
struct NativeAtomAttributesV1 {
  uint32_t  sectionNameOffset;
  uint16_t  align2;
  uint16_t  alignModulus;
  uint8_t   scope;
  uint8_t   interposable;
  uint8_t   merge;
  uint8_t   contentType;
  uint8_t   sectionChoice;
  uint8_t   deadStrip;
  uint8_t   permissions;
  uint8_t   alias;
};



//
// The NCS_UndefinedAtomsV1 chunk contains an array of these structs
//
struct NativeUndefinedAtomIvarsV1 {
  uint32_t  nameOffset;
  uint32_t  flags;
};



//
// The NCS_SharedLibraryAtomsV1 chunk contains an array of these structs
//
struct NativeSharedLibraryAtomIvarsV1 {
  uint32_t  nameOffset;
  uint32_t  loadNameOffset;
  uint32_t  flags;
};



//
// The NCS_AbsoluteAtomsV1 chunk contains an array of these structs
//
struct NativeAbsoluteAtomIvarsV1 {
  uint32_t  nameOffset;
  uint32_t  attributesOffset;
  uint32_t  reserved;
  uint64_t  value;
};



//
// The NCS_ReferencesArrayV1 chunk contains an array of these structs
//
struct NativeReferenceIvarsV1 {
  enum {
    noTarget = 0xFFFF
  };
  uint16_t  offsetInAtom;
   int16_t  kind;
  uint16_t  targetIndex;
  uint16_t  addendIndex;
};


//
// The NCS_ReferencesArrayV2 chunk contains an array of these structs
//
struct NativeReferenceIvarsV2 {
  uint64_t  offsetInAtom;
  int64_t   addend;
  int32_t   kind;
  uint32_t  targetIndex;
};


//
// The NCS_TargetsTable chunk contains an array of uint32_t entries.
// The C++ class Reference has a target() method that returns a
// pointer to another Atom.  We can't have pointers in object files,
// so instead  NativeReferenceIvarsV1 contains an index to the target.
// The index is into this NCS_TargetsTable of uint32_t entries.
// The values in this table are the index of the (target) atom in this file.
// For DefinedAtoms the value is from 0 to NCS_DefinedAtomsV1.elementCount.
// For UndefinedAtoms the value is from NCS_DefinedAtomsV1.elementCount to
// NCS_DefinedAtomsV1.elementCount+NCS_UndefinedAtomsV1.elementCount.
//


//
// The NCS_AddendsTable chunk contains an array of int64_t entries.
// If we allocated space for addends directly in NativeReferenceIvarsV1
// it would double the size of that struct.  But since addends are rare,
// we instead just keep a pool of addends and have NativeReferenceIvarsV1
// (if it needs an addend) just store the index (into the pool) of the
// addend it needs.
//



} // namespace lld

#endif // LLD_READER_WRITER_NATIVE_FILE_FORMAT_H_
