//===-- llvm/Target/TargetData.h - Data size & alignment info ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines target properties related to datatype size/offset/alignment
// information.  It uses lazy annotations to cache information about how 
// structure types are laid out and used.
//
// This structure should be created once, filled in if the defaults are not
// correct and then passed around by const&.  None of the members functions
// require modification to the object.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETDATA_H
#define LLVM_TARGET_TARGETDATA_H

#include "llvm/Pass.h"
#include "Support/DataTypes.h"
#include <vector>
#include <string>

namespace llvm {

class Value;
class Type;
class StructType;
class StructLayout;

class TargetData : public ImmutablePass {
  bool          LittleEndian;          // Defaults to false
  unsigned char ByteAlignment;         // Defaults to 1 bytes
  unsigned char ShortAlignment;        // Defaults to 2 bytes
  unsigned char IntAlignment;          // Defaults to 4 bytes
  unsigned char LongAlignment;         // Defaults to 8 bytes
  unsigned char FloatAlignment;        // Defaults to 4 bytes
  unsigned char DoubleAlignment;       // Defaults to 8 bytes
  unsigned char PointerSize;           // Defaults to 8 bytes
  unsigned char PointerAlignment;      // Defaults to 8 bytes
public:
  TargetData(const std::string &TargetName = "",
             bool LittleEndian = false,
             unsigned char PtrSize = 8,
	     unsigned char PtrAl = 8, unsigned char DoubleAl = 8,
	     unsigned char FloatAl = 4, unsigned char LongAl = 8, 
	     unsigned char IntAl = 4, unsigned char ShortAl = 2,
	     unsigned char ByteAl = 1);
  TargetData(const std::string &ToolName, const Module *M);
  ~TargetData();  // Not virtual, do not subclass this class

  /// Target endianness...
  bool          isLittleEndian()      const { return     LittleEndian; }
  bool          isBigEndian()         const { return    !LittleEndian; }

  /// Target alignment constraints
  unsigned char getByteAlignment()    const { return    ByteAlignment; }
  unsigned char getShortAlignment()   const { return   ShortAlignment; }
  unsigned char getIntAlignment()     const { return     IntAlignment; }
  unsigned char getLongAlignment()    const { return    LongAlignment; }
  unsigned char getFloatAlignment()   const { return   FloatAlignment; }
  unsigned char getDoubleAlignment()  const { return  DoubleAlignment; }
  unsigned char getPointerAlignment() const { return PointerAlignment; }
  unsigned char getPointerSize()      const { return      PointerSize; }

  /// getTypeSize - Return the number of bytes necessary to hold the specified
  /// type
  uint64_t getTypeSize(const Type *Ty) const;

  /// getTypeAlignment - Return the minimum required alignment for the specified
  /// type
  unsigned char getTypeAlignment(const Type *Ty) const;

  /// getIntPtrType - Return an unsigned integer type that is the same size or
  /// greater to the host pointer size.
  const Type *getIntPtrType() const;

  /// getIndexOffset - return the offset from the beginning of the type for the
  /// specified indices.  This is used to implement getelementptr.
  ///
  uint64_t getIndexedOffset(const Type *Ty, 
                            const std::vector<Value*> &Indices) const;
  
  const StructLayout *getStructLayout(const StructType *Ty) const;
};

// This object is used to lazily calculate structure layout information for a
// target machine, based on the TargetData structure.
//
struct StructLayout {
  std::vector<uint64_t> MemberOffsets;
  uint64_t StructSize;
  unsigned StructAlignment;
private:
  friend class TargetData;   // Only TargetData can create this class
  StructLayout(const StructType *ST, const TargetData &TD);
};

} // End llvm namespace

#endif
