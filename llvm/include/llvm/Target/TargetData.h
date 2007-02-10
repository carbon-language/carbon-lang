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
#include "llvm/Support/DataTypes.h"
#include <vector>
#include <string>

namespace llvm {

class Value;
class Type;
class StructType;
class StructLayout;
class GlobalVariable;

class TargetData : public ImmutablePass {
  bool          LittleEndian;          // Defaults to false

  // ABI alignments
  unsigned char BoolABIAlignment;       // Defaults to 1 byte
  unsigned char ByteABIAlignment;       // Defaults to 1 byte
  unsigned char ShortABIAlignment;      // Defaults to 2 bytes
  unsigned char IntABIAlignment;        // Defaults to 4 bytes
  unsigned char LongABIAlignment;       // Defaults to 8 bytes
  unsigned char FloatABIAlignment;      // Defaults to 4 bytes
  unsigned char DoubleABIAlignment;     // Defaults to 8 bytes
  unsigned char PointerMemSize;        // Defaults to 8 bytes
  unsigned char PointerABIAlignment;    // Defaults to 8 bytes

  // Preferred stack/global type alignments
  unsigned char BoolPrefAlignment;    // Defaults to BoolABIAlignment
  unsigned char BytePrefAlignment;    // Defaults to ByteABIAlignment
  unsigned char ShortPrefAlignment;   // Defaults to ShortABIAlignment
  unsigned char IntPrefAlignment;     // Defaults to IntABIAlignment
  unsigned char LongPrefAlignment;    // Defaults to LongABIAlignment
  unsigned char FloatPrefAlignment;   // Defaults to FloatABIAlignment
  unsigned char DoublePrefAlignment;  // Defaults to DoubleABIAlignment
  unsigned char PointerPrefAlignment; // Defaults to PointerABIAlignment
  unsigned char AggMinPrefAlignment;  // Defaults to 0 bytes

public:
  /// Default ctor - This has to exist, because this is a pass, but it should
  /// never be used.
  TargetData() {
    assert(0 && "ERROR: Bad TargetData ctor used.  "
           "Tool did not specify a TargetData to use?");
    abort();
  }
    
  /// Constructs a TargetData from a string of the following format:
  /// "E-p:64:64-d:64-f:32-l:64-i:32-s:16-b:8-B:8"
  /// The above string is considered the default, and any values not specified
  /// in the string will be assumed to be as above, with the caveat that unspecified
  /// values are always assumed to be smaller than the size of a pointer.
  TargetData(const std::string &TargetDescription) {
    init(TargetDescription);
  }

  /// Initialize target data from properties stored in the module.
  TargetData(const Module *M);

  TargetData(const TargetData &TD) : 
    ImmutablePass(),
    LittleEndian(TD.isLittleEndian()),
    BoolABIAlignment(TD.getBoolABIAlignment()),
    ByteABIAlignment(TD.getByteABIAlignment()),
    ShortABIAlignment(TD.getShortABIAlignment()),
    IntABIAlignment(TD.getIntABIAlignment()),
    LongABIAlignment(TD.getLongABIAlignment()),
    FloatABIAlignment(TD.getFloatABIAlignment()),
    DoubleABIAlignment(TD.getDoubleABIAlignment()),
    PointerMemSize(TD.getPointerSize()),
    PointerABIAlignment(TD.getPointerABIAlignment()),
    BoolPrefAlignment(TD.getBoolPrefAlignment()),
    BytePrefAlignment(TD.getBytePrefAlignment()),
    ShortPrefAlignment(TD.getShortPrefAlignment()),
    IntPrefAlignment(TD.getIntPrefAlignment()),
    LongPrefAlignment(TD.getLongPrefAlignment()),
    FloatPrefAlignment(TD.getFloatPrefAlignment()),
    DoublePrefAlignment(TD.getDoublePrefAlignment()),
    PointerPrefAlignment(TD.getPointerPrefAlignment()),
    AggMinPrefAlignment(TD.getAggMinPrefAlignment()) {
  }

  ~TargetData();  // Not virtual, do not subclass this class

  /// Parse a target data layout string and initialize TargetData members.
  ///
  /// Parse a target data layout string, initializing the various TargetData
  /// members along the way. A TargetData specification string looks like
  /// "E-p:64:64-d:64-f:32-l:64-i:32-s:16-b:8-B:8" and specifies the
  /// target's endianess, the ABI alignments of various data types and
  /// the size of pointers.
  ///
  /// "-" is used as a separator and ":" separates a token from its argument.
  ///
  /// Alignment is indicated in bits and internally converted to the
  /// appropriate number of bytes.
  ///
  /// The preferred stack/global alignment specifications (":[prefalign]") are
  /// optional and default to the ABI alignment.
  ///
  /// Valid tokens:
  /// <br>
  /// <em>E</em> specifies big endian architecture (1234) [default]<br>
  /// <em>e</em> specifies little endian architecture (4321) <br>
  /// <em>p:[ptr size]:[ptr align]</em> specifies pointer size and alignment
  /// [default = 64:64] <br>
  /// <em>d:[align]:[prefalign]</em> specifies double floating
  /// point alignment [default = 64] <br>
  /// <em>f:[align]:[prefalign]</em> specifies single floating
  /// point alignment [default = 32] <br>
  /// <em>l:[align]:[prefalign]:[globalign[</em> specifies long integer
  /// alignment [default = 64] <br>
  /// <em>i:[align]:[prefalign]</em> specifies integer alignment
  /// [default = 32] <br>
  /// <em>s:[align]:[prefalign]</em> specifies short integer
  /// alignment [default = 16] <br>
  /// <em>b:[align]:[prefalign]</em> specifies byte data type
  /// alignment [default = 8] <br>
  /// <em>B:[align]:[prefalign]</em> specifies boolean data type
  /// alignment [default = 8] <br>
  /// <em>A:[prefalign]</em> specifies an aggregates' minimum alignment
  /// on the stack and when emitted as a global. The default minimum aggregate
  /// alignment defaults to 0, which causes the aggregate's "natural" internal
  /// alignment calculated by llvm to be preferred.
  ///
  /// All other token types are silently ignored.
  void init(const std::string &TargetDescription);
  
  
  /// Target endianness...
  bool          isLittleEndian()       const { return     LittleEndian; }
  bool          isBigEndian()          const { return    !LittleEndian; }

  /// Target boolean alignment
  unsigned char getBoolABIAlignment()    const { return    BoolABIAlignment; }
  /// Target byte alignment
  unsigned char getByteABIAlignment()    const { return    ByteABIAlignment; }
  /// Target short alignment
  unsigned char getShortABIAlignment()   const { return   ShortABIAlignment; }
  /// Target integer alignment
  unsigned char getIntABIAlignment()     const { return     IntABIAlignment; }
  /// Target long alignment
  unsigned char getLongABIAlignment()    const { return    LongABIAlignment; }
  /// Target single precision float alignment
  unsigned char getFloatABIAlignment()   const { return   FloatABIAlignment; }
  /// Target double precision float alignment
  unsigned char getDoubleABIAlignment()  const { return  DoubleABIAlignment; }
  /// Target pointer alignment
  unsigned char getPointerABIAlignment() const { return PointerABIAlignment; }
  /// Target pointer size
  unsigned char getPointerSize()         const { return      PointerMemSize; }
  /// Target pointer size, in bits
  unsigned char getPointerSizeInBits()   const { return    8*PointerMemSize; }

  /// Return target's alignment for booleans on stack
  unsigned char getBoolPrefAlignment() const {
    return BoolPrefAlignment;
  }
  /// Return target's alignment for integers on stack
  unsigned char getBytePrefAlignment() const {
    return BytePrefAlignment;
  }
  /// Return target's alignment for shorts on stack
  unsigned char getShortPrefAlignment() const {
    return ShortPrefAlignment;
  }
  /// Return target's alignment for integers on stack
  unsigned char getIntPrefAlignment()     const {
    return IntPrefAlignment;
  }
  /// Return target's alignment for longs on stack
  unsigned char getLongPrefAlignment() const {
    return LongPrefAlignment;
  }
  /// Return target's alignment for single precision floats on stack
  unsigned char getFloatPrefAlignment() const {
    return FloatPrefAlignment;
  }
  /// Return target's alignment for double preceision floats on stack
  unsigned char getDoublePrefAlignment()  const {
    return DoublePrefAlignment;
  }
  /// Return target's alignment for stack-based pointers
  unsigned char getPointerPrefAlignment() const {
    return PointerPrefAlignment;
  }
  /// Return target's alignment for stack-based structures
  unsigned char getAggMinPrefAlignment() const {
    return AggMinPrefAlignment;
  }

  /// getStringRepresentation - Return the string representation of the
  /// TargetData.  This representation is in the same format accepted by the
  /// string constructor above.
  std::string getStringRepresentation() const;

  /// getTypeSize - Return the number of bytes necessary to hold the specified
  /// type.
  ///
  uint64_t getTypeSize(const Type *Ty) const;

  /// getTypeSizeInBits - Return the number of bytes necessary to hold the
  /// specified type.
  uint64_t getTypeSizeInBits(const Type* Ty) const;

  /// getTypeAlignmentABI - Return the minimum ABI-required alignment for the
  /// specified type.
  unsigned char getTypeAlignmentABI(const Type *Ty) const;

  /// getTypeAlignmentPref - Return the preferred stack/global alignment for
  /// the specified type.
  unsigned char getTypeAlignmentPref(const Type *Ty) const;

  /// getPreferredTypeAlignmentShift - Return the preferred alignment for the
  /// specified type, returned as log2 of the value (a shift amount).
  ///
  unsigned char getPreferredTypeAlignmentShift(const Type *Ty) const;

  /// getIntPtrType - Return an unsigned integer type that is the same size or
  /// greater to the host pointer size.
  ///
  const Type *getIntPtrType() const;

  /// getIndexOffset - return the offset from the beginning of the type for the
  /// specified indices.  This is used to implement getelementptr.
  ///
  uint64_t getIndexedOffset(const Type *Ty,
                            Value* const* Indices, unsigned NumIndices) const;
  
  uint64_t getIndexedOffset(const Type *Ty,
                            const std::vector<Value*> &Indices) const {
    return getIndexedOffset(Ty, &Indices[0], Indices.size());
  }

  /// getStructLayout - Return a StructLayout object, indicating the alignment
  /// of the struct, its size, and the offsets of its fields.  Note that this
  /// information is lazily cached.
  const StructLayout *getStructLayout(const StructType *Ty) const;
  
  /// InvalidateStructLayoutInfo - TargetData speculatively caches StructLayout
  /// objects.  If a TargetData object is alive when types are being refined and
  /// removed, this method must be called whenever a StructType is removed to
  /// avoid a dangling pointer in this cache.
  void InvalidateStructLayoutInfo(const StructType *Ty) const;

  /// getPreferredAlignmentLog - Return the preferred alignment of the
  /// specified global, returned in log form.  This includes an explicitly
  /// requested alignment (if the global has one).
  unsigned getPreferredAlignmentLog(const GlobalVariable *GV) const;
};

/// StructLayout - used to lazily calculate structure layout information for a
/// target machine, based on the TargetData structure.
///
class StructLayout {
  std::vector<uint64_t> MemberOffsets;
public:
  unsigned StructAlignment;
  uint64_t StructSize;

  /// getElementContainingOffset - Given a valid offset into the structure,
  /// return the structure index that contains it.
  ///
  unsigned getElementContainingOffset(uint64_t Offset) const;

  uint64_t getElementOffset(unsigned Idx) const {
    assert(Idx < MemberOffsets.size() && "Invalid element idx!");
    return MemberOffsets[Idx];
  }
  
private:
  friend class TargetData;   // Only TargetData can create this class
  StructLayout(const StructType *ST, const TargetData &TD);
};

} // End llvm namespace

#endif
