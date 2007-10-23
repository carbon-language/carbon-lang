 //=- Serialization.h - Generic Object Serialization to Bitcode ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for generic object serialization to
// LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_SERIALIZE
#define LLVM_BITCODE_SERIALIZE

#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

namespace llvm {

template <typename T> struct SerializeTrait;
  
class Serializer {
  BitstreamWriter& Stream;
  SmallVector<uint64_t,10> Record;
  bool inBlock;
public:
  Serializer(BitstreamWriter& stream, unsigned BlockID = 0);
  ~Serializer();
  
  template <typename T>
  inline void Emit(const T& X) { SerializeTrait<T>::Serialize(*this,X); }
    
  void EmitInt(unsigned X, unsigned bits);
  
  // FIXME: Substitute a better implementation which calculates the minimum
  // number of bits needed to serialize the enum.
  void EmitEnum(unsigned X, unsigned MinVal, unsigned MaxVal) { EmitInt(X,32); }
  
  void EmitCString(const char* cstr);

  void Flush() { if (inRecord()) EmitRecord(); }
  
private:
  void EmitRecord();
  inline bool inRecord() { return Record.size() > 0; }  
};
  
  
class Deserializer {
  BitstreamReader& Stream;
  SmallVector<uint64_t,10> Record;
  unsigned RecIdx;
public:
  Deserializer(BitstreamReader& stream);
  ~Deserializer();

  template <typename T>
  inline T& Read(T& X) { SerializeTrait<T>::Deserialize(*this,X); return X; }

  template <typename T>
  inline T* Materialize() {
    T* X = SerializeTrait<T>::Instantiate();
    Read(*X);
    return X;
  }
  
  uint64_t ReadInt(unsigned bits = 32);
  bool ReadBool() { return ReadInt(1); }
  
  // FIXME: Substitute a better implementation which calculates the minimum
  // number of bits needed to serialize the enum.
  template <typename EnumT>
  EnumT ReadEnum(unsigned MinVal, unsigned MaxVal) { 
    return static_cast<EnumT>(ReadInt(32));
  }
  
  char* ReadCString(char* cstr = NULL, unsigned MaxLen=0, bool isNullTerm=true);
  void ReadCString(std::vector<char>& buff, bool isNullTerm=false);
  
private:
  void ReadRecord();

  inline bool inRecord() { 
    if (Record.size() > 0) {
      if (RecIdx >= Record.size()) {
        RecIdx = 0;
        Record.clear();
        return false;
      }
      else return true;
    }
    else return false;
  }
};
 

template <typename uintty, unsigned Bits> 
struct SerializeIntTrait {
  static inline void Serialize(Serializer& S, uintty X) {
    S.EmitInt(X,Bits);
  }
  
  static inline void Deserialize(Deserializer& S, uintty& X) {
    X = (uintty) S.ReadInt(Bits);
  }
};
  
template <> struct SerializeTrait<bool>
  : public SerializeIntTrait<bool,1> {};

template <> struct SerializeTrait<char>
  : public SerializeIntTrait<char,8> {};
  
template <> struct SerializeTrait<short>
  : public SerializeIntTrait<short,16> {};

template <> struct SerializeTrait<unsigned>
  : public SerializeIntTrait<unsigned,32> {};

  
  
} // end namespace llvm
#endif
