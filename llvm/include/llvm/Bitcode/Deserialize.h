//=- Deserialize.h - Generic Object Deserialization from Bitcode --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for generic object deserialization from
// LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_SERIALIZE_INPUT
#define LLVM_BITCODE_SERIALIZE_INPUT

#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Bitcode/Serialization.h"
#include <vector>

namespace llvm {
  
class Deserializer {  
  BitstreamReader& Stream;
  SmallVector<uint64_t,10> Record;
  unsigned RecIdx;
public:
  Deserializer(BitstreamReader& stream);
  ~Deserializer();
  
  template <typename T>
  inline T& Read(T& X) {
    SerializeTrait<T>::Read(*this,X);
    return X;
  }
  
  template <typename T>
  inline T* Materialize() {
    return SerializeTrait<T>::Materialize(*this);
  }
    
  uint64_t ReadInt();
  bool ReadBool() { return ReadInt() ? true : false; }
  
  // FIXME: Substitute a better implementation which calculates the minimum
  // number of bits needed to serialize the enum.
  template <typename EnumT>
  EnumT ReadEnum(unsigned MinVal, unsigned MaxVal) { 
    return static_cast<EnumT>(ReadInt(32));
  }
  
  char* ReadCStr(char* cstr = NULL, unsigned MaxLen=0, bool isNullTerm=true);
  void ReadCStr(std::vector<char>& buff, bool isNullTerm=false);
  
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
  
} // end namespace llvm

#endif
