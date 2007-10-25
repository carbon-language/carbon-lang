//==- Serialize.h - Generic Object Serialization to Bitcode -------*- C++ -*-=//
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

#ifndef LLVM_BITCODE_SERIALIZE_OUTPUT
#define LLVM_BITCODE_SERIALIZE_OUTPUT

#include "llvm/Bitcode/Serialization.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {

class Serializer {
  BitstreamWriter& Stream;
  SmallVector<uint64_t,10> Record;
  bool inBlock;
  
  typedef DenseMap<void*,unsigned> MapTy;
  MapTy PtrMap;
  
public:
  Serializer(BitstreamWriter& stream, unsigned BlockID = 0);
  ~Serializer();
  
  template <typename T>
  inline void Emit(const T& X) { SerializeTrait<T>::Emit(*this,X); }
  
  void EmitInt(unsigned X);
  void EmitBool(bool X) { EmitInt(X); }
  void EmitCStr(const char* beg, const char* end);
  void EmitCStr(const char* cstr);
  
  void EmitPtr(void* ptr) { EmitInt(getPtrId(ptr)); }
  
  template <typename T>
  void EmitOwnedPtr(T* ptr) {
    EmitPtr(ptr);
    SerializeTrait<T>::Emit(*this,*ptr);    
  }

  void Flush() { if (inRecord()) EmitRecord(); }
  
private:
  void EmitRecord();
  inline bool inRecord() { return Record.size() > 0; }
  unsigned getPtrId(void* ptr);
};

} // end namespace llvm
#endif
