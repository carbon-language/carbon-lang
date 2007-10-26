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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DataTypes.h"
#include <vector>

namespace llvm {
  
class Deserializer {  
  BitstreamReader& Stream;
  SmallVector<uint64_t,10> Record;
  unsigned RecIdx;
  BumpPtrAllocator Allocator;
  
  struct PtrIdInfo {
    static inline unsigned getEmptyKey() { return ~((unsigned) 0x0); }
    static inline unsigned getTombstoneKey() { return getEmptyKey()-1; }
    static inline unsigned getHashValue(unsigned X) { return X; }
    static inline bool isEqual(unsigned X, unsigned Y) { return X == Y; }
    static inline bool isPod() { return true; }
  };
  
  struct BPatchNode {
    BPatchNode* const Next;
    uintptr_t& PtrRef;
    BPatchNode(BPatchNode* n, void*& pref) 
      : Next(n), PtrRef(reinterpret_cast<uintptr_t&>(pref)) {
        PtrRef = 0;
      }
  };
  
  struct BPatchEntry {
    BPatchNode* Head;
    void* Ptr;    
    BPatchEntry() : Head(NULL), Ptr(NULL) {}
    static inline bool isPod() { return true; }
  };  
  
  typedef llvm::DenseMap<unsigned,BPatchEntry,PtrIdInfo,BPatchEntry> MapTy;

  MapTy BPatchMap;  
  
public:
  Deserializer(BitstreamReader& stream);
  ~Deserializer();

  uint64_t ReadInt();
  bool ReadBool() {
    return ReadInt() ? true : false;
  }

  template <typename T>
  inline T& Read(T& X) {
    SerializeTrait<T>::Read(*this,X);
    return X;
  }
  
  template <typename T>
  inline T ReadVal() {
    return SerializeTrait<T>::ReadVal(*this);
  }

  template <typename T>
  inline T* Materialize() {
    return SerializeTrait<T>::Materialize(*this);
  }
  
  char* ReadCStr(char* cstr = NULL, unsigned MaxLen=0, bool isNullTerm=true);
  void ReadCStr(std::vector<char>& buff, bool isNullTerm=false);

  template <typename T>
  inline T* ReadOwnedPtr() {    
    unsigned PtrId = ReadInt();

    if (PtrId == 0)
      return NULL;
    
    T* x = SerializeTrait<T>::Materialize(*this);
    RegisterPtr(PtrId,x);
    return x;
  }
  
  void ReadPtr(void*& PtrRef);
  void ReadPtr(uintptr_t& PtrRef) { ReadPtr(reinterpret_cast<void*&>(PtrRef)); }
  
  void RegisterPtr(unsigned PtrId, void* Ptr);


  void BackpatchPointers();
private:
  void ReadRecord();  
  bool inRecord();
};
    
} // end namespace llvm

#endif
