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

  //===----------------------------------------------------------===//
  // Internal type definitions.
  //===----------------------------------------------------------===//

  struct PtrIdInfo {
    static inline unsigned getEmptyKey() { return ~((unsigned) 0x0); }
    static inline unsigned getTombstoneKey() { return getEmptyKey()-1; }
    static inline unsigned getHashValue(unsigned X) { return X; }
    static inline bool isEqual(unsigned X, unsigned Y) { return X == Y; }
    static inline bool isPod() { return true; }
  };
  
  struct BPNode {
    BPNode* Next;
    uintptr_t& PtrRef;
    BPNode(BPNode* n, uintptr_t& pref) 
      : Next(n), PtrRef(pref) {
        PtrRef = 0;
      }
  };
  
  class BPatchEntry {
    uintptr_t Ptr;
  public:
    
    BPatchEntry() : Ptr(0x1) {}
      
    BPatchEntry(void* P) : Ptr(reinterpret_cast<uintptr_t>(P)) {}

    bool hasFinalPtr() const { return Ptr & 0x1 ? false : true; }
    void setFinalPtr(BPNode*& FreeList, void* P);

    BPNode* getBPNode() const {
      assert (!hasFinalPtr());
      return reinterpret_cast<BPNode*>(Ptr & ~0x1);
    }
    
    void setBPNode(BPNode* N) {
      assert (!hasFinalPtr());
      Ptr = reinterpret_cast<uintptr_t>(N) | 0x1;
    }
    
    uintptr_t getFinalPtr() const {
      assert (!(Ptr & 0x1) && "Backpatch pointer not yet deserialized.");
      return Ptr;
    }    

    static inline bool isPod() { return true; }
  };

  typedef llvm::DenseMap<unsigned,BPatchEntry,PtrIdInfo,BPatchEntry> MapTy;

  //===----------------------------------------------------------===//
  // Internal data members.
  //===----------------------------------------------------------===//
  
  BitstreamReader& Stream;
  SmallVector<uint64_t,10> Record;
  unsigned RecIdx;
  BumpPtrAllocator Allocator;
  BPNode* FreeList;
  MapTy BPatchMap;  
  
  //===----------------------------------------------------------===//
  // Public Interface.
  //===----------------------------------------------------------===//
  
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
  
  template <typename T>
  void ReadPtr(T*& PtrRef) {
    ReadUIntPtr(reinterpret_cast<uintptr_t&>(PtrRef));
  }
  
  template <typename T>
  void ReadPtr(const T*& PtrRef) {
    ReadPtr(const_cast<T*&>(PtrRef));
  }            

  void ReadUIntPtr(uintptr_t& PtrRef);
  
  template <typename T>
  T& ReadRef() {
    T* p = reinterpret_cast<T*>(ReadInternalRefPtr());
    return *p;
  }

  
  void RegisterPtr(unsigned PtrId, void* Ptr);

private:
  void ReadRecord();  
  bool inRecord();
  uintptr_t ReadInternalRefPtr();
};
    
} // end namespace llvm

#endif
