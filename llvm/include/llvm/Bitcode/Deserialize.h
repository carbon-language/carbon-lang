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
  
  struct BPNode {
    BPNode* Next;
    uintptr_t& PtrRef;
    
    BPNode(BPNode* n, uintptr_t& pref) 
      : Next(n), PtrRef(pref) {
        PtrRef = 0;
      }
  };
  
  struct BPEntry { 
    union { BPNode* Head; void* Ptr; };
    
    BPEntry() : Head(NULL) {}
    
    static inline bool isPod() { return true; }
    
    void SetPtr(BPNode*& FreeList, void* P);    
  };  
  
  class BPKey {
    unsigned Raw;
    
  public:
    BPKey(unsigned PtrId) : Raw(PtrId << 1) { assert (PtrId > 0); }
    
    void MarkFinal() { Raw |= 0x1; }
    bool hasFinalPtr() const { return Raw & 0x1 ? true : false; }
    unsigned getID() const { return Raw >> 1; }
    
    static inline BPKey getEmptyKey() { return 0; }
    static inline BPKey getTombstoneKey() { return 1; }
    static inline unsigned getHashValue(const BPKey& K) { return K.Raw & ~0x1; }

    static bool isEqual(const BPKey& K1, const BPKey& K2) {
      return (K1.Raw ^ K2.Raw) & ~0x1 ? false : true;
    }
    
    static bool isPod() { return true; }
  };
  
  typedef llvm::DenseMap<BPKey,BPEntry,BPKey,BPEntry> MapTy;

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

  void RegisterPtr(unsigned PtrId, const void* Ptr);
  
  void RegisterPtr(const void* Ptr) {
    RegisterPtr(ReadInt(),Ptr);
  }

private:
  void ReadRecord();  
  bool inRecord();
  uintptr_t ReadInternalRefPtr();
  
  static inline bool HasFinalPtr(MapTy::value_type& V) {
    return V.first.hasFinalPtr();
  }
  
  static inline uintptr_t GetFinalPtr(MapTy::value_type& V) {
    return reinterpret_cast<uintptr_t>(V.second.Ptr);
  }
  
  static inline BPNode* GetBPNode(MapTy::value_type& V) {
    return V.second.Head;
  }
    
  static inline void SetBPNode(MapTy::value_type& V, BPNode* N) {
    V.second.Head = N;
  }
  
  void SetPtr(MapTy::value_type& V, const void* P) {
    V.first.MarkFinal();
    V.second.SetPtr(FreeList,const_cast<void*>(P));
  }
};
    
} // end namespace llvm

#endif
