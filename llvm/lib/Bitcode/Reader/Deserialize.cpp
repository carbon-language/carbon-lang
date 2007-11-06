//==- Deserialize.cpp - Generic Object Serialization to Bitcode --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the internal methods used for object serialization.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/Deserialize.h"

#ifdef DEBUG_BACKPATCH
#include "llvm/Support/Streams.h"
#endif

using namespace llvm;

Deserializer::Deserializer(BitstreamReader& stream)
  : Stream(stream), RecIdx(0), FreeList(NULL) {
}

Deserializer::~Deserializer() {
  assert (RecIdx >= Record.size() && 
          "Still scanning bitcode record when deserialization completed.");
 
#ifdef NDEBUG
  for (MapTy::iterator I=BPatchMap.begin(), E=BPatchMap.end(); I!=E; ++I)
    assert (I->first.hasFinalPtr() &&
            "Some pointers were not backpatched.");
#endif
}


bool Deserializer::inRecord() {
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

void Deserializer::ReadRecord() {
  // FIXME: Check if we haven't run off the edge of the stream.
  // FIXME: Handle abbreviations.

  assert (Record.size() == 0);
  
  unsigned Code;

  while (true) {
    
    if (Stream.AtEndOfStream())
      return;
    
    Code = Stream.ReadCode();
  
    if (Code == bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      unsigned id = Stream.ReadSubBlockID();
      Stream.EnterSubBlock(id);
      continue;
    }

    if (Code == bitc::END_BLOCK) {
      bool x = Stream.ReadBlockEnd();
      assert (!x && "Error at block end.");
      continue;
    }
    
    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }
    
    break;
  }
  
  assert (Record.size() == 0);  
  Stream.ReadRecord(Code,Record);  
  assert (Record.size() > 0 || Stream.AtEndOfStream());
}

bool Deserializer::AtEnd() {
  if (inRecord())
    return false;
  
  ReadRecord();
  
  return Stream.AtEndOfStream();
}

uint64_t Deserializer::ReadInt() {
  // FIXME: Any error recovery/handling with incomplete or bad files?
  if (!inRecord())
    ReadRecord();

  return Record[RecIdx++];
}

char* Deserializer::ReadCStr(char* cstr, unsigned MaxLen, bool isNullTerm) {
  if (cstr == NULL)
    MaxLen = 0; // Zero this just in case someone does something funny.
  
  unsigned len = ReadInt();

  assert (MaxLen == 0 || (len + (isNullTerm ? 1 : 0)) <= MaxLen);

  if (!cstr)
    cstr = new char[len + (isNullTerm ? 1 : 0)];
  
  assert (cstr != NULL);
  
  for (unsigned i = 0; i < len; ++i)
    cstr[i] = (char) ReadInt();
  
  if (isNullTerm)
    cstr[len+1] = '\0';
  
  return cstr;
}

void Deserializer::ReadCStr(std::vector<char>& buff, bool isNullTerm) {
  unsigned len = ReadInt();

  buff.clear();  
  buff.reserve(len);
  
  for (unsigned i = 0; i < len; ++i)
    buff.push_back((char) ReadInt());
  
  if (isNullTerm)
    buff.push_back('\0');
}

void Deserializer::RegisterPtr(unsigned PtrId, const void* Ptr) {
  MapTy::value_type& E = BPatchMap.FindAndConstruct(BPKey(PtrId));
  
  assert (!HasFinalPtr(E) && "Pointer already registered.");

#ifdef DEBUG_BACKPATCH
  llvm::cerr << "RegisterPtr: " << PtrId << " => " << Ptr << "\n";
#endif 
  
  SetPtr(E,Ptr);
}

void Deserializer::ReadUIntPtr(uintptr_t& PtrRef, bool AllowBackpatch) {
  unsigned PtrId = ReadInt();
  
  if (PtrId == 0) {
    PtrRef = 0;
    return;
  }
  
#ifdef DEBUG_BACKPATCH
  llvm::cerr << "ReadUintPtr: " << PtrId << "\n";
#endif
  
  MapTy::value_type& E = BPatchMap.FindAndConstruct(BPKey(PtrId));
  
  if (HasFinalPtr(E))
    PtrRef = GetFinalPtr(E);
  else {
    assert (AllowBackpatch &&
            "Client forbids backpatching for this pointer.");
    
    // Register backpatch.  Check the freelist for a BPNode.
    BPNode* N;

    if (FreeList) {
      N = FreeList;
      FreeList = FreeList->Next;
    }
    else // No available BPNode.  Allocate one.
      N = (BPNode*) Allocator.Allocate<BPNode>();
    
    new (N) BPNode(GetBPNode(E),PtrRef);
    SetBPNode(E,N);
  }
}

uintptr_t Deserializer::ReadInternalRefPtr() {
  unsigned PtrId = ReadInt();
  
  assert (PtrId != 0 && "References cannot refer the NULL address.");

  MapTy::value_type& E = BPatchMap.FindAndConstruct(BPKey(PtrId));
  
  assert (HasFinalPtr(E) &&
          "Cannot backpatch references.  Object must be already deserialized.");
  
  return GetFinalPtr(E);
}

void Deserializer::BPEntry::SetPtr(BPNode*& FreeList, void* P) {
  BPNode* Last = NULL;
  
  for (BPNode* N = Head; N != NULL; N=N->Next) {
    Last = N;
    N->PtrRef |= reinterpret_cast<uintptr_t>(P);
  }
  
  if (Last) {
    Last->Next = FreeList;
    FreeList = Head;
  }
  
  Ptr = const_cast<void*>(P);
}


#define INT_READ(TYPE)\
void SerializeTrait<TYPE>::Read(Deserializer& D, TYPE& X) {\
  X = (TYPE) D.ReadInt(); }

INT_READ(bool)
INT_READ(unsigned char)
INT_READ(unsigned short)
INT_READ(unsigned int)
INT_READ(unsigned long)
