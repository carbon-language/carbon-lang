//===--- Descriptor.cpp - Types for the constexpr VM ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Descriptor.h"
#include "Pointer.h"
#include "PrimType.h"
#include "Record.h"

using namespace clang;
using namespace clang::interp;

template <typename T>
static void ctorTy(Block *, char *Ptr, bool, bool, bool, Descriptor *) {
  new (Ptr) T();
}

template <typename T> static void dtorTy(Block *, char *Ptr, Descriptor *) {
  reinterpret_cast<T *>(Ptr)->~T();
}

template <typename T>
static void moveTy(Block *, char *Src, char *Dst, Descriptor *) {
  auto *SrcPtr = reinterpret_cast<T *>(Src);
  auto *DstPtr = reinterpret_cast<T *>(Dst);
  new (DstPtr) T(std::move(*SrcPtr));
}

template <typename T>
static void ctorArrayTy(Block *, char *Ptr, bool, bool, bool, Descriptor *D) {
  for (unsigned I = 0, NE = D->getNumElems(); I < NE; ++I) {
    new (&reinterpret_cast<T *>(Ptr)[I]) T();
  }
}

template <typename T>
static void dtorArrayTy(Block *, char *Ptr, Descriptor *D) {
  for (unsigned I = 0, NE = D->getNumElems(); I < NE; ++I) {
    reinterpret_cast<T *>(Ptr)[I].~T();
  }
}

template <typename T>
static void moveArrayTy(Block *, char *Src, char *Dst, Descriptor *D) {
  for (unsigned I = 0, NE = D->getNumElems(); I < NE; ++I) {
    auto *SrcPtr = &reinterpret_cast<T *>(Src)[I];
    auto *DstPtr = &reinterpret_cast<T *>(Dst)[I];
    new (DstPtr) T(std::move(*SrcPtr));
  }
}

static void ctorArrayDesc(Block *B, char *Ptr, bool IsConst, bool IsMutable,
                          bool IsActive, Descriptor *D) {
  const unsigned NumElems = D->getNumElems();
  const unsigned ElemSize =
      D->ElemDesc->getAllocSize() + sizeof(InlineDescriptor);

  unsigned ElemOffset = 0;
  for (unsigned I = 0; I < NumElems; ++I, ElemOffset += ElemSize) {
    auto *ElemPtr = Ptr + ElemOffset;
    auto *Desc = reinterpret_cast<InlineDescriptor *>(ElemPtr);
    auto *ElemLoc = reinterpret_cast<char *>(Desc + 1);
    auto *SD = D->ElemDesc;

    Desc->Offset = ElemOffset + sizeof(InlineDescriptor);
    Desc->Desc = SD;
    Desc->IsInitialized = true;
    Desc->IsBase = false;
    Desc->IsActive = IsActive;
    Desc->IsConst = IsConst || D->IsConst;
    Desc->IsMutable = IsMutable || D->IsMutable;
    if (auto Fn = D->ElemDesc->CtorFn)
      Fn(B, ElemLoc, Desc->IsConst, Desc->IsMutable, IsActive, D->ElemDesc);
  }
}

static void dtorArrayDesc(Block *B, char *Ptr, Descriptor *D) {
  const unsigned NumElems = D->getNumElems();
  const unsigned ElemSize =
      D->ElemDesc->getAllocSize() + sizeof(InlineDescriptor);

  unsigned ElemOffset = 0;
  for (unsigned I = 0; I < NumElems; ++I, ElemOffset += ElemSize) {
    auto *ElemPtr = Ptr + ElemOffset;
    auto *Desc = reinterpret_cast<InlineDescriptor *>(ElemPtr);
    auto *ElemLoc = reinterpret_cast<char *>(Desc + 1);
    if (auto Fn = D->ElemDesc->DtorFn)
      Fn(B, ElemLoc, D->ElemDesc);
  }
}

static void moveArrayDesc(Block *B, char *Src, char *Dst, Descriptor *D) {
  const unsigned NumElems = D->getNumElems();
  const unsigned ElemSize =
      D->ElemDesc->getAllocSize() + sizeof(InlineDescriptor);

  unsigned ElemOffset = 0;
  for (unsigned I = 0; I < NumElems; ++I, ElemOffset += ElemSize) {
    auto *SrcPtr = Src + ElemOffset;
    auto *DstPtr = Dst + ElemOffset;

    auto *SrcDesc = reinterpret_cast<InlineDescriptor *>(SrcPtr);
    auto *SrcElemLoc = reinterpret_cast<char *>(SrcDesc + 1);
    auto *DstDesc = reinterpret_cast<InlineDescriptor *>(DstPtr);
    auto *DstElemLoc = reinterpret_cast<char *>(DstDesc + 1);

    *DstDesc = *SrcDesc;
    if (auto Fn = D->ElemDesc->MoveFn)
      Fn(B, SrcElemLoc, DstElemLoc, D->ElemDesc);
  }
}

static void ctorRecord(Block *B, char *Ptr, bool IsConst, bool IsMutable,
                       bool IsActive, Descriptor *D) {
  const bool IsUnion = D->ElemRecord->isUnion();
  auto CtorSub = [=](unsigned SubOff, Descriptor *F, bool IsBase) {
    auto *Desc = reinterpret_cast<InlineDescriptor *>(Ptr + SubOff) - 1;
    Desc->Offset = SubOff;
    Desc->Desc = F;
    Desc->IsInitialized = (B->isStatic() || F->IsArray) && !IsBase;
    Desc->IsBase = IsBase;
    Desc->IsActive = IsActive && !IsUnion;
    Desc->IsConst = IsConst || F->IsConst;
    Desc->IsMutable = IsMutable || F->IsMutable;
    if (auto Fn = F->CtorFn)
      Fn(B, Ptr + SubOff, Desc->IsConst, Desc->IsMutable, Desc->IsActive, F);
  };
  for (const auto &B : D->ElemRecord->bases())
    CtorSub(B.Offset, B.Desc, /*isBase=*/true);
  for (const auto &F : D->ElemRecord->fields())
    CtorSub(F.Offset, F.Desc, /*isBase=*/false);
  for (const auto &V : D->ElemRecord->virtual_bases())
    CtorSub(V.Offset, V.Desc, /*isBase=*/true);
}

static void dtorRecord(Block *B, char *Ptr, Descriptor *D) {
  auto DtorSub = [=](unsigned SubOff, Descriptor *F) {
    if (auto Fn = F->DtorFn)
      Fn(B, Ptr + SubOff, F);
  };
  for (const auto &F : D->ElemRecord->bases())
    DtorSub(F.Offset, F.Desc);
  for (const auto &F : D->ElemRecord->fields())
    DtorSub(F.Offset, F.Desc);
  for (const auto &F : D->ElemRecord->virtual_bases())
    DtorSub(F.Offset, F.Desc);
}

static void moveRecord(Block *B, char *Src, char *Dst, Descriptor *D) {
  for (const auto &F : D->ElemRecord->fields()) {
    auto FieldOff = F.Offset;
    auto FieldDesc = F.Desc;

    *(reinterpret_cast<Descriptor **>(Dst + FieldOff) - 1) = FieldDesc;
    if (auto Fn = FieldDesc->MoveFn)
      Fn(B, Src + FieldOff, Dst + FieldOff, FieldDesc);
  }
}

static BlockCtorFn getCtorPrim(PrimType Type) {
  COMPOSITE_TYPE_SWITCH(Type, return ctorTy<T>, return nullptr);
}

static BlockDtorFn getDtorPrim(PrimType Type) {
  COMPOSITE_TYPE_SWITCH(Type, return dtorTy<T>, return nullptr);
}

static BlockMoveFn getMovePrim(PrimType Type) {
  COMPOSITE_TYPE_SWITCH(Type, return moveTy<T>, return nullptr);
}

static BlockCtorFn getCtorArrayPrim(PrimType Type) {
  COMPOSITE_TYPE_SWITCH(Type, return ctorArrayTy<T>, return nullptr);
}

static BlockDtorFn getDtorArrayPrim(PrimType Type) {
  COMPOSITE_TYPE_SWITCH(Type, return dtorArrayTy<T>, return nullptr);
}

static BlockMoveFn getMoveArrayPrim(PrimType Type) {
  COMPOSITE_TYPE_SWITCH(Type, return moveArrayTy<T>, return nullptr);
}

Descriptor::Descriptor(const DeclTy &D, PrimType Type, bool IsConst,
                       bool IsTemporary, bool IsMutable)
    : Source(D), ElemSize(primSize(Type)), Size(ElemSize), AllocSize(Size),
      IsConst(IsConst), IsMutable(IsMutable), IsTemporary(IsTemporary),
      CtorFn(getCtorPrim(Type)), DtorFn(getDtorPrim(Type)),
      MoveFn(getMovePrim(Type)) {
  assert(Source && "Missing source");
}

Descriptor::Descriptor(const DeclTy &D, PrimType Type, size_t NumElems,
                       bool IsConst, bool IsTemporary, bool IsMutable)
    : Source(D), ElemSize(primSize(Type)), Size(ElemSize * NumElems),
      AllocSize(align(Size) + sizeof(InitMap *)), IsConst(IsConst),
      IsMutable(IsMutable), IsTemporary(IsTemporary), IsArray(true),
      CtorFn(getCtorArrayPrim(Type)), DtorFn(getDtorArrayPrim(Type)),
      MoveFn(getMoveArrayPrim(Type)) {
  assert(Source && "Missing source");
}

Descriptor::Descriptor(const DeclTy &D, PrimType Type, bool IsTemporary,
                       UnknownSize)
    : Source(D), ElemSize(primSize(Type)), Size(UnknownSizeMark),
      AllocSize(alignof(void *)), IsConst(true), IsMutable(false),
      IsTemporary(IsTemporary), IsArray(true), CtorFn(getCtorArrayPrim(Type)),
      DtorFn(getDtorArrayPrim(Type)), MoveFn(getMoveArrayPrim(Type)) {
  assert(Source && "Missing source");
}

Descriptor::Descriptor(const DeclTy &D, Descriptor *Elem, unsigned NumElems,
                       bool IsConst, bool IsTemporary, bool IsMutable)
    : Source(D), ElemSize(Elem->getAllocSize() + sizeof(InlineDescriptor)),
      Size(ElemSize * NumElems),
      AllocSize(std::max<size_t>(alignof(void *), Size)), ElemDesc(Elem),
      IsConst(IsConst), IsMutable(IsMutable), IsTemporary(IsTemporary),
      IsArray(true), CtorFn(ctorArrayDesc), DtorFn(dtorArrayDesc),
      MoveFn(moveArrayDesc) {
  assert(Source && "Missing source");
}

Descriptor::Descriptor(const DeclTy &D, Descriptor *Elem, bool IsTemporary,
                       UnknownSize)
    : Source(D), ElemSize(Elem->getAllocSize() + sizeof(InlineDescriptor)),
      Size(UnknownSizeMark), AllocSize(alignof(void *)), ElemDesc(Elem),
      IsConst(true), IsMutable(false), IsTemporary(IsTemporary), IsArray(true),
      CtorFn(ctorArrayDesc), DtorFn(dtorArrayDesc), MoveFn(moveArrayDesc) {
  assert(Source && "Missing source");
}

Descriptor::Descriptor(const DeclTy &D, Record *R, bool IsConst,
                       bool IsTemporary, bool IsMutable)
    : Source(D), ElemSize(std::max<size_t>(alignof(void *), R->getFullSize())),
      Size(ElemSize), AllocSize(Size), ElemRecord(R), IsConst(IsConst),
      IsMutable(IsMutable), IsTemporary(IsTemporary), CtorFn(ctorRecord),
      DtorFn(dtorRecord), MoveFn(moveRecord) {
  assert(Source && "Missing source");
}

QualType Descriptor::getType() const {
  if (auto *E = asExpr())
    return E->getType();
  if (auto *D = asValueDecl())
    return D->getType();
  llvm_unreachable("Invalid descriptor type");
}

SourceLocation Descriptor::getLocation() const {
  if (auto *D = Source.dyn_cast<const Decl *>())
    return D->getLocation();
  if (auto *E = Source.dyn_cast<const Expr *>())
    return E->getExprLoc();
  llvm_unreachable("Invalid descriptor type");
}

InitMap::InitMap(unsigned N) : UninitFields(N) {
  for (unsigned I = 0; I < N / PER_FIELD; ++I) {
    data()[I] = 0;
  }
}

InitMap::T *InitMap::data() {
  auto *Start = reinterpret_cast<char *>(this) + align(sizeof(InitMap));
  return reinterpret_cast<T *>(Start);
}

bool InitMap::initialize(unsigned I) {
  unsigned Bucket = I / PER_FIELD;
  unsigned Mask = 1ull << static_cast<uint64_t>(I % PER_FIELD);
  if (!(data()[Bucket] & Mask)) {
    data()[Bucket] |= Mask;
    UninitFields -= 1;
  }
  return UninitFields == 0;
}

bool InitMap::isInitialized(unsigned I) {
  unsigned Bucket = I / PER_FIELD;
  unsigned Mask = 1ull << static_cast<uint64_t>(I % PER_FIELD);
  return data()[Bucket] & Mask;
}

InitMap *InitMap::allocate(unsigned N) {
  const size_t NumFields = ((N + PER_FIELD - 1) / PER_FIELD);
  const size_t Size = align(sizeof(InitMap)) + NumFields * PER_FIELD;
  return new (malloc(Size)) InitMap(N);
}
