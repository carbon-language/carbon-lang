//===--- TypeLocBuilder.h - Type Source Info collector ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines TypeLocBuilder, a class for building TypeLocs
//  bottom-up.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TYPELOCBUILDER_H
#define LLVM_CLANG_AST_TYPELOCBUILDER_H

#include "clang/AST/TypeLoc.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/AST/ASTContext.h"

namespace clang {

class TypeLocBuilder {
  enum { InlineCapacity = 8 * sizeof(SourceLocation) };

  /// The underlying location-data buffer.  Data grows from the end
  /// of the buffer backwards.
  char *Buffer;

  /// The capacity of the current buffer.
  size_t Capacity;

  /// The index of the first occupied byte in the buffer.
  size_t Index;

#ifndef NDEBUG
  /// The last type pushed on this builder.
  QualType LastTy;
#endif
    
  /// The inline buffer.
  char InlineBuffer[InlineCapacity];

 public:
  TypeLocBuilder()
    : Buffer(InlineBuffer), Capacity(InlineCapacity), Index(InlineCapacity)
  {}

  ~TypeLocBuilder() {
    if (Buffer != InlineBuffer)
      delete[] Buffer;
  }

  /// Ensures that this buffer has at least as much capacity as described.
  void reserve(size_t Requested) {
    if (Requested > Capacity)
      // For now, match the request exactly.
      grow(Requested);
  }

  /// Pushes a copy of the given TypeLoc onto this builder.  The builder
  /// must be empty for this to work.
  void pushFullCopy(TypeLoc L) {
#ifndef NDEBUG
    assert(LastTy.isNull() && "pushing copy on non-empty TypeLocBuilder");
    LastTy = L.getNextTypeLoc().getType();
#endif
    assert(Index == Capacity && "pushing copy on non-empty TypeLocBuilder");

    unsigned Size = L.getFullDataSize();
    TypeLoc Copy = pushImpl(L.getType(), Size);
    memcpy(Copy.getOpaqueData(), L.getOpaqueData(), Size);
  }

  /// Pushes space for a typespec TypeLoc.  Invalidates any TypeLocs
  /// previously retrieved from this builder.
  TypeSpecTypeLoc pushTypeSpec(QualType T) {
    size_t LocalSize = TypeSpecTypeLoc::LocalDataSize;
    return cast<TypeSpecTypeLoc>(pushImpl(T, LocalSize));
  }
  

  /// Pushes space for a new TypeLoc of the given type.  Invalidates
  /// any TypeLocs previously retrieved from this builder.
  template <class TyLocType> TyLocType push(QualType T) {
    size_t LocalSize = cast<TyLocType>(TypeLoc(T, 0)).getLocalDataSize();
    return cast<TyLocType>(pushImpl(T, LocalSize));
  }

  /// Creates a TypeSourceInfo for the given type.
  TypeSourceInfo *getTypeSourceInfo(ASTContext& Context, QualType T) {
#ifndef NDEBUG
    assert(T == LastTy && "type doesn't match last type pushed!");
#endif

    size_t FullDataSize = Capacity - Index;
    TypeSourceInfo *DI = Context.CreateTypeSourceInfo(T, FullDataSize);
    memcpy(DI->getTypeLoc().getOpaqueData(), &Buffer[Index], FullDataSize);
    return DI;
  }

private:
  TypeLoc pushImpl(QualType T, size_t LocalSize) {
#ifndef NDEBUG
    QualType TLast = TypeLoc(T, 0).getNextTypeLoc().getType();
    assert(TLast == LastTy &&
           "mismatch between last type and new type's inner type");
    LastTy = T;
#endif

    // If we need to grow, grow by a factor of 2.
    if (LocalSize > Index) {
      size_t RequiredCapacity = Capacity + (LocalSize - Index);
      size_t NewCapacity = Capacity * 2;
      while (RequiredCapacity > NewCapacity)
        NewCapacity *= 2;
      grow(NewCapacity);
    }

    Index -= LocalSize;

    return TypeLoc(T, &Buffer[Index]);
  }

  /// Grow to the given capacity.
  void grow(size_t NewCapacity) {
    assert(NewCapacity > Capacity);

    // Allocate the new buffer and copy the old data into it.
    char *NewBuffer = new char[NewCapacity];
    unsigned NewIndex = Index + NewCapacity - Capacity;
    memcpy(&NewBuffer[NewIndex],
           &Buffer[Index],
           Capacity - Index);

    if (Buffer != InlineBuffer)
      delete[] Buffer;

    Buffer = NewBuffer;
    Capacity = NewCapacity;
    Index = NewIndex;
  }
};

}

#endif
