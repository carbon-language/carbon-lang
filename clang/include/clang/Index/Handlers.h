//===--- Handlers.h - Interfaces for receiving information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Abstract interfaces for receiving information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_HANDLERS_H
#define LLVM_CLANG_INDEX_HANDLERS_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

namespace idx {
  class Entity;
  class TranslationUnit;
  class TULocation;

/// \brief Abstract interface for receiving Entities.
class EntityHandler {
public:
  typedef Entity receiving_type;

  virtual ~EntityHandler();
  virtual void Handle(Entity Ent) = 0;
};

/// \brief Abstract interface for receiving TranslationUnits.
class TranslationUnitHandler {
public:
  typedef TranslationUnit* receiving_type;

  virtual ~TranslationUnitHandler();
  virtual void Handle(TranslationUnit *TU) = 0;
};

/// \brief Abstract interface for receiving TULocations.
class TULocationHandler {
public:
  typedef TULocation receiving_type;

  virtual ~TULocationHandler();
  virtual void Handle(TULocation TULoc) = 0;
};

/// \brief Helper for the Handler classes. Stores the objects into a vector.
/// example:
/// @code
/// Storing<TranslationUnitHandler> TURes;
/// IndexProvider.GetTranslationUnitsFor(Entity, TURes);
/// for (Storing<TranslationUnitHandler>::iterator
///   I = TURes.begin(), E = TURes.end(); I != E; ++I) { ....
/// @endcode
template <typename handler_type>
class Storing : public handler_type {
  typedef typename handler_type::receiving_type receiving_type;
  typedef SmallVector<receiving_type, 8> StoreTy;
  StoreTy Store;

public:
  virtual void Handle(receiving_type Obj) {
    Store.push_back(Obj);
  }

  typedef typename StoreTy::const_iterator iterator;
  iterator begin() const { return Store.begin(); }
  iterator end() const { return Store.end(); }
};

} // namespace idx

} // namespace clang

#endif
