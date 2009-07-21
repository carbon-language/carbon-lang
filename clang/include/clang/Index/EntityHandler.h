//===--- EntityHandler.h - Interface for receiving entities -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Abstract interface for receiving Entities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_ENTITYHANDLER_H
#define LLVM_CLANG_INDEX_ENTITYHANDLER_H

namespace clang {

namespace idx {
  class Entity;

/// \brief Abstract interface for receiving Entities.
class EntityHandler {
public:
  virtual ~EntityHandler();
  virtual void HandleEntity(Entity Ent);
};
  
} // namespace idx

} // namespace clang

#endif
