//===- Core/Atom.cpp - The Fundimental Unit of Linking --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/Atom.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace lld {

  Atom::~Atom() {}

  bool Atom::translationUnitSource(llvm::StringRef &path) const {
    return false;
  }

  llvm::StringRef Atom::name() const {
    return llvm::StringRef();
  }

  llvm::StringRef Atom::customSectionName() const {
    return llvm::StringRef();
  }

  llvm::ArrayRef<uint8_t> Atom::rawContent() const {
    return llvm::ArrayRef<uint8_t>();
  }

  Atom::ContentPermissions Atom::permissions() const { 
    return perm___; 
  }

  Reference::iterator Atom::referencesBegin() const {
    return 0;
  }

  Reference::iterator Atom::referencesEnd() const{
    return 0;
  }

  Atom::UnwindInfo::iterator Atom::beginUnwind() const{
    return 0;
  }

  Atom::UnwindInfo::iterator Atom::endUnwind() const{
    return 0;
  }

} // namespace lld
