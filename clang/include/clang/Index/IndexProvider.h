//===--- IndexProvider.h - Maps information to translation units -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Maps information to TranslationUnits.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_INDEXPROVIDER_H
#define LLVM_CLANG_INDEX_INDEXPROVIDER_H

namespace clang {

namespace idx {
  class Entity;
  class TranslationUnitHandler;
  class GlobalSelector;

/// \brief Maps information to TranslationUnits.
class IndexProvider {
public:
  virtual ~IndexProvider();
  virtual void GetTranslationUnitsFor(Entity Ent,
                                      TranslationUnitHandler &Handler) = 0;
  virtual void GetTranslationUnitsFor(GlobalSelector Sel,
                                      TranslationUnitHandler &Handler) = 0;
};

} // namespace idx

} // namespace clang

#endif
