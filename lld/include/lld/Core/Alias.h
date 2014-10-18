//===- lld/Core/Alias.h - Alias atoms -------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Provide alias atoms.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_ALIAS_H
#define LLD_CORE_ALIAS_H

#include "lld/Core/LLVM.h"
#include "lld/Core/Simple.h"
#include "llvm/ADT/Optional.h"
#include <string>

namespace lld {

// An AliasAtom is a zero-size atom representing an alias for other atom. It has
// a LayoutAfter reference to the target atom, so that this atom and the target
// atom will be laid out at the same location in the final result. Initially
// the target atom is an undefined atom. Resolver will replace it with a defined
// one.
//
// It does not have attributes itself. Most member function calls are forwarded
// to the target atom.
class AliasAtom : public SimpleDefinedAtom {
public:
  AliasAtom(const File &file, StringRef name)
      : SimpleDefinedAtom(file), _target(nullptr), _name(name),
        _merge(DefinedAtom::mergeNo), _deadStrip(DefinedAtom::deadStripNormal) {
  }

  StringRef name() const override { return _name; }
  uint64_t size() const override { return 0; }
  ArrayRef<uint8_t> rawContent() const override { return ArrayRef<uint8_t>(); }

  Scope scope() const override {
    getTarget();
    return _target ? _target->scope() : scopeLinkageUnit;
  }

  Merge merge() const override {
    if (_merge.hasValue())
      return _merge.getValue();
    getTarget();
    return _target ? _target->merge() : mergeNo;
  }

  void setMerge(Merge val) { _merge = val; }

  ContentType contentType() const override {
    getTarget();
    return _target ? _target->contentType() : typeUnknown;
  }

  Interposable interposable() const override {
    getTarget();
    return _target ? _target->interposable() : interposeNo;
  }

  SectionChoice sectionChoice() const override {
    getTarget();
    return _target ? _target->sectionChoice() : sectionBasedOnContent;
  }

  StringRef customSectionName() const override {
    getTarget();
    return _target ? _target->customSectionName() : StringRef("");
  }

  DeadStripKind deadStrip() const override { return _deadStrip; }
  void setDeadStrip(DeadStripKind val) { _deadStrip = val; }

private:
  void getTarget() const {
    if (_target)
      return;
    for (const Reference *r : *this) {
      if (r->kindNamespace() == lld::Reference::KindNamespace::all &&
          r->kindValue() == lld::Reference::kindLayoutAfter) {
        _target = dyn_cast<DefinedAtom>(r->target());
        return;
      }
    }
  }

  mutable const DefinedAtom *_target;
  std::string _name;
  llvm::Optional<Merge> _merge;
  DeadStripKind _deadStrip;
};

} // end namespace lld

#endif
