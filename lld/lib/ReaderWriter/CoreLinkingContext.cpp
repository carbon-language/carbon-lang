//===- lib/ReaderWriter/CoreLinkingContext.cpp ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/Pass.h"
#include "lld/Core/PassManager.h"
#include "lld/Core/Simple.h"
#include "lld/ReaderWriter/CoreLinkingContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

using namespace lld;

namespace {

class OrderPass : public Pass {
public:
  /// Sorts atoms by position
  std::error_code perform(SimpleFile &file) override {
    SimpleFile::DefinedAtomRange defined = file.definedAtoms();
    std::sort(defined.begin(), defined.end(), DefinedAtom::compareByPosition);
    return std::error_code();
  }
};

} // anonymous namespace

CoreLinkingContext::CoreLinkingContext() {}

bool CoreLinkingContext::validateImpl(raw_ostream &) {
  _writer = createWriterYAML(*this);
  return true;
}

void CoreLinkingContext::addPasses(PassManager &pm) {
  for (StringRef name : _passNames) {
    (void)name;
    assert(name == "order" && "bad pass name");
    pm.add(llvm::make_unique<OrderPass>());
  }
}

Writer &CoreLinkingContext::writer() const { return *_writer; }
