//===- lld/ReaderWriter/CoreLinkingContext.h ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_CORE_LINKER_CONTEXT_H
#define LLD_READER_WRITER_CORE_LINKER_CONTEXT_H

#include "lld/Core/LinkingContext.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"
#include "llvm/Support/ErrorHandling.h"

namespace lld {

class CoreLinkingContext : public LinkingContext {
public:
  CoreLinkingContext();

  enum {
    TEST_RELOC_CALL32        = 1,
    TEST_RELOC_PCREL32       = 2,
    TEST_RELOC_GOT_LOAD32    = 3,
    TEST_RELOC_GOT_USE32     = 4,
    TEST_RELOC_LEA32_WAS_GOT = 5,
  };

  bool validateImpl(raw_ostream &diagnostics) override;
  void addPasses(PassManager &pm) override;

  void addPassNamed(StringRef name) { _passNames.push_back(name); }

protected:
  Writer &writer() const override;

private:
  std::unique_ptr<Writer>  _writer;
  std::vector<StringRef>   _passNames;
};

} // end namespace lld

#endif
