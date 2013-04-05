//===- lld/ReaderWriter/CoreTargetInfo.h ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_CORE_TARGET_INFO_H
#define LLD_READER_WRITER_CORE_TARGET_INFO_H

#include "lld/Core/TargetInfo.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/Support/ErrorHandling.h"

namespace lld {

class CoreTargetInfo : public TargetInfo {
public:
  CoreTargetInfo(); 

  virtual bool validate(raw_ostream &diagnostics) {
    return false;
  }
   
  virtual void addPasses(PassManager &pm) const;
  virtual ErrorOr<Reference::Kind>    relocKindFromString(StringRef str) const;
  virtual ErrorOr<std::string> stringFromRelocKind(Reference::Kind kind) const;

  virtual error_code
  parseFile(std::unique_ptr<MemoryBuffer> mb,
            std::vector<std::unique_ptr<File>> &result) const;

  void addPassNamed(StringRef name) {
    _passNames.push_back(name);
  }

protected:
  virtual Writer &writer() const;

private:
  mutable std::unique_ptr<Reader>   _reader;
  mutable std::unique_ptr<Writer>   _writer;
  std::vector<StringRef>            _passNames;
};

} // end namespace lld

#endif
