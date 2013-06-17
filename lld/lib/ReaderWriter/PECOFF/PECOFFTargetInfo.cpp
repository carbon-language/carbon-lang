//===- lib/ReaderWriter/PECOFF/PECOFFTargetInfo.cpp -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/PECOFFTargetInfo.h"

#include "lld/Core/PassManager.h"
#include "lld/Passes/LayoutPass.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/Support/Debug.h"

namespace lld {

error_code PECOFFTargetInfo::parseFile(
    std::unique_ptr<MemoryBuffer> &mb,
    std::vector<std::unique_ptr<File>> &result) const {
  return _reader->parseFile(mb, result);
}

bool PECOFFTargetInfo::validateImpl(raw_ostream &diagnostics) {
  if (_stackReserve < _stackCommit) {
    diagnostics << "Invalid stack size: reserve size must be equal to or "
                << "greater than commit size, but got "
                << _stackCommit << " and " << _stackReserve << ".\n";
    return true;
  }

  if (_heapReserve < _heapCommit) {
    diagnostics << "Invalid heap size: reserve size must be equal to or "
                << "greater than commit size, but got "
                << _heapCommit << " and " << _heapReserve << ".\n";
    return true;
  }

  _reader = createReaderPECOFF(*this);
  _writer = createWriterPECOFF(*this);
  return false;
}

Writer &PECOFFTargetInfo::writer() const {
  return *_writer;
}

ErrorOr<Reference::Kind>
PECOFFTargetInfo::relocKindFromString(StringRef str) const {
  return make_error_code(yaml_reader_error::illegal_value);
}

ErrorOr<std::string>
PECOFFTargetInfo::stringFromRelocKind(Reference::Kind kind) const {
  return make_error_code(yaml_reader_error::illegal_value);
}

void PECOFFTargetInfo::addPasses(PassManager &pm) const {
  pm.add(std::unique_ptr<Pass>(new LayoutPass()));
}

} // end namespace lld
