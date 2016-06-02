//===- ModuleSubstreamRecord.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_MODULESUBSTREAMRECORD_H
#define LLVM_DEBUGINFO_PDB_RAW_MODULESUBSTREAMRECORD_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/StreamArray.h"
#include "llvm/DebugInfo/CodeView/StreamRef.h"
#include "llvm/Support/Error.h"

namespace llvm {

namespace pdb {
class ModuleSubstreamRecord {
public:
  ModuleSubstreamRecord();
  ModuleSubstreamRecord(codeview::ModuleSubstreamKind Kind,
                        codeview::StreamRef Data);
  static Error initialize(codeview::StreamRef Stream,
                          ModuleSubstreamRecord &Info);
  uint32_t getRecordLength() const;
  codeview::ModuleSubstreamKind getSubstreamKind() const;
  codeview::StreamRef getRecordData() const;

private:
  codeview::ModuleSubstreamKind Kind;
  codeview::StreamRef Data;
};
}

namespace codeview {
template <> struct VarStreamArrayExtractor<pdb::ModuleSubstreamRecord> {
  Error operator()(StreamRef Stream, uint32_t &Length,
                   pdb::ModuleSubstreamRecord &Info) const {
    if (auto EC = pdb::ModuleSubstreamRecord::initialize(Stream, Info))
      return EC;
    Length = Info.getRecordLength();
    return Error::success();
  }
};
}
}

#endif // LLVM_DEBUGINFO_PDB_RAW_MODULESUBSTREAMRECORD_H
