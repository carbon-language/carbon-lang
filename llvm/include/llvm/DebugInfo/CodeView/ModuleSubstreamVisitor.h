//===- ModuleSubstreamVisitor.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_MODULESUBSTREAMVISITOR_H
#define LLVM_DEBUGINFO_CODEVIEW_MODULESUBSTREAMVISITOR_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/DebugInfo/CodeView/Line.h"
#include "llvm/DebugInfo/CodeView/ModuleSubstream.h"
#include "llvm/DebugInfo/CodeView/StreamReader.h"
#include "llvm/DebugInfo/CodeView/StreamRef.h"

namespace llvm {
namespace codeview {

struct LineColumnEntry {
  support::ulittle32_t Offset;
  FixedStreamArray<LineNumberEntry> LineNumbers;
  FixedStreamArray<ColumnNumberEntry> Columns;
};

class FileLineInfoExtractor {
public:
  FileLineInfoExtractor(const LineSubstreamHeader *Header) : Header(Header) {}

  Error operator()(StreamRef Stream, uint32_t &Len,
                   LineColumnEntry &Item) const {
    const LineFileBlockHeader *BlockHeader;
    StreamReader Reader(Stream);
    if (auto EC = Reader.readObject(BlockHeader))
      return EC;
    bool HasColumn = Header->Flags & LineFlags::HaveColumns;
    uint32_t LineInfoSize =
        BlockHeader->NumLines *
        (sizeof(LineNumberEntry) + (HasColumn ? sizeof(ColumnNumberEntry) : 0));
    if (BlockHeader->BlockSize < sizeof(LineFileBlockHeader))
      return make_error<CodeViewError>(cv_error_code::corrupt_record,
                                       "Invalid line block record size");
    uint32_t Size = BlockHeader->BlockSize - sizeof(LineFileBlockHeader);
    if (LineInfoSize > Size)
      return make_error<CodeViewError>(cv_error_code::corrupt_record,
                                       "Invalid line block record size");
    // The value recorded in BlockHeader->BlockSize includes the size of
    // LineFileBlockHeader.
    Len = BlockHeader->BlockSize;
    Item.Offset = BlockHeader->FileOffset;
    if (auto EC = Reader.readArray(Item.LineNumbers, BlockHeader->NumLines))
      return EC;
    if (HasColumn) {
      if (auto EC = Reader.readArray(Item.Columns, BlockHeader->NumLines))
        return EC;
    }
    return Error::success();
  }

private:
  const LineSubstreamHeader *Header;
};

typedef VarStreamArray<LineColumnEntry, FileLineInfoExtractor> LineInfoArray;

class IModuleSubstreamVisitor {
public:
  virtual ~IModuleSubstreamVisitor() {}

  virtual Error visitUnknown(ModuleSubstreamKind Kind, StreamRef Data) = 0;
  virtual Error visitSymbols(StreamRef Data);
  virtual Error visitLines(StreamRef Data, const LineSubstreamHeader *Header,
                           LineInfoArray Lines);
  virtual Error visitStringTable(StreamRef Data);
  virtual Error visitFileChecksums(StreamRef Data);
  virtual Error visitFrameData(StreamRef Data);
  virtual Error visitInlineeLines(StreamRef Data);
  virtual Error visitCrossScopeImports(StreamRef Data);
  virtual Error visitCrossScopeExports(StreamRef Data);
  virtual Error visitILLines(StreamRef Data);
  virtual Error visitFuncMDTokenMap(StreamRef Data);
  virtual Error visitTypeMDTokenMap(StreamRef Data);
  virtual Error visitMergedAssemblyInput(StreamRef Data);
  virtual Error visitCoffSymbolRVA(StreamRef Data);
};

Error visitModuleSubstream(const ModuleSubstream &R,
                           IModuleSubstreamVisitor &V);

} // namespace codeview
} // namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_MODULESUBSTREAMVISITOR_H
