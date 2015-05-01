//===-- PDBContext.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===/

#include "llvm/DebugInfo/PDB/PDBContext.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBLineNumber.h"
#include "llvm/DebugInfo/PDB/IPDBSourceFile.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/Object/COFF.h"

using namespace llvm;
using namespace llvm::object;

PDBContext::PDBContext(const COFFObjectFile &Object,
                       std::unique_ptr<IPDBSession> PDBSession)
    : DIContext(CK_PDB), Session(std::move(PDBSession)) {
  uint64_t ImageBase = 0;
  if (Object.is64()) {
    const pe32plus_header *Header = nullptr;
    Object.getPE32PlusHeader(Header);
    if (Header)
      ImageBase = Header->ImageBase;
  } else {
    const pe32_header *Header = nullptr;
    Object.getPE32Header(Header);
    if (Header)
      ImageBase = static_cast<uint64_t>(Header->ImageBase);
  }
  Session->setLoadAddress(ImageBase);
}

void PDBContext::dump(raw_ostream &OS, DIDumpType DumpType) {}

DILineInfo PDBContext::getLineInfoForAddress(uint64_t Address,
                                             DILineInfoSpecifier Specifier) {
  auto Symbol = Session->findSymbolByAddress(Address, PDB_SymType::None);

  uint32_t Length = 1;
  DILineInfo Result;
  if (auto Func = dyn_cast_or_null<PDBSymbolFunc>(Symbol.get())) {
    if (Specifier.FNKind == DINameKind::LinkageName)
      Result.FunctionName = Func->getUndecoratedName();
    else if (Specifier.FNKind == DINameKind::ShortName)
      Result.FunctionName = Func->getName();

    Length = Func->getLength();
  } else if (auto Data = dyn_cast_or_null<PDBSymbolData>(Symbol.get())) {
    Length = Data->getLength();
  }

  // If we couldn't find a symbol, then just assume 1 byte, so that we get
  // only the line number of the first instruction.
  auto LineNumbers = Session->findLineNumbersByAddress(Address, Length);
  if (!LineNumbers || LineNumbers->getChildCount() == 0)
    return Result;

  auto LineInfo = LineNumbers->getNext();
  assert(LineInfo);
  auto SourceFile = Session->getSourceFileById(LineInfo->getSourceFileId());

  if (SourceFile &&
      Specifier.FLIKind != DILineInfoSpecifier::FileLineInfoKind::None)
    Result.FileName = SourceFile->getFileName();
  Result.Column = LineInfo->getColumnNumber();
  Result.Line = LineInfo->getLineNumber();
  return Result;
}

DILineInfoTable
PDBContext::getLineInfoForAddressRange(uint64_t Address, uint64_t Size,
                                       DILineInfoSpecifier Specifier) {
  if (Size == 0)
    return DILineInfoTable();

  DILineInfoTable Table;
  auto LineNumbers = Session->findLineNumbersByAddress(Address, Size);
  if (!LineNumbers || LineNumbers->getChildCount() == 0)
    return Table;

  while (auto LineInfo = LineNumbers->getNext()) {
    DILineInfo LineEntry =
        getLineInfoForAddress(LineInfo->getVirtualAddress(), Specifier);
    Table.push_back(std::make_pair(LineInfo->getVirtualAddress(), LineEntry));
  }
  return Table;
}

DIInliningInfo
PDBContext::getInliningInfoForAddress(uint64_t Address,
                                      DILineInfoSpecifier Specifier) {
  DIInliningInfo InlineInfo;
  DILineInfo Frame = getLineInfoForAddress(Address, Specifier);
  InlineInfo.addFrame(Frame);
  return InlineInfo;
}
