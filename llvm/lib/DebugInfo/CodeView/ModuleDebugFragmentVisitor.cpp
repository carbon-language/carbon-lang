//===- ModuleDebugFragmentVisitor.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/ModuleDebugFragmentVisitor.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamRef.h"

using namespace llvm;
using namespace llvm::codeview;

Error ModuleDebugFragmentVisitor::visitSymbols(BinaryStreamRef Data) {
  return visitUnknown(ModuleDebugFragmentKind::Symbols, Data);
}
Error ModuleDebugFragmentVisitor::visitLines(BinaryStreamRef Data,
                                             const LineFragmentHeader *Header,
                                             const LineInfoArray &Lines) {
  return visitUnknown(ModuleDebugFragmentKind::Lines, Data);
}
Error ModuleDebugFragmentVisitor::visitStringTable(BinaryStreamRef Data) {
  return visitUnknown(ModuleDebugFragmentKind::StringTable, Data);
}
Error ModuleDebugFragmentVisitor::visitFileChecksums(
    BinaryStreamRef Data, const FileChecksumArray &Checksums) {
  return visitUnknown(ModuleDebugFragmentKind::FileChecksums, Data);
}
Error ModuleDebugFragmentVisitor::visitFrameData(BinaryStreamRef Data) {
  return visitUnknown(ModuleDebugFragmentKind::FrameData, Data);
}
Error ModuleDebugFragmentVisitor::visitInlineeLines(BinaryStreamRef Data) {
  return visitUnknown(ModuleDebugFragmentKind::InlineeLines, Data);
}
Error ModuleDebugFragmentVisitor::visitCrossScopeImports(BinaryStreamRef Data) {
  return visitUnknown(ModuleDebugFragmentKind::CrossScopeExports, Data);
}
Error ModuleDebugFragmentVisitor::visitCrossScopeExports(BinaryStreamRef Data) {
  return visitUnknown(ModuleDebugFragmentKind::CrossScopeImports, Data);
}
Error ModuleDebugFragmentVisitor::visitILLines(BinaryStreamRef Data) {
  return visitUnknown(ModuleDebugFragmentKind::ILLines, Data);
}
Error ModuleDebugFragmentVisitor::visitFuncMDTokenMap(BinaryStreamRef Data) {
  return visitUnknown(ModuleDebugFragmentKind::FuncMDTokenMap, Data);
}
Error ModuleDebugFragmentVisitor::visitTypeMDTokenMap(BinaryStreamRef Data) {
  return visitUnknown(ModuleDebugFragmentKind::TypeMDTokenMap, Data);
}
Error ModuleDebugFragmentVisitor::visitMergedAssemblyInput(
    BinaryStreamRef Data) {
  return visitUnknown(ModuleDebugFragmentKind::MergedAssemblyInput, Data);
}
Error ModuleDebugFragmentVisitor::visitCoffSymbolRVA(BinaryStreamRef Data) {
  return visitUnknown(ModuleDebugFragmentKind::CoffSymbolRVA, Data);
}

Error llvm::codeview::visitModuleDebugFragment(const ModuleDebugFragment &R,
                                               ModuleDebugFragmentVisitor &V) {
  switch (R.kind()) {
  case ModuleDebugFragmentKind::Symbols:
    return V.visitSymbols(R.getRecordData());
  case ModuleDebugFragmentKind::Lines: {
    BinaryStreamReader Reader(R.getRecordData());
    const LineFragmentHeader *Header;
    if (auto EC = Reader.readObject(Header))
      return EC;
    VarStreamArrayExtractor<LineColumnEntry> E(Header);
    LineInfoArray LineInfos(E);
    if (auto EC = Reader.readArray(LineInfos, Reader.bytesRemaining()))
      return EC;
    return V.visitLines(R.getRecordData(), Header, LineInfos);
  }
  case ModuleDebugFragmentKind::StringTable:
    return V.visitStringTable(R.getRecordData());
  case ModuleDebugFragmentKind::FileChecksums: {
    BinaryStreamReader Reader(R.getRecordData());
    FileChecksumArray Checksums;
    if (auto EC = Reader.readArray(Checksums, Reader.bytesRemaining()))
      return EC;
    return V.visitFileChecksums(R.getRecordData(), Checksums);
  }
  case ModuleDebugFragmentKind::FrameData:
    return V.visitFrameData(R.getRecordData());
  case ModuleDebugFragmentKind::InlineeLines:
    return V.visitInlineeLines(R.getRecordData());
  case ModuleDebugFragmentKind::CrossScopeImports:
    return V.visitCrossScopeImports(R.getRecordData());
  case ModuleDebugFragmentKind::CrossScopeExports:
    return V.visitCrossScopeExports(R.getRecordData());
  case ModuleDebugFragmentKind::ILLines:
    return V.visitILLines(R.getRecordData());
  case ModuleDebugFragmentKind::FuncMDTokenMap:
    return V.visitFuncMDTokenMap(R.getRecordData());
  case ModuleDebugFragmentKind::TypeMDTokenMap:
    return V.visitTypeMDTokenMap(R.getRecordData());
  case ModuleDebugFragmentKind::MergedAssemblyInput:
    return V.visitMergedAssemblyInput(R.getRecordData());
  case ModuleDebugFragmentKind::CoffSymbolRVA:
    return V.visitCoffSymbolRVA(R.getRecordData());
  default:
    return V.visitUnknown(R.kind(), R.getRecordData());
  }
}
