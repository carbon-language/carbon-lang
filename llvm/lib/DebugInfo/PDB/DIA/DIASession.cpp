//===- DIASession.cpp - DIA implementation of IPDBSession -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"

#include "llvm/DebugInfo/PDB/DIA/DIAEnumDebugStreams.h"
#include "llvm/DebugInfo/PDB/DIA/DIAEnumSourceFiles.h"
#include "llvm/DebugInfo/PDB/DIA/DIARawSymbol.h"
#include "llvm/DebugInfo/PDB/DIA/DIASession.h"
#include "llvm/DebugInfo/PDB/DIA/DIASourceFile.h"
#include "llvm/Support/ConvertUTF.h"

using namespace llvm;

namespace {}

DIASession::DIASession(CComPtr<IDiaSession> DiaSession) : Session(DiaSession) {}

DIASession *DIASession::createFromPdb(StringRef Path) {
  CComPtr<IDiaDataSource> DataSource;
  CComPtr<IDiaSession> Session;

  // We assume that CoInitializeEx has already been called by the executable.
  HRESULT Result = ::CoCreateInstance(CLSID_DiaSource, nullptr,
                                      CLSCTX_INPROC_SERVER, IID_IDiaDataSource,
                                      reinterpret_cast<LPVOID *>(&DataSource));
  if (FAILED(Result))
    return nullptr;

  llvm::SmallVector<UTF16, 128> Path16;
  if (!llvm::convertUTF8ToUTF16String(Path, Path16))
    return nullptr;

  const wchar_t *Path16Str = reinterpret_cast<const wchar_t*>(Path16.data());
  if (FAILED(DataSource->loadDataFromPdb(Path16Str)))
    return nullptr;

  if (FAILED(DataSource->openSession(&Session)))
    return nullptr;
  return new DIASession(Session);
}

uint64_t DIASession::getLoadAddress() const {
  uint64_t LoadAddress;
  bool success = (S_OK == Session->get_loadAddress(&LoadAddress));
  return (success) ? LoadAddress : 0;
}

void DIASession::setLoadAddress(uint64_t Address) {
  Session->put_loadAddress(Address);
}

std::unique_ptr<PDBSymbolExe> DIASession::getGlobalScope() const {
  CComPtr<IDiaSymbol> GlobalScope;
  if (S_OK != Session->get_globalScope(&GlobalScope))
    return nullptr;

  auto RawSymbol = std::make_unique<DIARawSymbol>(*this, GlobalScope);
  auto PdbSymbol(PDBSymbol::create(*this, std::move(RawSymbol)));
  std::unique_ptr<PDBSymbolExe> ExeSymbol(
      static_cast<PDBSymbolExe *>(PdbSymbol.release()));
  return ExeSymbol;
}

std::unique_ptr<PDBSymbol> DIASession::getSymbolById(uint32_t SymbolId) const {
  CComPtr<IDiaSymbol> LocatedSymbol;
  if (S_OK != Session->symbolById(SymbolId, &LocatedSymbol))
    return nullptr;

  auto RawSymbol = std::make_unique<DIARawSymbol>(*this, LocatedSymbol);
  return PDBSymbol::create(*this, std::move(RawSymbol));
}

std::unique_ptr<IPDBEnumSourceFiles> DIASession::getAllSourceFiles() const {
  CComPtr<IDiaEnumSourceFiles> Files;
  if (S_OK != Session->findFile(nullptr, nullptr, nsNone, &Files))
    return nullptr;

  return std::make_unique<DIAEnumSourceFiles>(*this, Files);
}

std::unique_ptr<IPDBEnumSourceFiles> DIASession::getSourceFilesForCompiland(
    const PDBSymbolCompiland &Compiland) const {
  CComPtr<IDiaEnumSourceFiles> Files;

  const DIARawSymbol &RawSymbol =
      static_cast<const DIARawSymbol &>(Compiland.getRawSymbol());
  if (S_OK !=
      Session->findFile(RawSymbol.getDiaSymbol(), nullptr, nsNone, &Files))
    return nullptr;

  return std::make_unique<DIAEnumSourceFiles>(*this, Files);
}

std::unique_ptr<IPDBSourceFile>
DIASession::getSourceFileById(uint32_t FileId) const {
  CComPtr<IDiaSourceFile> LocatedFile;
  if (S_OK != Session->findFileById(FileId, &LocatedFile))
    return nullptr;

  return std::make_unique<DIASourceFile>(*this, LocatedFile);
}

std::unique_ptr<IPDBEnumDataStreams> DIASession::getDebugStreams() const {
  CComPtr<IDiaEnumDebugStreams> DiaEnumerator;
  if (S_OK != Session->getEnumDebugStreams(&DiaEnumerator))
    return nullptr;

  return std::unique_ptr<IPDBEnumDataStreams>(
      new DIAEnumDebugStreams(DiaEnumerator));
}
