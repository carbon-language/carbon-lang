//===- llvm-pdbdump.cpp - Dump debug info from a PDB file -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Dumps debug information present in PDB files.  This utility makes use of
// the Microsoft Windows SDK, so will not compile or run on non-Windows
// platforms.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"

#include "llvm-pdbdump.h"
#include "COMExtras.h"
#include "DIAExtras.h"
#include "DIASymbol.h"

using namespace llvm;
using namespace llvm::sys::windows;

namespace opts {
cl::list<std::string> InputFilenames(cl::Positional,
                                     cl::desc("<input PDB files>"),
                                     cl::OneOrMore);

cl::opt<bool> Streams("streams", cl::desc("Display data stream information"));
cl::alias StreamsShort("x", cl::desc("Alias for --streams"),
                       cl::aliasopt(Streams));

cl::opt<bool> StreamData("stream-data",
                         cl::desc("Dumps stream record data as bytes"));
cl::alias StreamDataShort("X", cl::desc("Alias for --stream-data"),
                          cl::aliasopt(StreamData));

cl::opt<bool> Tables("tables",
                     cl::desc("Display summary information for all of the "
                              "debug tables in the input file"));
cl::alias TablesShort("t", cl::desc("Alias for --tables"),
                      cl::aliasopt(Tables));

cl::opt<bool> SourceFiles("source-files",
                          cl::desc("Display a list of the source files "
                                   "contained in the PDB"));
cl::alias SourceFilesShort("f", cl::desc("Alias for --source-files"),
                           cl::aliasopt(SourceFiles));

cl::opt<bool> Compilands("compilands",
                         cl::desc("Display a list of compilands (e.g. object "
                                  "files) and their source file composition"));
cl::alias CompilandsShort("c", cl::desc("Alias for --compilands"),
                          cl::aliasopt(Compilands));

cl::opt<bool> Symbols("symbols", cl::desc("Display symbols"));
cl::alias SymbolsShort("s", cl::desc("Alias for --symbols"),
                       cl::aliasopt(Symbols));

cl::opt<bool> SymbolDetails("symbol-details",
                            cl::desc("Display symbol details"));
cl::alias SymbolDetailsShort("S", cl::desc("Alias for --symbol-details"),
                             cl::aliasopt(SymbolDetails));
}

template <typename TableType>
static HRESULT getDIATable(IDiaSession *Session, TableType **Table) {
  CComPtr<IDiaEnumTables> EnumTables = nullptr;
  HRESULT Error = S_OK;
  if (FAILED(Error = Session->getEnumTables(&EnumTables)))
    return Error;

  for (auto CurTable : make_com_enumerator(EnumTables)) {
    TableType *ResultTable = nullptr;
    if (FAILED(CurTable->QueryInterface(
            __uuidof(TableType), reinterpret_cast<void **>(&ResultTable))))
      continue;

    *Table = ResultTable;
    return S_OK;
  }
  return E_FAIL;
}

static void dumpBasicFileInfo(StringRef Path, IDiaSession *Session) {
  CComPtr<IDiaSymbol> GlobalScope;
  HRESULT hr = Session->get_globalScope(&GlobalScope);
  DIASymbol GlobalScopeSymbol(GlobalScope);
  if (S_OK == hr)
    GlobalScopeSymbol.getSymbolsFileName().dump("File", 0);
  else
    outs() << "File: " << Path << "\n";
  HANDLE FileHandle = ::CreateFile(
      Path.data(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  LARGE_INTEGER FileSize;
  if (INVALID_HANDLE_VALUE != FileHandle) {
    outs().indent(2);
    if (::GetFileSizeEx(FileHandle, &FileSize))
      outs() << "Size: " << FileSize.QuadPart << " bytes\n";
    else
      outs() << "Size: (Unable to obtain file size)\n";
    FILETIME ModifiedTime;
    outs().indent(2);
    if (::GetFileTime(FileHandle, nullptr, nullptr, &ModifiedTime)) {
      ULARGE_INTEGER TimeInteger;
      TimeInteger.LowPart = ModifiedTime.dwLowDateTime;
      TimeInteger.HighPart = ModifiedTime.dwHighDateTime;
      llvm::sys::TimeValue Time;
      Time.fromWin32Time(TimeInteger.QuadPart);
      outs() << "Timestamp: " << Time.str() << "\n";
    } else {
      outs() << "Timestamp: (Unable to obtain time stamp)\n";
    }
    ::CloseHandle(FileHandle);
  }

  if (S_OK == hr)
    GlobalScopeSymbol.fullDump(2);
  outs() << "\n";
  outs().flush();
}

static void dumpDataStreams(IDiaSession *Session) {
  CComPtr<IDiaEnumDebugStreams> DebugStreams = nullptr;
  if (FAILED(Session->getEnumDebugStreams(&DebugStreams)))
    return;

  LONG Count = 0;
  if (FAILED(DebugStreams->get_Count(&Count)))
    return;
  outs() << "Data Streams [count=" << Count << "]\n";

  std::string Name8;

  for (auto Stream : make_com_enumerator(DebugStreams)) {
    BSTR Name16;
    if (FAILED(Stream->get_name(&Name16)))
      continue;
    if (BSTRToUTF8(Name16, Name8))
      outs() << "  " << Name8;
    ::SysFreeString(Name16);
    if (FAILED(Stream->get_Count(&Count))) {
      outs() << "\n";
      continue;
    }

    outs() << " [" << Count << " records]\n";
    if (opts::StreamData) {
      int RecordIndex = 0;
      for (auto StreamRecord : make_com_data_record_enumerator(Stream)) {
        outs() << "    Record " << RecordIndex << " [" << StreamRecord.size()
               << " bytes]";
        for (uint8_t byte : StreamRecord) {
          outs() << " " << llvm::format_hex_no_prefix(byte, 2, true);
        }
        outs() << "\n";
        ++RecordIndex;
      }
    }
  }
  outs() << "\n";
  outs().flush();
}

static void dumpDebugTables(IDiaSession *Session) {
  CComPtr<IDiaEnumTables> EnumTables = nullptr;
  if (SUCCEEDED(Session->getEnumTables(&EnumTables))) {
    LONG Count = 0;
    if (FAILED(EnumTables->get_Count(&Count)))
      return;

    outs() << "Debug Tables [count=" << Count << "]\n";

    std::string Name8;
    for (auto Table : make_com_enumerator(EnumTables)) {
      BSTR Name16;
      if (FAILED(Table->get_name(&Name16)))
        continue;
      if (BSTRToUTF8(Name16, Name8))
        outs() << "  " << Name8;
      ::SysFreeString(Name16);
      if (SUCCEEDED(Table->get_Count(&Count))) {
        outs() << " [" << Count << " items]\n";
      } else
        outs() << "\n";
    }
  }
  outs() << "\n";
  outs().flush();
}

static void dumpSourceFiles(IDiaSession *Session) {
  CComPtr<IDiaEnumSourceFiles> EnumSourceFileList;
  if (FAILED(getDIATable(Session, &EnumSourceFileList)))
    return;

  LONG SourceFileCount = 0;
  EnumSourceFileList->get_Count(&SourceFileCount);

  outs() << "Dumping source files [" << SourceFileCount << " files]\n";
  for (auto SourceFile : make_com_enumerator(EnumSourceFileList)) {
    CComBSTR SourceFileName;
    if (S_OK != SourceFile->get_fileName(&SourceFileName))
      continue;
    outs().indent(2);
    std::string SourceFileName8;
    BSTRToUTF8(SourceFileName, SourceFileName8);
    outs() << SourceFileName8 << "\n";
  }
  outs() << "\n";
  outs().flush();
}

static void dumpCompilands(IDiaSession *Session) {
  CComPtr<IDiaEnumSourceFiles> EnumSourceFileList;
  if (FAILED(getDIATable(Session, &EnumSourceFileList)))
    return;

  LONG SourceFileCount = 0;
  EnumSourceFileList->get_Count(&SourceFileCount);

  CComPtr<IDiaSymbol> GlobalScope;
  HRESULT hr = Session->get_globalScope(&GlobalScope);
  DIASymbol GlobalScopeSymbol(GlobalScope);
  if (S_OK != hr)
    return;

  CComPtr<IDiaEnumSymbols> EnumCompilands;
  if (S_OK !=
      GlobalScope->findChildren(SymTagCompiland, nullptr, nsNone,
                                &EnumCompilands))
    return;

  LONG CompilandCount = 0;
  EnumCompilands->get_Count(&CompilandCount);
  outs() << "Dumping compilands [" << CompilandCount
         << " compilands containing " << SourceFileCount << " source files]\n";

  for (auto Compiland : make_com_enumerator(EnumCompilands)) {
    DIASymbol CompilandSymbol(Compiland);
    outs().indent(2);
    outs() << CompilandSymbol.getName().value() << "\n";

    CComPtr<IDiaEnumSourceFiles> EnumFiles;
    if (S_OK != Session->findFile(Compiland, nullptr, nsNone, &EnumFiles))
      continue;

    for (auto SourceFile : make_com_enumerator(EnumFiles)) {
      DWORD ChecksumType = 0;
      DWORD ChecksumSize = 0;
      std::vector<uint8_t> Checksum;
      outs().indent(4);
      SourceFile->get_checksumType(&ChecksumType);
      if (S_OK == SourceFile->get_checksum(0, &ChecksumSize, nullptr)) {
        Checksum.resize(ChecksumSize);
        if (S_OK ==
            SourceFile->get_checksum(ChecksumSize, &ChecksumSize,
                                     &Checksum[0])) {
          outs() << "[" << ((ChecksumType == HashMD5) ? "MD5  " : "SHA-1")
                 << ": ";
          for (auto byte : Checksum)
            outs() << format_hex_no_prefix(byte, 2, true);
          outs() << "] ";
        }
      }
      CComBSTR SourceFileName;
      if (S_OK != SourceFile->get_fileName(&SourceFileName))
        continue;

      std::string SourceFileName8;
      BSTRToUTF8(SourceFileName, SourceFileName8);
      outs() << SourceFileName8 << "\n";
    }
  }

  outs() << "\n";
  outs().flush();
}

static void dumpSymbols(IDiaSession *Session) {
  CComPtr<IDiaEnumSymbols> EnumSymbols;
  if (FAILED(getDIATable(Session, &EnumSymbols)))
    return;

  LONG SymbolCount = 0;
  EnumSymbols->get_Count(&SymbolCount);

  outs() << "Dumping symbols [" << SymbolCount << " symbols]\n";
  int UnnamedSymbolCount = 0;
  for (auto Symbol : make_com_enumerator(EnumSymbols)) {
    DIASymbol SymbolSymbol(Symbol);
    DIAResult<DIAString> SymbolName = SymbolSymbol.getName();
    if (!SymbolName.hasValue() || SymbolName.value().empty()) {
      ++UnnamedSymbolCount;
      outs() << "  (Unnamed symbol)\n";
    } else {
      outs() << "  " << SymbolSymbol.getName().value() << "\n";
    }
    if (opts::SymbolDetails)
      SymbolSymbol.fullDump(4);
  }
  outs() << "(Found " << UnnamedSymbolCount << " unnamed symbols)\n";
  outs().flush();
}

static void dumpInput(StringRef Path) {
  SmallVector<UTF16, 128> Path16String;
  llvm::convertUTF8ToUTF16String(Path, Path16String);
  wchar_t *Path16 = reinterpret_cast<wchar_t *>(Path16String.data());
  CComPtr<IDiaDataSource> source;
  HRESULT hr =
      ::CoCreateInstance(CLSID_DiaSource, nullptr, CLSCTX_INPROC_SERVER,
                         __uuidof(IDiaDataSource), (void **)&source);
  if (FAILED(hr))
    return;
  if (FAILED(source->loadDataFromPdb(Path16)))
    return;
  CComPtr<IDiaSession> Session;
  if (FAILED(source->openSession(&Session)))
    return;

  dumpBasicFileInfo(Path, Session);
  if (opts::Streams || opts::StreamData) {
    dumpDataStreams(Session);
  }

  if (opts::Tables) {
    dumpDebugTables(Session);
  }

  if (opts::SourceFiles) {
    dumpSourceFiles(Session);
  }

  if (opts::Compilands) {
    dumpCompilands(Session);
  }

  if (opts::Symbols || opts::SymbolDetails) {
    dumpSymbols(Session);
  }
}

int main(int argc_, const char *argv_[]) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc_, argv_);

  SmallVector<const char *, 256> argv;
  llvm::SpecificBumpPtrAllocator<char> ArgAllocator;
  std::error_code EC = llvm::sys::Process::GetArgumentVector(
      argv, llvm::makeArrayRef(argv_, argc_), ArgAllocator);
  if (EC) {
    llvm::errs() << "error: couldn't get arguments: " << EC.message() << '\n';
    return 1;
  }

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argv.size(), argv.data(), "LLVM PDB Dumper\n");

  CoInitializeEx(nullptr, COINIT_MULTITHREADED);

  std::for_each(opts::InputFilenames.begin(), opts::InputFilenames.end(),
                dumpInput);

  CoUninitialize();
  return 0;
}
