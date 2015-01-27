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

#define NTDDI_VERSION NTDDI_VISTA
#define _WIN32_WINNT _WIN32_WINNT_VISTA
#define WINVER _WIN32_WINNT_VISTA
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <atlbase.h>
#include <windows.h>
#include <dia2.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"

#include "COMExtras.h"

using namespace llvm;
using namespace llvm::sys::windows;

namespace opts {
cl::list<std::string> InputFilenames(cl::Positional,
                                     cl::desc("<input PDB files>"),
                                     cl::OneOrMore);

cl::opt<bool> Streams("streams", cl::desc("Display data stream information"));
cl::alias StreamsShort("s", cl::desc("Alias for --streams"),
                       cl::aliasopt(Streams));

cl::opt<bool> StreamData("stream-data",
                         cl::desc("Dumps stream record data as bytes"));
cl::alias StreamDataShort("S", cl::desc("Alias for --stream-data"),
                          cl::aliasopt(StreamData));
}

namespace {
bool BSTRToUTF8(BSTR String16, std::string &String8) {
  UINT ByteLength = ::SysStringByteLen(String16);
  char *Bytes = reinterpret_cast<char *>(String16);
  String8.clear();
  return llvm::convertUTF16ToUTF8String(ArrayRef<char>(Bytes, ByteLength),
                                        String8);
}
}

static void dumpDataStreams(IDiaSession *session) {
  CComPtr<IDiaEnumDebugStreams> DebugStreams = nullptr;
  if (FAILED(session->getEnumDebugStreams(&DebugStreams)))
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
  CComPtr<IDiaSession> session;
  if (FAILED(source->openSession(&session)))
    return;
  if (opts::Streams || opts::StreamData) {
    dumpDataStreams(session);
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
