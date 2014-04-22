//===--- llvm/Support/DataStream.cpp - Lazy streamed data -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements DataStreamer, which fetches bytes of Data from
// a stream source. It provides support for streaming (lazy reading) of
// bitcode. An example implementation of streaming from a file or stdin
// is included.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DataStream.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/system_error.h"
#include <cerrno>
#include <cstdio>
#include <string>
#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif
using namespace llvm;

#define DEBUG_TYPE "Data-stream"

// Interface goals:
// * StreamableMemoryObject doesn't care about complexities like using
//   threads/async callbacks to actually overlap download+compile
// * Don't want to duplicate Data in memory
// * Don't need to know total Data len in advance
// Non-goals:
// StreamableMemoryObject already has random access so this interface only does
// in-order streaming (no arbitrary seeking, else we'd have to buffer all the
// Data here in addition to MemoryObject).  This also means that if we want
// to be able to to free Data, BitstreamBytes/BitcodeReader will implement it

STATISTIC(NumStreamFetches, "Number of calls to Data stream fetch");

namespace llvm {
DataStreamer::~DataStreamer() {}
}

namespace {

// Very simple stream backed by a file. Mostly useful for stdin and debugging;
// actual file access is probably still best done with mmap.
class DataFileStreamer : public DataStreamer {
 int Fd;
public:
  DataFileStreamer() : Fd(0) {}
  virtual ~DataFileStreamer() {
    close(Fd);
  }
  size_t GetBytes(unsigned char *buf, size_t len) override {
    NumStreamFetches++;
    return read(Fd, buf, len);
  }

  error_code OpenFile(const std::string &Filename) {
    if (Filename == "-") {
      Fd = 0;
      sys::ChangeStdinToBinary();
      return error_code::success();
    }

    return sys::fs::openFileForRead(Filename, Fd);
  }
};

}

namespace llvm {
DataStreamer *getDataFileStreamer(const std::string &Filename,
                                  std::string *StrError) {
  DataFileStreamer *s = new DataFileStreamer();
  if (error_code e = s->OpenFile(Filename)) {
    *StrError = std::string("Could not open ") + Filename + ": " +
        e.message() + "\n";
    return nullptr;
  }
  return s;
}

}
