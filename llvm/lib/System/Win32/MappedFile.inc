//===- Win32/MappedFile.cpp - Win32 MappedFile Implementation ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 implementation of the MappedFile concept.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 code.
//===----------------------------------------------------------------------===//

#include "Win32.h"
#include "llvm/System/Process.h"

namespace llvm {
using namespace sys;

struct sys::MappedFileInfo {
  HANDLE hFile;
  HANDLE hMapping;
  size_t size;
};

bool MappedFile::initialize(std::string* ErrMsg) {
  assert(!MapInfo);
  MapInfo = new MappedFileInfo;
  MapInfo->hFile = INVALID_HANDLE_VALUE;
  MapInfo->hMapping = NULL;

  MapInfo->hFile = CreateFile(Path.c_str(), GENERIC_READ, 0, NULL,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (MapInfo->hFile == INVALID_HANDLE_VALUE) {
    delete MapInfo;
    MapInfo = NULL;
    return MakeErrMsg(ErrMsg,
                      std::string("Can't open file: ") + Path.toString());
  }

  LARGE_INTEGER size;
  if (!GetFileSizeEx(MapInfo->hFile, &size) ||
      (MapInfo->size = size_t(size.QuadPart), MapInfo->size != size.QuadPart)) {
    CloseHandle(MapInfo->hFile);
    delete MapInfo;
    MapInfo = NULL;
    return MakeErrMsg(ErrMsg, 
      std::string("Can't get size of file: ") + Path.toString());
  }

  return false;
}

void MappedFile::terminate() {
  unmap();
  if (MapInfo->hFile != INVALID_HANDLE_VALUE)
    CloseHandle(MapInfo->hFile);
  delete MapInfo;
  MapInfo = NULL;
}

void MappedFile::unmap() {
  assert(MapInfo && "MappedFile not initialized");
  if (isMapped()) {
    UnmapViewOfFile(BasePtr);
    BasePtr = NULL;
  }
  if (MapInfo->hMapping != INVALID_HANDLE_VALUE) {
    CloseHandle(MapInfo->hMapping);
    MapInfo->hMapping = NULL;
  }
}

const void* MappedFile::map(std::string* ErrMsg) {
  if (!isMapped()) {
    MapInfo->hMapping = CreateFileMapping(MapInfo->hFile, NULL, PAGE_READONLY,
                                          0, 0, NULL);
    if (MapInfo->hMapping == NULL) {
      MakeErrMsg(ErrMsg, std::string("Can't map file: ") + Path.toString());
      return 0;
    }

    BasePtr = MapViewOfFileEx(MapInfo->hMapping, FILE_MAP_READ, 0, 0, 0, NULL);
    if (BasePtr == NULL) {
      CloseHandle(MapInfo->hMapping);
      MapInfo->hMapping = NULL;
      MakeErrMsg(ErrMsg, std::string("Can't map file: ") + Path.toString());
      return 0;
    }
  }
  return BasePtr;
}

size_t MappedFile::size() const {
  assert(MapInfo && "MappedFile not initialized");
  return MapInfo->size;
}

}

