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

  DWORD mode = Options & WRITE_ACCESS ? GENERIC_WRITE : GENERIC_READ;
  DWORD disposition = Options & WRITE_ACCESS ? OPEN_ALWAYS : OPEN_EXISTING;
  DWORD share = Options & WRITE_ACCESS ? FILE_SHARE_WRITE : FILE_SHARE_READ;
  share = Options & SHARED_MAPPING ? share : 0;
  MapInfo->hFile = CreateFile(Path.c_str(), mode, share, NULL, disposition,
                            FILE_ATTRIBUTE_NORMAL, NULL);
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

void* MappedFile::map(std::string* ErrMsg) {
  if (!isMapped()) {
    DWORD prot = PAGE_READONLY;
    if (Options & EXEC_ACCESS)
      prot = SEC_IMAGE;
    else if (Options & WRITE_ACCESS)
      prot = PAGE_READWRITE;
    MapInfo->hMapping = CreateFileMapping(MapInfo->hFile, NULL, prot, 0, 0, NULL);
    if (MapInfo->hMapping == NULL) {
      MakeErrMsg(ErrMsg, std::string("Can't map file: ") + Path.toString());
      return 0;
    }

    prot = (Options & WRITE_ACCESS) ? FILE_MAP_WRITE : FILE_MAP_READ;
    BasePtr = MapViewOfFileEx(MapInfo->hMapping, prot, 0, 0, 0, NULL);
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

bool MappedFile::resize(size_t new_size, std::string* ErrMsg) {
  assert(MapInfo && "MappedFile not initialized");

  // Take the mapping out of memory.
  unmap();

  // Adjust the new_size to a page boundary.
  size_t pagesizem1 = Process::GetPageSize() - 1;
  new_size = (new_size + pagesizem1) & ~pagesizem1;

  // If the file needs to be extended, do so.
  if (new_size > MapInfo->size) {
    LARGE_INTEGER eof;
    eof.QuadPart = new_size;
    if (!SetFilePointerEx(MapInfo->hFile, eof, NULL, FILE_BEGIN))
      return MakeErrMsg(ErrMsg, 
        std::string("Can't set end of file: ") + Path.toString());
    if (!SetEndOfFile(MapInfo->hFile))
      return MakeErrMsg(ErrMsg, 
        std::string("Can't set end of file: ") + Path.toString());
    MapInfo->size = new_size;
  }

  // Remap the file.
  return map(ErrMsg);
}

}

