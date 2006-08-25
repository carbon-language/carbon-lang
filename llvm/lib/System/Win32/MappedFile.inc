//===- Win32/MappedFile.cpp - Win32 MappedFile Implementation ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Jeff Cohen and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
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
  assert(!info_);
  info_ = new MappedFileInfo;
  info_->hFile = INVALID_HANDLE_VALUE;
  info_->hMapping = NULL;

  DWORD mode = options_ & WRITE_ACCESS ? GENERIC_WRITE : GENERIC_READ;
  DWORD disposition = options_ & WRITE_ACCESS ? OPEN_ALWAYS : OPEN_EXISTING;
  DWORD share = options_ & WRITE_ACCESS ? FILE_SHARE_WRITE : FILE_SHARE_READ;
  share = options_ & SHARED_MAPPING ? share : 0;
  info_->hFile = CreateFile(path_.c_str(), mode, share, NULL, disposition,
                            FILE_ATTRIBUTE_NORMAL, NULL);
  if (info_->hFile == INVALID_HANDLE_VALUE) {
    delete info_;
    info_ = NULL;
    return MakeErrMsg(ErrMsg,
      std::string("Can't open file: ") + path_.toString());
  }

  LARGE_INTEGER size;
  if (!GetFileSizeEx(info_->hFile, &size) ||
      (info_->size = size_t(size.QuadPart), info_->size != size.QuadPart)) {
    CloseHandle(info_->hFile);
    delete info_;
    info_ = NULL;
    return MakeErrMsg(ErrMsg, 
      std::string("Can't get size of file: ") + path_.toString());
  }
}

void MappedFile::terminate() {
  unmap();
  if (info_->hFile != INVALID_HANDLE_VALUE)
    CloseHandle(info_->hFile);
  delete info_;
  info_ = NULL;
}

void MappedFile::unmap() {
  assert(info_ && "MappedFile not initialized");
  if (isMapped()) {
    UnmapViewOfFile(base_);
    base_ = NULL;
  }
  if (info_->hMapping != INVALID_HANDLE_VALUE) {
    CloseHandle(info_->hMapping);
    info_->hMapping = NULL;
  }
}

void* MappedFile::map(std::string* ErrMsg) {
  if (!isMapped()) {
    DWORD prot = PAGE_READONLY;
    if (options_ & EXEC_ACCESS)
      prot = SEC_IMAGE;
    else if (options_ & WRITE_ACCESS)
      prot = PAGE_READWRITE;
    info_->hMapping = CreateFileMapping(info_->hFile, NULL, prot, 0, 0, NULL);
    if (info_->hMapping == NULL) {
      MakeErrMsg(ErrMsg, std::string("Can't map file: ") + path_.toString());
      return 0;
    }

    prot = (options_ & WRITE_ACCESS) ? FILE_MAP_WRITE : FILE_MAP_READ;
    base_ = MapViewOfFileEx(info_->hMapping, prot, 0, 0, 0, NULL);
    if (base_ == NULL) {
      CloseHandle(info_->hMapping);
      info_->hMapping = NULL;
      MakeErrMsg(ErrMsg, std::string("Can't map file: ") + path_.toString());
      return 0;
    }
  }
  return base_;
}

size_t MappedFile::size() const {
  assert(info_ && "MappedFile not initialized");
  return info_->size;
}

bool MappedFile::size(size_t new_size, std::string* ErrMsg) {
  assert(info_ && "MappedFile not initialized");

  // Take the mapping out of memory.
  unmap();

  // Adjust the new_size to a page boundary.
  size_t pagesizem1 = Process::GetPageSize() - 1;
  new_size = (new_size + pagesizem1) & ~pagesizem1;

  // If the file needs to be extended, do so.
  if (new_size > info_->size) {
    LARGE_INTEGER eof;
    eof.QuadPart = new_size;
    if (!SetFilePointerEx(info_->hFile, eof, NULL, FILE_BEGIN))
      return MakeErrMsg(ErrMsg, 
        std::string("Can't set end of file: ") + path_.toString());
    if (!SetEndOfFile(info_->hFile))
      return MakeErrMsg(ErrMsg, 
        std::string("Can't set end of file: ") + path_.toString());
    info_->size = new_size;
  }

  // Remap the file.
  return map(ErrMsg);
}

}

