// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_DIRENT_H_
#define CARBON_COMMON_DIRENT_H_
#ifdef _WIN32

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <io.h>
#include <windows.h>

#include <string>

#ifndef DT_UNKNOWN
#define DT_UNKNOWN 0
#define DT_DIR 4
#define DT_REG 8
#define DT_LNK 10
#endif

struct dirent {
  unsigned char d_type;
  char d_name[MAX_PATH];
};

struct DIR {
  HANDLE handle;
  WIN32_FIND_DATAW find_data;
  bool has_next;
  bool first;
  dirent entry;
  int fd;  // for dirfd()
  std::wstring path;
};

inline static DIR* opendir(const char* name) {
  // Convert to wstring
  int wlen = MultiByteToWideChar(CP_UTF8, 0, name, -1, nullptr, 0);
  std::wstring wname(wlen, 0);
  MultiByteToWideChar(CP_UTF8, 0, name, -1, &wname[0], wlen);
  // Remove null terminator added by MultiByteToWideChar
  if (!wname.empty() && wname.back() == 0) {
    wname.pop_back();
  }

  std::wstring pattern = wname;
  if (!pattern.empty() && pattern.back() != L'\\') {
    pattern += L'\\';
  }
  pattern += L'*';

  DIR* dir = new DIR();
  dir->handle = FindFirstFileW(pattern.c_str(), &dir->find_data);
  if (dir->handle == INVALID_HANDLE_VALUE) {
    delete dir;
    return nullptr;
  }
  dir->has_next = true;
  dir->first = true;
  dir->path = wname;

  // Open fd for dirfd()
  dir->fd = -1;
  HANDLE h =
      CreateFileW(wname.c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  if (h != INVALID_HANDLE_VALUE) {
    dir->fd = _open_osfhandle((intptr_t)h, 0);
  }
  return dir;
}

inline static DIR* opendir(const wchar_t* wname_in) {
  std::wstring wname(wname_in);
  std::wstring pattern = wname;
  if (!pattern.empty() && pattern.back() != L'\\') {
    pattern += L'\\';
  }
  pattern += L'*';

  DIR* dir = new DIR();
  dir->handle = FindFirstFileW(pattern.c_str(), &dir->find_data);
  if (dir->handle == INVALID_HANDLE_VALUE) {
    delete dir;
    return nullptr;
  }
  dir->has_next = true;
  dir->first = true;
  dir->path = wname;
  dir->fd = -1;
  HANDLE h =
      CreateFileW(wname.c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  if (h != INVALID_HANDLE_VALUE) {
    dir->fd = _open_osfhandle((intptr_t)h, 0);
  }
  return dir;
}

inline static dirent* readdir(DIR* dir) {
  if (!dir || !dir->has_next) {
    return nullptr;
  }

  if (dir->first) {
    dir->first = false;
  } else {
    if (!FindNextFileW(dir->handle, &dir->find_data)) {
      dir->has_next = false;
      return nullptr;
    }
  }

  // Convert filename to UTF-8
  WideCharToMultiByte(CP_UTF8, 0, dir->find_data.cFileName, -1,
                      dir->entry.d_name, MAX_PATH, nullptr, nullptr);

  // Set type
  if (dir->find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
    dir->entry.d_type = DT_DIR;
  } else if (dir->find_data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
    dir->entry.d_type = DT_LNK;
  } else {
    dir->entry.d_type = DT_REG;
  }

  return &dir->entry;
}

inline static int closedir(DIR* dir) {
  if (!dir) {
    return -1;
  }
  if (dir->handle != INVALID_HANDLE_VALUE) {
    FindClose(dir->handle);
  }
  if (dir->fd != -1) {
    _close(dir->fd);
  }
  delete dir;
  return 0;
}

inline static DIR* fdopendir(int fd) {
  // Get the path from the fd using GetFinalPathNameByHandleW
  HANDLE h = (HANDLE)_get_osfhandle(fd);
  if (h == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return nullptr;
  }

  wchar_t dir_buf[32768];
  DWORD len =
      GetFinalPathNameByHandleW(h, dir_buf, 32768, FILE_NAME_NORMALIZED);
  if (len == 0) {
    errno = EBADF;
    return nullptr;
  }

  std::wstring wpath(dir_buf, len);
  // Strip \?\ prefix
  if (wpath.size() >= 4 && wpath[0] == L'\\' && wpath[1] == L'\\' &&
      wpath[2] == L'?' && wpath[3] == L'\\') {
    wpath = wpath.substr(4);
  }

  DIR* dir = opendir(wpath.c_str());
  if (dir) {
    // Close the original fd since DIR now owns its own handle
    // But keep fd open so callers can still use dirfd()
    dir->fd = fd;
  }
  return dir;
}

inline static int dirfd(DIR* dir) { return dir ? dir->fd : -1; }

#endif  // _WIN32
#endif  // CARBON_COMMON_DIRENT_H_
