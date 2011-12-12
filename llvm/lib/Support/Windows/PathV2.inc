//===- llvm/Support/Windows/PathV2.inc - Windows Path Impl ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Windows specific implementation of the PathV2 API.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic Windows code that
//===          is guaranteed to work on *all* Windows variants.
//===----------------------------------------------------------------------===//

#include "Windows.h"
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <sys/types.h>

// MinGW doesn't define this.
#ifndef _ERRNO_T_DEFINED
#define _ERRNO_T_DEFINED
typedef int errno_t;
#endif

using namespace llvm;

namespace {
  typedef BOOLEAN (WINAPI *PtrCreateSymbolicLinkW)(
    /*__in*/ LPCWSTR lpSymlinkFileName,
    /*__in*/ LPCWSTR lpTargetFileName,
    /*__in*/ DWORD dwFlags);

  PtrCreateSymbolicLinkW create_symbolic_link_api = PtrCreateSymbolicLinkW(
    ::GetProcAddress(::GetModuleHandleA("kernel32.dll"),
                     "CreateSymbolicLinkW"));

  error_code UTF8ToUTF16(StringRef utf8, SmallVectorImpl<wchar_t> &utf16) {
    int len = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                    utf8.begin(), utf8.size(),
                                    utf16.begin(), 0);

    if (len == 0)
      return windows_error(::GetLastError());

    utf16.reserve(len + 1);
    utf16.set_size(len);

    len = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                    utf8.begin(), utf8.size(),
                                    utf16.begin(), utf16.size());

    if (len == 0)
      return windows_error(::GetLastError());

    // Make utf16 null terminated.
    utf16.push_back(0);
    utf16.pop_back();

    return success;
  }

  error_code UTF16ToUTF8(const wchar_t *utf16, size_t utf16_len,
                               SmallVectorImpl<char> &utf8) {
    // Get length.
    int len = ::WideCharToMultiByte(CP_UTF8, 0,
                                    utf16, utf16_len,
                                    utf8.begin(), 0,
                                    NULL, NULL);

    if (len == 0)
      return windows_error(::GetLastError());

    utf8.reserve(len);
    utf8.set_size(len);

    // Now do the actual conversion.
    len = ::WideCharToMultiByte(CP_UTF8, 0,
                                utf16, utf16_len,
                                utf8.data(), utf8.size(),
                                NULL, NULL);

    if (len == 0)
      return windows_error(::GetLastError());

    // Make utf8 null terminated.
    utf8.push_back(0);
    utf8.pop_back();

    return success;
  }

  error_code TempDir(SmallVectorImpl<wchar_t> &result) {
  retry_temp_dir:
    DWORD len = ::GetTempPathW(result.capacity(), result.begin());

    if (len == 0)
      return windows_error(::GetLastError());

    if (len > result.capacity()) {
      result.reserve(len);
      goto retry_temp_dir;
    }

    result.set_size(len);
    return success;
  }

  bool is_separator(const wchar_t value) {
    switch (value) {
    case L'\\':
    case L'/':
      return true;
    default:
      return false;
    }
  }
}

namespace llvm {
namespace sys  {
namespace fs {

error_code current_path(SmallVectorImpl<char> &result) {
  SmallVector<wchar_t, 128> cur_path;
  cur_path.reserve(128);
retry_cur_dir:
  DWORD len = ::GetCurrentDirectoryW(cur_path.capacity(), cur_path.data());

  // A zero return value indicates a failure other than insufficient space.
  if (len == 0)
    return windows_error(::GetLastError());

  // If there's insufficient space, the len returned is larger than the len
  // given.
  if (len > cur_path.capacity()) {
    cur_path.reserve(len);
    goto retry_cur_dir;
  }

  cur_path.set_size(len);
  // cur_path now holds the current directory in utf-16. Convert to utf-8.

  // Find out how much space we need. Sadly, this function doesn't return the
  // size needed unless you tell it the result size is 0, which means you
  // _always_ have to call it twice.
  len = ::WideCharToMultiByte(CP_UTF8, 0,
                              cur_path.data(), cur_path.size(),
                              result.data(), 0,
                              NULL, NULL);

  if (len == 0)
    return make_error_code(windows_error(::GetLastError()));

  result.reserve(len);
  result.set_size(len);
  // Now do the actual conversion.
  len = ::WideCharToMultiByte(CP_UTF8, 0,
                              cur_path.data(), cur_path.size(),
                              result.data(), result.size(),
                              NULL, NULL);
  if (len == 0)
    return windows_error(::GetLastError());

  return success;
}

error_code copy_file(const Twine &from, const Twine &to, copy_option copt) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toStringRef(from_storage);
  StringRef t = to.toStringRef(to_storage);

  // Convert to utf-16.
  SmallVector<wchar_t, 128> wide_from;
  SmallVector<wchar_t, 128> wide_to;
  if (error_code ec = UTF8ToUTF16(f, wide_from)) return ec;
  if (error_code ec = UTF8ToUTF16(t, wide_to)) return ec;

  // Copy the file.
  BOOL res = ::CopyFileW(wide_from.begin(), wide_to.begin(),
                         copt != copy_option::overwrite_if_exists);

  if (res == 0)
    return windows_error(::GetLastError());

  return success;
}

error_code create_directory(const Twine &path, bool &existed) {
  SmallString<128> path_storage;
  SmallVector<wchar_t, 128> path_utf16;

  if (error_code ec = UTF8ToUTF16(path.toStringRef(path_storage),
                                  path_utf16))
    return ec;

  if (!::CreateDirectoryW(path_utf16.begin(), NULL)) {
    error_code ec = windows_error(::GetLastError());
    if (ec == windows_error::already_exists)
      existed = true;
    else
      return ec;
  } else
    existed = false;

  return success;
}

error_code create_hard_link(const Twine &to, const Twine &from) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toStringRef(from_storage);
  StringRef t = to.toStringRef(to_storage);

  // Convert to utf-16.
  SmallVector<wchar_t, 128> wide_from;
  SmallVector<wchar_t, 128> wide_to;
  if (error_code ec = UTF8ToUTF16(f, wide_from)) return ec;
  if (error_code ec = UTF8ToUTF16(t, wide_to)) return ec;

  if (!::CreateHardLinkW(wide_from.begin(), wide_to.begin(), NULL))
    return windows_error(::GetLastError());

  return success;
}

error_code create_symlink(const Twine &to, const Twine &from) {
  // Only do it if the function is available at runtime.
  if (!create_symbolic_link_api)
    return make_error_code(errc::function_not_supported);

  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toStringRef(from_storage);
  StringRef t = to.toStringRef(to_storage);

  // Convert to utf-16.
  SmallVector<wchar_t, 128> wide_from;
  SmallVector<wchar_t, 128> wide_to;
  if (error_code ec = UTF8ToUTF16(f, wide_from)) return ec;
  if (error_code ec = UTF8ToUTF16(t, wide_to)) return ec;

  if (!create_symbolic_link_api(wide_from.begin(), wide_to.begin(), 0))
    return windows_error(::GetLastError());

  return success;
}

error_code remove(const Twine &path, bool &existed) {
  SmallString<128> path_storage;
  SmallVector<wchar_t, 128> path_utf16;

  file_status st;
  if (error_code ec = status(path, st))
    return ec;

  if (error_code ec = UTF8ToUTF16(path.toStringRef(path_storage),
                                  path_utf16))
    return ec;

  if (st.type() == file_type::directory_file) {
    if (!::RemoveDirectoryW(c_str(path_utf16))) {
      error_code ec = windows_error(::GetLastError());
      if (ec != windows_error::file_not_found)
        return ec;
      existed = false;
    } else
      existed = true;
  } else {
    if (!::DeleteFileW(c_str(path_utf16))) {
      error_code ec = windows_error(::GetLastError());
      if (ec != windows_error::file_not_found)
        return ec;
      existed = false;
    } else
      existed = true;
  }

  return success;
}

error_code rename(const Twine &from, const Twine &to) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toStringRef(from_storage);
  StringRef t = to.toStringRef(to_storage);

  // Convert to utf-16.
  SmallVector<wchar_t, 128> wide_from;
  SmallVector<wchar_t, 128> wide_to;
  if (error_code ec = UTF8ToUTF16(f, wide_from)) return ec;
  if (error_code ec = UTF8ToUTF16(t, wide_to)) return ec;

  if (!::MoveFileExW(wide_from.begin(), wide_to.begin(),
                     MOVEFILE_COPY_ALLOWED | MOVEFILE_REPLACE_EXISTING))
    return windows_error(::GetLastError());

  return success;
}

error_code resize_file(const Twine &path, uint64_t size) {
  SmallString<128> path_storage;
  SmallVector<wchar_t, 128> path_utf16;

  if (error_code ec = UTF8ToUTF16(path.toStringRef(path_storage),
                                  path_utf16))
    return ec;

  int fd = ::_wopen(path_utf16.begin(), O_BINARY, S_IREAD | S_IWRITE);
  if (fd == -1)
    return error_code(errno, generic_category());
#ifdef HAVE__CHSIZE_S
  errno_t error = ::_chsize_s(fd, size);
#else
  errno_t error = ::_chsize(fd, size);
#endif
  ::close(fd);
  return error_code(error, generic_category());
}

error_code exists(const Twine &path, bool &result) {
  SmallString<128> path_storage;
  SmallVector<wchar_t, 128> path_utf16;

  if (error_code ec = UTF8ToUTF16(path.toStringRef(path_storage),
                                  path_utf16))
    return ec;

  DWORD attributes = ::GetFileAttributesW(path_utf16.begin());

  if (attributes == INVALID_FILE_ATTRIBUTES) {
    // See if the file didn't actually exist.
    error_code ec = make_error_code(windows_error(::GetLastError()));
    if (ec != windows_error::file_not_found &&
        ec != windows_error::path_not_found)
      return ec;
    result = false;
  } else
    result = true;
  return success;
}

error_code equivalent(const Twine &A, const Twine &B, bool &result) {
  // Get arguments.
  SmallString<128> a_storage;
  SmallString<128> b_storage;
  StringRef a = A.toStringRef(a_storage);
  StringRef b = B.toStringRef(b_storage);

  // Convert to utf-16.
  SmallVector<wchar_t, 128> wide_a;
  SmallVector<wchar_t, 128> wide_b;
  if (error_code ec = UTF8ToUTF16(a, wide_a)) return ec;
  if (error_code ec = UTF8ToUTF16(b, wide_b)) return ec;

  ScopedFileHandle HandleB(
    ::CreateFileW(wide_b.begin(),
                  0,
                  FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
                  0,
                  OPEN_EXISTING,
                  FILE_FLAG_BACKUP_SEMANTICS,
                  0));

  ScopedFileHandle HandleA(
    ::CreateFileW(wide_a.begin(),
                  0,
                  FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
                  0,
                  OPEN_EXISTING,
                  FILE_FLAG_BACKUP_SEMANTICS,
                  0));

  // If both handles are invalid, it's an error.
  if (!HandleA && !HandleB)
    return windows_error(::GetLastError());

  // If only one is invalid, it's false.
  if (!HandleA || !HandleB) {
    result = false;
    return success;
  }

  // Get file information.
  BY_HANDLE_FILE_INFORMATION InfoA, InfoB;
  if (!::GetFileInformationByHandle(HandleA, &InfoA))
    return windows_error(::GetLastError());
  if (!::GetFileInformationByHandle(HandleB, &InfoB))
    return windows_error(::GetLastError());

  // See if it's all the same.
  result =
    InfoA.dwVolumeSerialNumber           == InfoB.dwVolumeSerialNumber &&
    InfoA.nFileIndexHigh                 == InfoB.nFileIndexHigh &&
    InfoA.nFileIndexLow                  == InfoB.nFileIndexLow &&
    InfoA.nFileSizeHigh                  == InfoB.nFileSizeHigh &&
    InfoA.nFileSizeLow                   == InfoB.nFileSizeLow &&
    InfoA.ftLastWriteTime.dwLowDateTime  ==
      InfoB.ftLastWriteTime.dwLowDateTime &&
    InfoA.ftLastWriteTime.dwHighDateTime ==
      InfoB.ftLastWriteTime.dwHighDateTime;

  return success;
}

error_code file_size(const Twine &path, uint64_t &result) {
  SmallString<128> path_storage;
  SmallVector<wchar_t, 128> path_utf16;

  if (error_code ec = UTF8ToUTF16(path.toStringRef(path_storage),
                                  path_utf16))
    return ec;

  WIN32_FILE_ATTRIBUTE_DATA FileData;
  if (!::GetFileAttributesExW(path_utf16.begin(),
                              ::GetFileExInfoStandard,
                              &FileData))
    return windows_error(::GetLastError());

  result =
    (uint64_t(FileData.nFileSizeHigh) << (sizeof(FileData.nFileSizeLow) * 8))
    + FileData.nFileSizeLow;

  return success;
}

static bool isReservedName(StringRef path) {
  // This list of reserved names comes from MSDN, at:
  // http://msdn.microsoft.com/en-us/library/aa365247%28v=vs.85%29.aspx
  static const char *sReservedNames[] = { "nul", "con", "prn", "aux",
                              "com1", "com2", "com3", "com4", "com5", "com6",
                              "com7", "com8", "com9", "lpt1", "lpt2", "lpt3",
                              "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9" };

  // First, check to see if this is a device namespace, which always
  // starts with \\.\, since device namespaces are not legal file paths.
  if (path.startswith("\\\\.\\"))
    return true;

  // Then compare against the list of ancient reserved names
  for (size_t i = 0; i < sizeof(sReservedNames) / sizeof(const char *); ++i) {
    if (path.equals_lower(sReservedNames[i]))
      return true;
  }

  // The path isn't what we consider reserved.
  return false;
}

error_code status(const Twine &path, file_status &result) {
  SmallString<128> path_storage;
  SmallVector<wchar_t, 128> path_utf16;

  StringRef path8 = path.toStringRef(path_storage);
  if (isReservedName(path8)) {
    result = file_status(file_type::character_file);
    return success;
  }

  if (error_code ec = UTF8ToUTF16(path8,
                                  path_utf16))
    return ec;

  DWORD attr = ::GetFileAttributesW(path_utf16.begin());
  if (attr == INVALID_FILE_ATTRIBUTES)
    goto handle_status_error;

  // Handle reparse points.
  if (attr & FILE_ATTRIBUTE_REPARSE_POINT) {
    ScopedFileHandle h(
      ::CreateFileW(path_utf16.begin(),
                    0, // Attributes only.
                    FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
                    NULL,
                    OPEN_EXISTING,
                    FILE_FLAG_BACKUP_SEMANTICS,
                    0));
    if (!h)
      goto handle_status_error;
  }

  if (attr & FILE_ATTRIBUTE_DIRECTORY)
    result = file_status(file_type::directory_file);
  else
    result = file_status(file_type::regular_file);

  return success;

handle_status_error:
  error_code ec = windows_error(::GetLastError());
  if (ec == windows_error::file_not_found ||
      ec == windows_error::path_not_found)
    result = file_status(file_type::file_not_found);
  else if (ec == windows_error::sharing_violation)
    result = file_status(file_type::type_unknown);
  else {
    result = file_status(file_type::status_error);
    return ec;
  }

  return success;
}

error_code unique_file(const Twine &model, int &result_fd,
                             SmallVectorImpl<char> &result_path,
                             bool makeAbsolute) {
  // Use result_path as temp storage.
  result_path.set_size(0);
  StringRef m = model.toStringRef(result_path);

  SmallVector<wchar_t, 128> model_utf16;
  if (error_code ec = UTF8ToUTF16(m, model_utf16)) return ec;

  if (makeAbsolute) {
    // Make model absolute by prepending a temp directory if it's not already.
    bool absolute = path::is_absolute(m);

    if (!absolute) {
      SmallVector<wchar_t, 64> temp_dir;
      if (error_code ec = TempDir(temp_dir)) return ec;
      // Handle c: by removing it.
      if (model_utf16.size() > 2 && model_utf16[1] == L':') {
        model_utf16.erase(model_utf16.begin(), model_utf16.begin() + 2);
      }
      model_utf16.insert(model_utf16.begin(), temp_dir.begin(), temp_dir.end());
    }
  }

  // Replace '%' with random chars. From here on, DO NOT modify model. It may be
  // needed if the randomly chosen path already exists.
  SmallVector<wchar_t, 128> random_path_utf16;

  // Get a Crypto Provider for CryptGenRandom.
  HCRYPTPROV HCPC;
  if (!::CryptAcquireContextW(&HCPC,
                              NULL,
                              NULL,
                              PROV_RSA_FULL,
                              CRYPT_VERIFYCONTEXT))
    return windows_error(::GetLastError());
  ScopedCryptContext CryptoProvider(HCPC);

retry_random_path:
  random_path_utf16.set_size(0);
  for (SmallVectorImpl<wchar_t>::const_iterator i = model_utf16.begin(),
                                                e = model_utf16.end();
                                                i != e; ++i) {
    if (*i == L'%') {
      BYTE val = 0;
      if (!::CryptGenRandom(CryptoProvider, 1, &val))
          return windows_error(::GetLastError());
      random_path_utf16.push_back("0123456789abcdef"[val & 15]);
    }
    else
      random_path_utf16.push_back(*i);
  }
  // Make random_path_utf16 null terminated.
  random_path_utf16.push_back(0);
  random_path_utf16.pop_back();

  // Try to create + open the path.
retry_create_file:
  HANDLE TempFileHandle = ::CreateFileW(random_path_utf16.begin(),
                                        GENERIC_READ | GENERIC_WRITE,
                                        FILE_SHARE_READ,
                                        NULL,
                                        // Return ERROR_FILE_EXISTS if the file
                                        // already exists.
                                        CREATE_NEW,
                                        FILE_ATTRIBUTE_TEMPORARY,
                                        NULL);
  if (TempFileHandle == INVALID_HANDLE_VALUE) {
    // If the file existed, try again, otherwise, error.
    error_code ec = windows_error(::GetLastError());
    if (ec == windows_error::file_exists)
      goto retry_random_path;
    // Check for non-existing parent directories.
    if (ec == windows_error::path_not_found) {
      // Create the directories using result_path as temp storage.
      if (error_code ec = UTF16ToUTF8(random_path_utf16.begin(),
                                      random_path_utf16.size(), result_path))
        return ec;
      StringRef p(result_path.begin(), result_path.size());
      SmallString<64> dir_to_create;
      for (path::const_iterator i = path::begin(p),
                                e = --path::end(p); i != e; ++i) {
        path::append(dir_to_create, *i);
        bool Exists;
        if (error_code ec = exists(Twine(dir_to_create), Exists)) return ec;
        if (!Exists) {
          // If c: doesn't exist, bail.
          if (i->endswith(":"))
            return ec;

          SmallVector<wchar_t, 64> dir_to_create_utf16;
          if (error_code ec = UTF8ToUTF16(dir_to_create, dir_to_create_utf16))
            return ec;

          // Create the directory.
          if (!::CreateDirectoryW(dir_to_create_utf16.begin(), NULL))
            return windows_error(::GetLastError());
        }
      }
      goto retry_create_file;
    }
    return ec;
  }

  // Set result_path to the utf-8 representation of the path.
  if (error_code ec = UTF16ToUTF8(random_path_utf16.begin(),
                                  random_path_utf16.size(), result_path)) {
    ::CloseHandle(TempFileHandle);
    ::DeleteFileW(random_path_utf16.begin());
    return ec;
  }

  // Convert the Windows API file handle into a C-runtime handle.
  int fd = ::_open_osfhandle(intptr_t(TempFileHandle), 0);
  if (fd == -1) {
    ::CloseHandle(TempFileHandle);
    ::DeleteFileW(random_path_utf16.begin());
    // MSDN doesn't say anything about _open_osfhandle setting errno or
    // GetLastError(), so just return invalid_handle.
    return windows_error::invalid_handle;
  }

  result_fd = fd;
  return success;
}

error_code get_magic(const Twine &path, uint32_t len,
                     SmallVectorImpl<char> &result) {
  SmallString<128> path_storage;
  SmallVector<wchar_t, 128> path_utf16;
  result.set_size(0);

  // Convert path to UTF-16.
  if (error_code ec = UTF8ToUTF16(path.toStringRef(path_storage),
                                  path_utf16))
    return ec;

  // Open file.
  HANDLE file = ::CreateFileW(c_str(path_utf16),
                              GENERIC_READ,
                              FILE_SHARE_READ,
                              NULL,
                              OPEN_EXISTING,
                              FILE_ATTRIBUTE_READONLY,
                              NULL);
  if (file == INVALID_HANDLE_VALUE)
    return windows_error(::GetLastError());

  // Allocate buffer.
  result.reserve(len);

  // Get magic!
  DWORD bytes_read = 0;
  BOOL read_success = ::ReadFile(file, result.data(), len, &bytes_read, NULL);
  error_code ec = windows_error(::GetLastError());
  ::CloseHandle(file);
  if (!read_success || (bytes_read != len)) {
    // Set result size to the number of bytes read if it's valid.
    if (bytes_read <= len)
      result.set_size(bytes_read);
    // ERROR_HANDLE_EOF is mapped to errc::value_too_large.
    return ec;
  }

  result.set_size(len);
  return success;
}

error_code detail::directory_iterator_construct(detail::DirIterState &it,
                                                StringRef path){
  SmallVector<wchar_t, 128> path_utf16;

  if (error_code ec = UTF8ToUTF16(path,
                                  path_utf16))
    return ec;

  // Convert path to the format that Windows is happy with.
  if (path_utf16.size() > 0 &&
      !is_separator(path_utf16[path.size() - 1]) &&
      path_utf16[path.size() - 1] != L':') {
    path_utf16.push_back(L'\\');
    path_utf16.push_back(L'*');
  } else {
    path_utf16.push_back(L'*');
  }

  //  Get the first directory entry.
  WIN32_FIND_DATAW FirstFind;
  ScopedFindHandle FindHandle(::FindFirstFileW(c_str(path_utf16), &FirstFind));
  if (!FindHandle)
    return windows_error(::GetLastError());

  size_t FilenameLen = ::wcslen(FirstFind.cFileName);
  while ((FilenameLen == 1 && FirstFind.cFileName[0] == L'.') ||
         (FilenameLen == 2 && FirstFind.cFileName[0] == L'.' &&
                              FirstFind.cFileName[1] == L'.'))
    if (!::FindNextFileW(FindHandle, &FirstFind)) {
      error_code ec = windows_error(::GetLastError());
      // Check for end.
      if (ec == windows_error::no_more_files)
        return detail::directory_iterator_destruct(it);
      return ec;
    } else
      FilenameLen = ::wcslen(FirstFind.cFileName);

  // Construct the current directory entry.
  SmallString<128> directory_entry_name_utf8;
  if (error_code ec = UTF16ToUTF8(FirstFind.cFileName,
                                  ::wcslen(FirstFind.cFileName),
                                  directory_entry_name_utf8))
    return ec;

  it.IterationHandle = intptr_t(FindHandle.take());
  SmallString<128> directory_entry_path(path);
  path::append(directory_entry_path, directory_entry_name_utf8.str());
  it.CurrentEntry = directory_entry(directory_entry_path.str());

  return success;
}

error_code detail::directory_iterator_destruct(detail::DirIterState &it) {
  if (it.IterationHandle != 0)
    // Closes the handle if it's valid.
    ScopedFindHandle close(HANDLE(it.IterationHandle));
  it.IterationHandle = 0;
  it.CurrentEntry = directory_entry();
  return success;
}

error_code detail::directory_iterator_increment(detail::DirIterState &it) {
  WIN32_FIND_DATAW FindData;
  if (!::FindNextFileW(HANDLE(it.IterationHandle), &FindData)) {
    error_code ec = windows_error(::GetLastError());
    // Check for end.
    if (ec == windows_error::no_more_files)
      return detail::directory_iterator_destruct(it);
    return ec;
  }

  size_t FilenameLen = ::wcslen(FindData.cFileName);
  if ((FilenameLen == 1 && FindData.cFileName[0] == L'.') ||
      (FilenameLen == 2 && FindData.cFileName[0] == L'.' &&
                           FindData.cFileName[1] == L'.'))
    return directory_iterator_increment(it);

  SmallString<128> directory_entry_path_utf8;
  if (error_code ec = UTF16ToUTF8(FindData.cFileName,
                                  ::wcslen(FindData.cFileName),
                                  directory_entry_path_utf8))
    return ec;

  it.CurrentEntry.replace_filename(Twine(directory_entry_path_utf8));
  return success;
}

} // end namespace fs
} // end namespace sys
} // end namespace llvm
