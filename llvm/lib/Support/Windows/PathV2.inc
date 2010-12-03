//===- llvm/Support/Win32/PathV2.cpp - Windows Path Impl --------*- C++ -*-===//
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
#include <WinCrypt.h>
#include <io.h>

using namespace llvm;

namespace {
  typedef BOOLEAN (WINAPI *PtrCreateSymbolicLinkW)(
    /*__in*/ LPCWSTR lpSymlinkFileName,
    /*__in*/ LPCWSTR lpTargetFileName,
    /*__in*/ DWORD dwFlags);

  PtrCreateSymbolicLinkW create_symbolic_link_api = PtrCreateSymbolicLinkW(
    ::GetProcAddress(::GetModuleHandleA("kernel32.dll"),
                     "CreateSymbolicLinkW"));

  error_code UTF8ToUTF16(const StringRef &utf8,
                               SmallVectorImpl<wchar_t> &utf16) {
    int len = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                    utf8.begin(), utf8.size(),
                                    utf16.begin(), 0);

    if (len == 0)
      return make_error_code(windows_error(::GetLastError()));

    utf16.reserve(len + 1);
    utf16.set_size(len);

    len = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                    utf8.begin(), utf8.size(),
                                    utf16.begin(), utf16.size());

    if (len == 0)
      return make_error_code(windows_error(::GetLastError()));

    // Make utf16 null terminated.
    utf16.push_back(0);
    utf16.pop_back();

    return make_error_code(errc::success);
  }

  error_code UTF16ToUTF8(const wchar_t *utf16, size_t utf16_len,
                               SmallVectorImpl<char> &utf8) {
    // Get length.
    int len = ::WideCharToMultiByte(CP_UTF8, NULL,
                                    utf16, utf16_len,
                                    utf8.begin(), 0,
                                    NULL, NULL);

    if (len == 0)
      return make_error_code(windows_error(::GetLastError()));

    utf8.reserve(len);
    utf8.set_size(len);

    // Now do the actual conversion.
    len = ::WideCharToMultiByte(CP_UTF8, NULL,
                                utf16, utf16_len,
                                utf8.data(), utf8.size(),
                                NULL, NULL);

    if (len == 0)
      return make_error_code(windows_error(::GetLastError()));

    // Make utf8 null terminated.
    utf8.push_back(0);
    utf8.pop_back();

    return make_error_code(errc::success);
  }

  error_code TempDir(SmallVectorImpl<wchar_t> &result) {
  retry_temp_dir:
    DWORD len = ::GetTempPathW(result.capacity(), result.begin());

    if (len == 0)
      return make_error_code(windows_error(::GetLastError()));

    if (len > result.capacity()) {
      result.reserve(len);
      goto retry_temp_dir;
    }

    result.set_size(len);
    return make_error_code(errc::success);
  }

  struct AutoCryptoProvider {
    HCRYPTPROV CryptoProvider;

    ~AutoCryptoProvider() {
      ::CryptReleaseContext(CryptoProvider, 0);
    }

    operator HCRYPTPROV() const {return CryptoProvider;}
  };
}

namespace llvm {
namespace sys  {
namespace path {

error_code current_path(SmallVectorImpl<char> &result) {
  SmallVector<wchar_t, 128> cur_path;
  cur_path.reserve(128);
retry_cur_dir:
  DWORD len = ::GetCurrentDirectoryW(cur_path.capacity(), cur_path.data());

  // A zero return value indicates a failure other than insufficient space.
  if (len == 0)
    return make_error_code(windows_error(::GetLastError()));

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
  len = ::WideCharToMultiByte(CP_UTF8, NULL,
                              cur_path.data(), cur_path.size(),
                              result.data(), 0,
                              NULL, NULL);

  if (len == 0)
    return make_error_code(windows_error(::GetLastError()));

  result.reserve(len);
  result.set_size(len);
  // Now do the actual conversion.
  len = ::WideCharToMultiByte(CP_UTF8, NULL,
                              cur_path.data(), cur_path.size(),
                              result.data(), result.size(),
                              NULL, NULL);
  if (len == 0)
    return make_error_code(windows_error(::GetLastError()));

  return make_error_code(errc::success);
}

} // end namespace path

namespace fs {

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
    return make_error_code(windows_error(::GetLastError()));

  return make_error_code(errc::success);
}

error_code create_directory(const Twine &path, bool &existed) {
  SmallString<128> path_storage;
  SmallVector<wchar_t, 128> path_utf16;

  if (error_code ec = UTF8ToUTF16(path.toStringRef(path_storage),
                                  path_utf16))
    return ec;

  if (!::CreateDirectoryW(path_utf16.begin(), NULL)) {
    error_code ec = make_error_code(windows_error(::GetLastError()));
    if (ec == make_error_code(windows_error::already_exists))
      existed = true;
    else
      return ec;
  } else
    existed = false;

  return make_error_code(errc::success);
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
    return make_error_code(windows_error(::GetLastError()));

  return make_error_code(errc::success);
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

  if (!create_symbolic_link_api(wide_from.begin(), wide_to.begin(), NULL))
    return make_error_code(windows_error(::GetLastError()));

  return make_error_code(errc::success);
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
    if (ec != error_code(windows_error::file_not_found) &&
        ec != error_code(windows_error::path_not_found))
      return ec;
    result = false;
  } else
    result = true;
  return make_error_code(errc::success);
}

error_code unique_file(const Twine &model, int &result_fd,
                             SmallVectorImpl<char> &result_path) {
  // Use result_path as temp storage.
  result_path.set_size(0);
  StringRef m = model.toStringRef(result_path);

  SmallVector<wchar_t, 128> model_utf16;
  if (error_code ec = UTF8ToUTF16(m, model_utf16)) return ec;

  // Make model absolute by prepending a temp directory if it's not already.
  bool absolute;
  if (error_code ec = path::is_absolute(m, absolute)) return ec;

  if (!absolute) {
    SmallVector<wchar_t, 64> temp_dir;
    if (error_code ec = TempDir(temp_dir)) return ec;
    // Handle c: by removing it.
    if (model_utf16.size() > 2 && model_utf16[1] == L':') {
      model_utf16.erase(model_utf16.begin(), model_utf16.begin() + 2);
    }
    model_utf16.insert(model_utf16.begin(), temp_dir.begin(), temp_dir.end());
  }

  // Replace '%' with random chars. From here on, DO NOT modify model. It may be
  // needed if the randomly chosen path already exists.
  SmallVector<wchar_t, 128> random_path_utf16;

  // Get a Crypto Provider for CryptGenRandom.
  AutoCryptoProvider CryptoProvider;
  BOOL success = ::CryptAcquireContextW(&CryptoProvider.CryptoProvider,
                                        NULL,
                                        NULL,
                                        PROV_RSA_FULL,
                                        NULL);
  if (!success)
    return make_error_code(windows_error(::GetLastError()));

retry_random_path:
  random_path_utf16.set_size(0);
  for (SmallVectorImpl<wchar_t>::const_iterator i = model_utf16.begin(),
                                                e = model_utf16.end();
                                                i != e; ++i) {
    if (*i == L'%') {
      BYTE val = 0;
      if (!::CryptGenRandom(CryptoProvider, 1, &val))
          return make_error_code(windows_error(::GetLastError()));
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
    error_code ec = make_error_code(windows_error(::GetLastError()));
    if (ec == error_code(windows_error::file_exists))
      goto retry_random_path;
    // Check for non-existing parent directories.
    if (ec == error_code(windows_error::path_not_found)) {
      // Create the directories using result_path as temp storage.
      if (error_code ec = UTF16ToUTF8(random_path_utf16.begin(),
                                      random_path_utf16.size(), result_path))
        return ec;
      StringRef p(result_path.begin(), result_path.size());
      SmallString<64> dir_to_create;
      for (path::const_iterator i = path::begin(p),
                                e = --path::end(p); i != e; ++i) {
        if (error_code ec = path::append(dir_to_create, *i)) return ec;
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
            return make_error_code(windows_error(::GetLastError()));
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
    return make_error_code(windows_error::invalid_handle);
  }

  result_fd = fd;
  return make_error_code(errc::success);
}
} // end namespace fs
} // end namespace sys
} // end namespace llvm
