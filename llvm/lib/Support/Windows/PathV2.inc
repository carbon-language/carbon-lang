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

using namespace llvm;

namespace {
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
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

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

} // end namespace fs
} // end namespace sys
} // end namespace llvm
