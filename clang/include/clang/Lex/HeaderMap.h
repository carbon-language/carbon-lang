//===--- HeaderMap.h - A file that acts like dir of symlinks ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the HeaderMap interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_HEADERMAP_H
#define LLVM_CLANG_LEX_HEADERMAP_H

namespace clang {

/// This class represents an Apple concept known as a 'header map'.  To the
/// #include file resolution process, it basically acts like a directory of
/// symlinks to files.  Its advantages are that it is dense and more efficient
/// to create and process than a directory of symlinks.
class HeaderMap {
public:
  /// HeaderMap::Create - This attempts to load the specified file as a header
  /// map.  If it doesn't look like a HeaderMap, it gives up and returns null.
  /// If it looks like a HeaderMap but is obviously corrupted, it puts a reason
  /// into the string error argument and returns null.
  static const HeaderMap *Create(const FileEntry *FE, std::string &ErrorInfo) { 
    // FIXME: woot!
    return 0; 
  }
};

} // end namespace clang.

#endif