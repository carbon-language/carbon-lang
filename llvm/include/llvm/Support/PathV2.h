//===- llvm/Support/PathV2.h - Path Operating System Concept ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::path namespace. It is designed after
// TR2/boost filesystem (v3), but modified to remove exception handling and the
// path class.
//
// All functions return void and their actual work via the last out argument.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PATHV2_H
#define LLVM_SUPPORT_PATHV2_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DataTypes.h"
#include <iterator>

namespace llvm {
namespace sys {
namespace path {

/// @name Lexical Component Iterator
/// @{

/// @brief Path iterator.
///
/// This is a bidirectional iterator that iterates over the individual
/// components in \a path. The forward traversal order is as follows:
/// * The root-name element, if present.
/// * The root-directory element, if present.
/// * Each successive filename element, if present.
/// * Dot, if one or more trailing non-root slash characters are present.
/// The backwards traversal order is the reverse of forward traversal.
///
/// Iteration examples. Each component is separated by ',':
/// /          => /
/// /foo       => /,foo
/// foo/       => foo,.
/// /foo/bar   => /,foo,bar
/// ../        => ..,.
/// C:\foo\bar => C:,/,foo,bar
///
class const_iterator {
  StringRef Path;      //< The entire path.
  StringRef Component; //< The current component. Not necessarily in Path.
  size_t    Position;  //< The iterators current position within Path.

  // An end iterator has Position = Path.size() + 1.
  friend const_iterator begin(const StringRef &path);
  friend const_iterator end(const StringRef &path);

public:
  typedef const StringRef value_type;
  typedef ptrdiff_t difference_type;
  typedef value_type &reference;
  typedef value_type *pointer;
  typedef std::bidirectional_iterator_tag iterator_category;

  reference operator*() const { return Component; }
  pointer   operator->() const { return &Component; }
  const_iterator &operator++();    // preincrement
  const_iterator &operator++(int); // postincrement
  const_iterator &operator--();    // predecrement
  const_iterator &operator--(int); // postdecrement
  bool operator==(const const_iterator &RHS) const;
  bool operator!=(const const_iterator &RHS) const;

  /// @brief Difference in bytes between this and RHS.
  ptrdiff_t operator-(const const_iterator &RHS) const;
};

typedef std::reverse_iterator<const_iterator> reverse_iterator;

/// @brief Get begin iterator over \a path.
/// @param path Input path.
/// @returns Iterator initialized with the first component of \a path.
const_iterator begin(const StringRef &path);

/// @brief Get end iterator over \a path.
/// @param path Input path.
/// @returns Iterator initialized to the end of \a path.
const_iterator end(const StringRef &path);

/// @brief Get reverse begin iterator over \a path.
/// @param path Input path.
/// @returns Iterator initialized with the first reverse component of \a path.
inline reverse_iterator rbegin(const StringRef &path) {
  return reverse_iterator(end(path));
}

/// @brief Get reverse end iterator over \a path.
/// @param path Input path.
/// @returns Iterator initialized to the reverse end of \a path.
inline reverse_iterator rend(const StringRef &path) {
  return reverse_iterator(begin(path));
}

/// @}
/// @name Lexical Modifiers
/// @{

/// @brief Remove the last component from \a path unless it is the root dir.
///
/// directory/filename.cpp => directory/
/// directory/             => directory
/// /                      => /
///
/// @param path A path that is modified to not have a file component.
void remove_filename(SmallVectorImpl<char> &path);

/// @brief Replace the file extension of \a path with \a extension.
///
/// ./filename.cpp => ./filename.extension
/// ./filename     => ./filename.extension
/// ./             => ./.extension
///
/// @param path A path that has its extension replaced with \a extension.
/// @param extension The extension to be added. It may be empty. It may also
///                  optionally start with a '.', if it does not, one will be
///                  prepended.
void replace_extension(SmallVectorImpl<char> &path,
                             const Twine &extension);

/// @brief Append to path.
///
/// /foo  + bar/f => /foo/bar/f
/// /foo/ + bar/f => /foo/bar/f
/// foo   + bar/f => foo/bar/f
///
/// @param path Set to \a path + \a component.
/// @param component The component to be appended to \a path.
void append(SmallVectorImpl<char> &path, const Twine &a,
                                               const Twine &b = "",
                                               const Twine &c = "",
                                               const Twine &d = "");

/// @brief Append to path.
///
/// /foo  + [bar,f] => /foo/bar/f
/// /foo/ + [bar,f] => /foo/bar/f
/// foo   + [bar,f] => foo/bar/f
///
/// @param path Set to \a path + [\a begin, \a end).
/// @param begin Start of components to append.
/// @param end One past the end of components to append.
void append(SmallVectorImpl<char> &path,
                  const_iterator begin, const_iterator end);

/// @}
/// @name Transforms (or some other better name)
/// @{

/// Convert path to the native form. This is used to give paths to users and
/// operating system calls in the platform's normal way. For example, on Windows
/// all '/' are converted to '\'.
///
/// @param path A path that is transformed to native format.
/// @param result Holds the result of the transformation.
void native(const Twine &path, SmallVectorImpl<char> &result);

/// @}
/// @name Lexical Observers
/// @{

/// @brief Get root name.
///
/// //net/hello => //net
/// c:/hello    => c: (on Windows, on other platforms nothing)
/// /hello      => <empty>
///
/// @param path Input path.
/// @param result Set to the root name of \a path if it has one, otherwise "".
void root_name(const StringRef &path, StringRef &result);

/// @brief Get root directory.
///
/// /goo/hello => /
/// c:/hello   => /
/// d/file.txt => <empty>
///
/// @param path Input path.
/// @param result Set to the root directory of \a path if it has one, otherwise
///               "".
void root_directory(const StringRef &path, StringRef &result);

/// @brief Get root path.
///
/// Equivalent to root_name + root_directory.
///
/// @param path Input path.
/// @param result Set to the root path of \a path if it has one, otherwise "".
void root_path(const StringRef &path, StringRef &result);

/// @brief Get relative path.
///
/// C:\hello\world => hello\world
/// foo/bar        => foo/bar
/// /foo/bar       => foo/bar
///
/// @param path Input path.
/// @param result Set to the path starting after root_path if one exists,
///               otherwise "".
void relative_path(const StringRef &path, StringRef &result);

/// @brief Get parent path.
///
/// /          => <empty>
/// /foo       => /
/// foo/../bar => foo/..
///
/// @param path Input path.
/// @param result Set to the parent path of \a path if one exists, otherwise "".
void parent_path(const StringRef &path, StringRef &result);

/// @brief Get filename.
///
/// /foo.txt    => foo.txt
/// .          => .
/// ..         => ..
/// /          => /
///
/// @param path Input path.
/// @param result Set to the filename part of \a path. This is defined as the
///               last component of \a path.
void filename(const StringRef &path, StringRef &result);

/// @brief Get stem.
///
/// If filename contains a dot but not solely one or two dots, result is the
/// substring of filename ending at (but not including) the last dot. Otherwise
/// it is filename.
///
/// /foo/bar.txt => bar
/// /foo/bar     => bar
/// /foo/.txt    => <empty>
/// /foo/.       => .
/// /foo/..      => ..
///
/// @param path Input path.
/// @param result Set to the stem of \a path.
void stem(const StringRef &path, StringRef &result);

/// @brief Get extension.
///
/// If filename contains a dot but not solely one or two dots, result is the
/// substring of filename starting at (and including) the last dot, and ending
/// at the end of \a path. Otherwise "".
///
/// /foo/bar.txt => .txt
/// /foo/bar     => <empty>
/// /foo/.txt    => .txt
///
/// @param path Input path.
/// @param result Set to the extension of \a path.
void extension(const StringRef &path, StringRef &result);

/// @brief Has root name?
///
/// root_name != ""
///
/// @param path Input path.
/// @param result Set to true if the path has a root name, false otherwise.
void has_root_name(const Twine &path, bool &result);

/// @brief Has root directory?
///
/// root_directory != ""
///
/// @param path Input path.
/// @param result Set to true if the path has a root directory, false otherwise.
void has_root_directory(const Twine &path, bool &result);

/// @brief Has root path?
///
/// root_path != ""
///
/// @param path Input path.
/// @param result Set to true if the path has a root path, false otherwise.
void has_root_path(const Twine &path, bool &result);

/// @brief Has relative path?
///
/// relative_path != ""
///
/// @param path Input path.
/// @param result Set to true if the path has a relative path, false otherwise.
void has_relative_path(const Twine &path, bool &result);

/// @brief Has parent path?
///
/// parent_path != ""
///
/// @param path Input path.
/// @param result Set to true if the path has a parent path, false otherwise.
void has_parent_path(const Twine &path, bool &result);

/// @brief Has filename?
///
/// filename != ""
///
/// @param path Input path.
/// @param result Set to true if the path has a filename, false otherwise.
void has_filename(const Twine &path, bool &result);

/// @brief Has stem?
///
/// stem != ""
///
/// @param path Input path.
/// @param result Set to true if the path has a stem, false otherwise.
void has_stem(const Twine &path, bool &result);

/// @brief Has extension?
///
/// extension != ""
///
/// @param path Input path.
/// @param result Set to true if the path has a extension, false otherwise.
void has_extension(const Twine &path, bool &result);

/// @brief Is path absolute?
///
/// @param path Input path.
/// @param result Set to true if the path is absolute, false if it is not.
void is_absolute(const Twine &path, bool &result);

/// @brief Is path relative?
///
/// @param path Input path.
/// @param result Set to true if the path is relative, false if it is not.
void is_relative(const Twine &path, bool &result);

} // end namespace path
} // end namespace sys
} // end namespace llvm

#endif
