//===-- PathV2.cpp - Implement OS Path Concept ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the operating system PathV2 API.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PathV2.h"
#include "llvm/Support/ErrorHandling.h"
#include <cctype>

namespace {
  using llvm::StringRef;

  bool is_separator(const char value) {
    switch(value) {
#ifdef LLVM_ON_WIN32
    case '\\': // fall through
#endif
    case '/': return true;
    default: return false;
    }
  }

#ifdef LLVM_ON_WIN32
  const StringRef separators = "\\/";
  const char       prefered_separator = '\\';
#else
  const StringRef separators = "/";
  const char      prefered_separator = '/';
#endif

  StringRef find_first_component(const StringRef  &path) {
    // Look for this first component in the following order.
    // * empty (in this case we return an empty string)
    // * either C: or {//,\\}net.
    // * {/,\}
    // * {.,..}
    // * {file,directory}name

    if (path.empty())
      return path;

    // C:
    if (path.size() >= 2 && std::isalpha(path[0]) && path[1] == ':')
      return StringRef(path.begin(), 2);

    // //net
    if ((path.size() > 2) &&
        (path.startswith("\\\\") || path.startswith("//")) &&
        (path[2] != '\\' && path[2] != '/')) {
      // Find the next directory separator.
      size_t end = path.find_first_of("\\/", 2);
      if (end == StringRef::npos)
        return path;
      else
        return StringRef(path.begin(), end);
    }

    // {/,\}
    if (path[0] == '\\' || path[0] == '/')
      return StringRef(path.begin(), 1);

    if (path.startswith(".."))
      return StringRef(path.begin(), 2);

    if (path[0] == '.')
      return StringRef(path.begin(), 1);

    // * {file,directory}name
    size_t end = path.find_first_of("\\/", 2);
    if (end == StringRef::npos)
      return path;
    else
      return StringRef(path.begin(), end);

    return StringRef();
  }

  size_t filename_pos(const StringRef &str) {
    if (str.size() == 2 &&
        is_separator(str[0]) &&
        is_separator(str[1]))
      return 0;

    if (str.size() > 0 && is_separator(str[str.size() - 1]))
      return str.size() - 1;

    size_t pos = str.find_last_of(separators, str.size() - 1);

#ifdef LLVM_ON_WIN32
    if (pos == StringRef::npos)
      pos = str.find_last_of(':', str.size() - 2);
#endif

    if (pos == StringRef::npos ||
        (pos == 1 && is_separator(str[0])))
      return 0;

    return pos + 1;
  }

  size_t root_dir_start(const StringRef &str) {
    // case "c:/"
#ifdef LLVM_ON_WIN32
    if (str.size() > 2 &&
        str[1] == ':' &&
        is_separator(str[2]))
      return 2;
#endif

    // case "//"
    if (str.size() == 2 &&
        is_separator(str[0]) &&
        str[0] == str[1])
      return StringRef::npos;

    // case "//net"
    if (str.size() > 3 &&
        is_separator(str[0]) &&
        str[0] == str[1] &&
        !is_separator(str[2])) {
      return str.find_first_of(separators, 2);
    }

    // case "/"
    if (str.size() > 0 && is_separator(str[0]))
      return 0;

    return StringRef::npos;
  }

  size_t parent_path_end(const StringRef &path) {
    size_t end_pos = filename_pos(path);

    bool filename_was_sep = path.size() > 0 && is_separator(path[end_pos]);

    // Skip separators except for root dir.
    size_t root_dir_pos = root_dir_start(StringRef(path.begin(), end_pos));

    while(end_pos > 0 &&
          (end_pos - 1) != root_dir_pos &&
          is_separator(path[end_pos - 1]))
      --end_pos;

    if (end_pos == 1 && root_dir_pos == 0 && filename_was_sep)
      return StringRef::npos;

    return end_pos;
  }
}

namespace llvm {
namespace sys  {
namespace path {

const_iterator begin(const StringRef &path) {
  const_iterator i;
  i.Path      = path;
  i.Component = find_first_component(path);
  i.Position  = 0;
  return i;
}

const_iterator end(const StringRef &path) {
  const_iterator i;
  i.Path      = path;
  i.Position  = path.size();
  return i;
}

const_iterator &const_iterator::operator++() {
  assert(Position < Path.size() && "Tried to increment past end!");

  // Increment Position to past the current component
  Position += Component.size();

  // Check for end.
  if (Position == Path.size()) {
    Component = StringRef();
    return *this;
  }

  // Both POSIX and Windows treat paths that begin with exactly two separators
  // specially.
  bool was_net = Component.size() > 2 &&
    is_separator(Component[0]) &&
    Component[1] == Component[0] &&
    !is_separator(Component[2]);

  // Handle separators.
  if (is_separator(Path[Position])) {
    // Root dir.
    if (was_net
#ifdef LLVM_ON_WIN32
        // c:/
        || Component.endswith(":")
#endif
        ) {
      Component = StringRef(Path.begin() + Position, 1);
      return *this;
    }

    // Skip extra separators.
    while (Position != Path.size() &&
           is_separator(Path[Position])) {
      ++Position;
    }

    // Treat trailing '/' as a '.'.
    if (Position == Path.size()) {
      --Position;
      Component = ".";
      return *this;
    }
  }

  // Find next component.
  size_t end_pos = Path.find_first_of(separators, Position);
  if (end_pos == StringRef::npos)
    end_pos = Path.size();
  Component = StringRef(Path.begin() + Position, end_pos - Position);

  return *this;
}

const_iterator &const_iterator::operator--() {
  // If we're at the end and the previous char was a '/', return '.'.
  if (Position == Path.size() &&
      Path.size() > 1 &&
      is_separator(Path[Position - 1])
#ifdef LLVM_ON_WIN32
      && Path[Position - 2] != ':'
#endif
      ) {
    --Position;
    Component = ".";
    return *this;
  }

  // Skip separators unless it's the root directory.
  size_t root_dir_pos = root_dir_start(Path);
  size_t end_pos = Position;

  while(end_pos > 0 &&
        (end_pos - 1) != root_dir_pos &&
        is_separator(Path[end_pos - 1]))
    --end_pos;

  // Find next separator.
  size_t start_pos = filename_pos(StringRef(Path.begin(), end_pos));
  Component = StringRef(Path.begin() + start_pos, end_pos - start_pos);
  Position = start_pos;
  return *this;
}

bool const_iterator::operator==(const const_iterator &RHS) const {
  return Path.begin() == RHS.Path.begin() &&
         Position == RHS.Position;
}

bool const_iterator::operator!=(const const_iterator &RHS) const {
  return !(*this == RHS);
}

ptrdiff_t const_iterator::operator-(const const_iterator &RHS) const {
  return Position - RHS.Position;
}

error_code root_path(const StringRef &path, StringRef &result) {
  const_iterator b = begin(path),
                 pos = b,
                 e = end(path);
  if (b != e) {
    bool has_net = b->size() > 2 && is_separator((*b)[0]) && (*b)[1] == (*b)[0];
    bool has_drive =
#ifdef LLVM_ON_WIN32
      b->endswith(":");
#else
      false;
#endif

    if (has_net || has_drive) {
      if ((++pos != e) && is_separator((*pos)[0])) {
        // {C:/,//net/}, so get the first two components.
        result = StringRef(path.begin(), b->size() + pos->size());
        return make_error_code(errc::success);
      } else {
        // just {C:,//net}, return the first component.
        result = *b;
        return make_error_code(errc::success);
      }
    }

    // POSIX style root directory.
    if (is_separator((*b)[0])) {
      result = *b;
      return make_error_code(errc::success);
    }

    // No root_path.
    result = StringRef();
    return make_error_code(errc::success);
  }

  // No path :(.
  result = StringRef();
  return make_error_code(errc::success);
}

error_code root_name(const StringRef &path, StringRef &result) {
  const_iterator b = begin(path),
                 e = end(path);
  if (b != e) {
    bool has_net = b->size() > 2 && is_separator((*b)[0]) && (*b)[1] == (*b)[0];
    bool has_drive =
#ifdef LLVM_ON_WIN32
      b->endswith(":");
#else
      false;
#endif

    if (has_net || has_drive) {
      // just {C:,//net}, return the first component.
      result = *b;
      return make_error_code(errc::success);
    }
  }

  // No path or no name.
  result = StringRef();
  return make_error_code(errc::success);
}

error_code root_directory(const StringRef &path, StringRef &result) {
  const_iterator b = begin(path),
                 pos = b,
                 e = end(path);
  if (b != e) {
    bool has_net = b->size() > 2 && is_separator((*b)[0]) && (*b)[1] == (*b)[0];
    bool has_drive =
#ifdef LLVM_ON_WIN32
      b->endswith(":");
#else
      false;
#endif

    if ((has_net || has_drive) &&
        // {C:,//net}, skip to the next component.
        (++pos != e) && is_separator((*pos)[0])) {
      result = *pos;
      return make_error_code(errc::success);
    }

    // POSIX style root directory.
    if (!has_net && is_separator((*b)[0])) {
      result = *b;
      return make_error_code(errc::success);
    }
  }

  // No path or no root.
  result = StringRef();
  return make_error_code(errc::success);
}

error_code has_root_name(const Twine &path, bool &result) {
  SmallString<128> storage;
  StringRef p = path.toStringRef(storage);

  if (error_code ec = root_name(p, p)) return ec;
  result = !p.empty();
  return make_error_code(errc::success);
}

error_code has_root_directory(const Twine &path, bool &result) {
  SmallString<128> storage;
  StringRef p = path.toStringRef(storage);

  if (error_code ec = root_directory(p, p)) return ec;
  result = !p.empty();
  return make_error_code(errc::success);
}

error_code relative_path(const StringRef &path, StringRef &result) {
  StringRef root;
  if (error_code ec = root_path(path, root)) return ec;
  result = StringRef(path.begin() + root.size(), path.size() - root.size());
  return make_error_code(errc::success);
}

error_code append(SmallVectorImpl<char> &path, const Twine &a,
                                               const Twine &b,
                                               const Twine &c,
                                               const Twine &d) {
  SmallString<32> a_storage;
  SmallString<32> b_storage;
  SmallString<32> c_storage;
  SmallString<32> d_storage;

  SmallVector<StringRef, 4> components;
  if (!a.isTriviallyEmpty()) components.push_back(a.toStringRef(a_storage));
  if (!b.isTriviallyEmpty()) components.push_back(b.toStringRef(b_storage));
  if (!c.isTriviallyEmpty()) components.push_back(c.toStringRef(c_storage));
  if (!d.isTriviallyEmpty()) components.push_back(d.toStringRef(d_storage));

  for (SmallVectorImpl<StringRef>::const_iterator i = components.begin(),
                                                  e = components.end();
                                                  i != e; ++i) {
    bool path_has_sep = !path.empty() && is_separator(path[path.size() - 1]);
    bool component_has_sep = !i->empty() && is_separator((*i)[0]);
    bool is_root_name;
    if (error_code ec = has_root_name(*i, is_root_name)) return ec;

    if (path_has_sep) {
      // Strip separators from beginning of component.
      size_t loc = i->find_first_not_of(separators);
      StringRef c = StringRef(i->begin() + loc, i->size() - loc);

      // Append it.
      path.append(c.begin(), c.end());
      continue;
    }

    if (!component_has_sep && !(path.empty() && is_root_name)) {
      // Add a separator.
      path.push_back(prefered_separator);
    }

    path.append(i->begin(), i->end());
  }

  return make_error_code(errc::success);
}

error_code make_absolute(SmallVectorImpl<char> &path) {
  StringRef p(path.data(), path.size());

  bool rootName, rootDirectory;
  if (error_code ec = has_root_name(p, rootName)) return ec;
  if (error_code ec = has_root_directory(p, rootDirectory)) return ec;

  // Already absolute.
  if (rootName && rootDirectory)
    return make_error_code(errc::success);

  // All of the following conditions will need the current directory.
  SmallString<128> current_dir;
  if (error_code ec = current_path(current_dir)) return ec;

  // Relative path. Prepend the current directory.
  if (!rootName && !rootDirectory) {
    // Append path to the current directory.
    if (error_code ec = append(current_dir, p)) return ec;
    // Set path to the result.
    path.swap(current_dir);
    return make_error_code(errc::success);
  }

  if (!rootName && rootDirectory) {
    StringRef cdrn;
    if (error_code ec = root_name(current_dir, cdrn)) return ec;
    SmallString<128> curDirRootName(cdrn.begin(), cdrn.end());
    if (error_code ec = append(curDirRootName, p)) return ec;
    // Set path to the result.
    path.swap(curDirRootName);
    return make_error_code(errc::success);
  }

  if (rootName && !rootDirectory) {
    StringRef pRootName;
    StringRef bRootDirectory;
    StringRef bRelativePath;
    StringRef pRelativePath;
    if (error_code ec = root_name(p, pRootName)) return ec;
    if (error_code ec = root_directory(current_dir, bRootDirectory)) return ec;
    if (error_code ec = relative_path(current_dir, bRelativePath)) return ec;
    if (error_code ec = relative_path(p, pRelativePath)) return ec;

    SmallString<128> res;
    if (error_code ec = append(res, pRootName, bRootDirectory,
                                    bRelativePath, pRelativePath)) return ec;
    path.swap(res);
    return make_error_code(errc::success);
  }

  llvm_unreachable("All rootName and rootDirectory combinations should have "
                   "occurred above!");
}

error_code parent_path(const StringRef &path, StringRef &result) {
  size_t end_pos = parent_path_end(path);
  if (end_pos == StringRef::npos)
    result = StringRef();
  else
    result = StringRef(path.data(), end_pos);
  return make_error_code(errc::success);
}

}
}
}

// Include the truly platform-specific parts.
#if defined(LLVM_ON_UNIX)
#include "Unix/PathV2.inc"
#endif
#if defined(LLVM_ON_WIN32)
#include "Windows/PathV2.inc"
#endif
