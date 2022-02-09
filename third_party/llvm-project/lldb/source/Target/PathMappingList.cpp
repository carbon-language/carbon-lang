//===-- PathMappingList.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <climits>
#include <cstring>

#include "lldb/Host/FileSystem.h"
#include "lldb/Host/PosixApi.h"
#include "lldb/Target/PathMappingList.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "lldb/lldb-private-enumerations.h"

using namespace lldb;
using namespace lldb_private;

namespace {
  // We must normalize our path pairs that we store because if we don't then
  // things won't always work. We found a case where if we did:
  // (lldb) settings set target.source-map . /tmp
  // We would store a path pairs of "." and "/tmp" as raw strings. If the debug
  // info contains "./foo/bar.c", the path will get normalized to "foo/bar.c".
  // When PathMappingList::RemapPath() is called, it expects the path to start
  // with the raw path pair, which doesn't work anymore because the paths have
  // been normalized when the debug info was loaded. So we need to store
  // nomalized path pairs to ensure things match up.
std::string NormalizePath(llvm::StringRef path) {
  // If we use "path" to construct a FileSpec, it will normalize the path for
  // us. We then grab the string.
  return FileSpec(path).GetPath();
}
}
// PathMappingList constructor
PathMappingList::PathMappingList() : m_pairs() {}

PathMappingList::PathMappingList(ChangedCallback callback, void *callback_baton)
    : m_pairs(), m_callback(callback), m_callback_baton(callback_baton),
      m_mod_id(0) {}

PathMappingList::PathMappingList(const PathMappingList &rhs)
    : m_pairs(rhs.m_pairs), m_callback(nullptr), m_callback_baton(nullptr),
      m_mod_id(0) {}

const PathMappingList &PathMappingList::operator=(const PathMappingList &rhs) {
  if (this != &rhs) {
    m_pairs = rhs.m_pairs;
    m_callback = nullptr;
    m_callback_baton = nullptr;
    m_mod_id = rhs.m_mod_id;
  }
  return *this;
}

PathMappingList::~PathMappingList() = default;

void PathMappingList::Append(llvm::StringRef path, llvm::StringRef replacement,
                             bool notify) {
  ++m_mod_id;
  m_pairs.emplace_back(pair(NormalizePath(path), NormalizePath(replacement)));
  if (notify && m_callback)
    m_callback(*this, m_callback_baton);
}

void PathMappingList::Append(const PathMappingList &rhs, bool notify) {
  ++m_mod_id;
  if (!rhs.m_pairs.empty()) {
    const_iterator pos, end = rhs.m_pairs.end();
    for (pos = rhs.m_pairs.begin(); pos != end; ++pos)
      m_pairs.push_back(*pos);
    if (notify && m_callback)
      m_callback(*this, m_callback_baton);
  }
}

void PathMappingList::Insert(llvm::StringRef path, llvm::StringRef replacement,
                             uint32_t index, bool notify) {
  ++m_mod_id;
  iterator insert_iter;
  if (index >= m_pairs.size())
    insert_iter = m_pairs.end();
  else
    insert_iter = m_pairs.begin() + index;
  m_pairs.emplace(insert_iter, pair(NormalizePath(path),
                                    NormalizePath(replacement)));
  if (notify && m_callback)
    m_callback(*this, m_callback_baton);
}

bool PathMappingList::Replace(llvm::StringRef path, llvm::StringRef replacement,
                              uint32_t index, bool notify) {
  if (index >= m_pairs.size())
    return false;
  ++m_mod_id;
  m_pairs[index] = pair(NormalizePath(path), NormalizePath(replacement));
  if (notify && m_callback)
    m_callback(*this, m_callback_baton);
  return true;
}

bool PathMappingList::Remove(size_t index, bool notify) {
  if (index >= m_pairs.size())
    return false;

  ++m_mod_id;
  iterator iter = m_pairs.begin() + index;
  m_pairs.erase(iter);
  if (notify && m_callback)
    m_callback(*this, m_callback_baton);
  return true;
}

// For clients which do not need the pair index dumped, pass a pair_index >= 0
// to only dump the indicated pair.
void PathMappingList::Dump(Stream *s, int pair_index) {
  unsigned int numPairs = m_pairs.size();

  if (pair_index < 0) {
    unsigned int index;
    for (index = 0; index < numPairs; ++index)
      s->Printf("[%d] \"%s\" -> \"%s\"\n", index,
                m_pairs[index].first.GetCString(),
                m_pairs[index].second.GetCString());
  } else {
    if (static_cast<unsigned int>(pair_index) < numPairs)
      s->Printf("%s -> %s", m_pairs[pair_index].first.GetCString(),
                m_pairs[pair_index].second.GetCString());
  }
}

void PathMappingList::Clear(bool notify) {
  if (!m_pairs.empty())
    ++m_mod_id;
  m_pairs.clear();
  if (notify && m_callback)
    m_callback(*this, m_callback_baton);
}

bool PathMappingList::RemapPath(ConstString path,
                                ConstString &new_path) const {
  if (llvm::Optional<FileSpec> remapped = RemapPath(path.GetStringRef())) {
    new_path.SetString(remapped->GetPath());
    return true;
  }
  return false;
}

/// Append components to path, applying style.
static void AppendPathComponents(FileSpec &path, llvm::StringRef components,
                                 llvm::sys::path::Style style) {
    auto component = llvm::sys::path::begin(components, style);
    auto e = llvm::sys::path::end(components);
    while (component != e &&
        llvm::sys::path::is_separator(*component->data(), style))
      ++component;
    for (; component != e; ++component)
      path.AppendPathComponent(*component);
}

llvm::Optional<FileSpec>
PathMappingList::RemapPath(llvm::StringRef mapping_path,
                           bool only_if_exists) const {
  if (m_pairs.empty() || mapping_path.empty())
    return {};
  LazyBool path_is_relative = eLazyBoolCalculate;

  for (const auto &it : m_pairs) {
    llvm::StringRef prefix = it.first.GetStringRef();
    // We create a copy of mapping_path because StringRef::consume_from
    // effectively modifies the instance itself.
    llvm::StringRef path = mapping_path;
    if (!path.consume_front(prefix)) {
      // Relative paths won't have a leading "./" in them unless "." is the
      // only thing in the relative path so we need to work around "."
      // carefully.
      if (prefix != ".")
        continue;
      // We need to figure out if the "path" argument is relative. If it is,
      // then we should remap, else skip this entry.
      if (path_is_relative == eLazyBoolCalculate) {
        path_is_relative =
            FileSpec(path).IsRelative() ? eLazyBoolYes : eLazyBoolNo;
      }
      if (!path_is_relative)
        continue;
    }
    FileSpec remapped(it.second.GetStringRef());
    auto orig_style = FileSpec::GuessPathStyle(prefix).getValueOr(
        llvm::sys::path::Style::native);
    AppendPathComponents(remapped, path, orig_style);
    if (!only_if_exists || FileSystem::Instance().Exists(remapped))
      return remapped;
  }
  return {};
}

bool PathMappingList::ReverseRemapPath(const FileSpec &file, FileSpec &fixed) const {
  std::string path = file.GetPath();
  llvm::StringRef path_ref(path);
  for (const auto &it : m_pairs) {
    if (!path_ref.consume_front(it.second.GetStringRef()))
      continue;
    auto orig_file = it.first.GetStringRef();
    auto orig_style = FileSpec::GuessPathStyle(orig_file).getValueOr(
        llvm::sys::path::Style::native);
    fixed.SetFile(orig_file, orig_style);
    AppendPathComponents(fixed, path_ref, orig_style);
    return true;
  }
  return false;
}

llvm::Optional<FileSpec> PathMappingList::FindFile(const FileSpec &orig_spec) const {
  // We must normalize the orig_spec again using the host's path style,
  // otherwise there will be mismatch between the host and remote platform
  // if they use different path styles.
  if (auto remapped = RemapPath(NormalizePath(orig_spec.GetPath()),
                                /*only_if_exists=*/true))
    return remapped;

  return {};
}

bool PathMappingList::Replace(llvm::StringRef path, llvm::StringRef new_path,
                              bool notify) {
  uint32_t idx = FindIndexForPath(path);
  if (idx < m_pairs.size()) {
    ++m_mod_id;
    m_pairs[idx].second = ConstString(new_path);
    if (notify && m_callback)
      m_callback(*this, m_callback_baton);
    return true;
  }
  return false;
}

bool PathMappingList::Remove(ConstString path, bool notify) {
  iterator pos = FindIteratorForPath(path);
  if (pos != m_pairs.end()) {
    ++m_mod_id;
    m_pairs.erase(pos);
    if (notify && m_callback)
      m_callback(*this, m_callback_baton);
    return true;
  }
  return false;
}

PathMappingList::const_iterator
PathMappingList::FindIteratorForPath(ConstString path) const {
  const_iterator pos;
  const_iterator begin = m_pairs.begin();
  const_iterator end = m_pairs.end();

  for (pos = begin; pos != end; ++pos) {
    if (pos->first == path)
      break;
  }
  return pos;
}

PathMappingList::iterator
PathMappingList::FindIteratorForPath(ConstString path) {
  iterator pos;
  iterator begin = m_pairs.begin();
  iterator end = m_pairs.end();

  for (pos = begin; pos != end; ++pos) {
    if (pos->first == path)
      break;
  }
  return pos;
}

bool PathMappingList::GetPathsAtIndex(uint32_t idx, ConstString &path,
                                      ConstString &new_path) const {
  if (idx < m_pairs.size()) {
    path = m_pairs[idx].first;
    new_path = m_pairs[idx].second;
    return true;
  }
  return false;
}

uint32_t PathMappingList::FindIndexForPath(llvm::StringRef orig_path) const {
  const ConstString path = ConstString(NormalizePath(orig_path));
  const_iterator pos;
  const_iterator begin = m_pairs.begin();
  const_iterator end = m_pairs.end();

  for (pos = begin; pos != end; ++pos) {
    if (pos->first == path)
      return std::distance(begin, pos);
  }
  return UINT32_MAX;
}
