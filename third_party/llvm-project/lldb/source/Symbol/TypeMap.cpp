//===-- TypeMap.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeMap.h"

using namespace lldb;
using namespace lldb_private;

TypeMap::TypeMap() : m_types() {}

// Destructor
TypeMap::~TypeMap() = default;

void TypeMap::Insert(const TypeSP &type_sp) {
  // Just push each type on the back for now. We will worry about uniquing
  // later
  if (type_sp)
    m_types.insert(std::make_pair(type_sp->GetID(), type_sp));
}

bool TypeMap::InsertUnique(const TypeSP &type_sp) {
  if (type_sp) {
    user_id_t type_uid = type_sp->GetID();
    iterator pos, end = m_types.end();

    for (pos = m_types.find(type_uid);
         pos != end && pos->second->GetID() == type_uid; ++pos) {
      if (pos->second.get() == type_sp.get())
        return false;
    }
    Insert(type_sp);
  }
  return true;
}

// Find a base type by its unique ID.
// TypeSP
// TypeMap::FindType(lldb::user_id_t uid)
//{
//    iterator pos = m_types.find(uid);
//    if (pos != m_types.end())
//        return pos->second;
//    return TypeSP();
//}

// Find a type by name.
// TypeMap
// TypeMap::FindTypes (ConstString name)
//{
//    // Do we ever need to make a lookup by name map? Here we are doing
//    // a linear search which isn't going to be fast.
//    TypeMap types(m_ast.getTargetInfo()->getTriple().getTriple().c_str());
//    iterator pos, end;
//    for (pos = m_types.begin(), end = m_types.end(); pos != end; ++pos)
//        if (pos->second->GetName() == name)
//            types.Insert (pos->second);
//    return types;
//}

void TypeMap::Clear() { m_types.clear(); }

uint32_t TypeMap::GetSize() const { return m_types.size(); }

bool TypeMap::Empty() const { return m_types.empty(); }

// GetTypeAtIndex isn't used a lot for large type lists, currently only for
// type lists that are returned for "image dump -t TYPENAME" commands and other
// simple symbol queries that grab the first result...

TypeSP TypeMap::GetTypeAtIndex(uint32_t idx) {
  iterator pos, end;
  uint32_t i = idx;
  for (pos = m_types.begin(), end = m_types.end(); pos != end; ++pos) {
    if (i == 0)
      return pos->second;
    --i;
  }
  return TypeSP();
}

void TypeMap::ForEach(
    std::function<bool(const lldb::TypeSP &type_sp)> const &callback) const {
  for (auto pos = m_types.begin(), end = m_types.end(); pos != end; ++pos) {
    if (!callback(pos->second))
      break;
  }
}

void TypeMap::ForEach(
    std::function<bool(lldb::TypeSP &type_sp)> const &callback) {
  for (auto pos = m_types.begin(), end = m_types.end(); pos != end; ++pos) {
    if (!callback(pos->second))
      break;
  }
}

bool TypeMap::Remove(const lldb::TypeSP &type_sp) {
  if (type_sp) {
    lldb::user_id_t uid = type_sp->GetID();
    for (iterator pos = m_types.find(uid), end = m_types.end();
         pos != end && pos->first == uid; ++pos) {
      if (pos->second == type_sp) {
        m_types.erase(pos);
        return true;
      }
    }
  }
  return false;
}

void TypeMap::Dump(Stream *s, bool show_context, lldb::DescriptionLevel level) {
  for (iterator pos = m_types.begin(), end = m_types.end(); pos != end; ++pos) {
    pos->second->Dump(s, show_context, level);
  }
}

void TypeMap::RemoveMismatchedTypes(const char *qualified_typename,
                                    bool exact_match) {
  llvm::StringRef type_scope;
  llvm::StringRef type_basename;
  TypeClass type_class = eTypeClassAny;
  if (!Type::GetTypeScopeAndBasename(qualified_typename, type_scope,
                                     type_basename, type_class)) {
    type_basename = qualified_typename;
    type_scope = "";
  }
  return RemoveMismatchedTypes(std::string(type_scope),
                               std::string(type_basename), type_class,
                               exact_match);
}

void TypeMap::RemoveMismatchedTypes(const std::string &type_scope,
                                    const std::string &type_basename,
                                    TypeClass type_class, bool exact_match) {
  // Our "collection" type currently is a std::map which doesn't have any good
  // way to iterate and remove items from the map so we currently just make a
  // new list and add all of the matching types to it, and then swap it into
  // m_types at the end
  collection matching_types;

  iterator pos, end = m_types.end();

  for (pos = m_types.begin(); pos != end; ++pos) {
    Type *the_type = pos->second.get();
    bool keep_match = false;
    TypeClass match_type_class = eTypeClassAny;

    if (type_class != eTypeClassAny) {
      match_type_class = the_type->GetForwardCompilerType().GetTypeClass();
      if ((match_type_class & type_class) == 0)
        continue;
    }

    ConstString match_type_name_const_str(the_type->GetQualifiedName());
    if (match_type_name_const_str) {
      const char *match_type_name = match_type_name_const_str.GetCString();
      llvm::StringRef match_type_scope;
      llvm::StringRef match_type_basename;
      if (Type::GetTypeScopeAndBasename(match_type_name, match_type_scope,
                                        match_type_basename,
                                        match_type_class)) {
        if (match_type_basename == type_basename) {
          const size_t type_scope_size = type_scope.size();
          const size_t match_type_scope_size = match_type_scope.size();
          if (exact_match || (type_scope_size == match_type_scope_size)) {
            keep_match = match_type_scope == type_scope;
          } else {
            if (match_type_scope_size > type_scope_size) {
              const size_t type_scope_pos = match_type_scope.rfind(type_scope);
              if (type_scope_pos == match_type_scope_size - type_scope_size) {
                if (type_scope_pos >= 2) {
                  // Our match scope ends with the type scope we were looking
                  // for, but we need to make sure what comes before the
                  // matching type scope is a namespace boundary in case we are
                  // trying to match: type_basename = "d" type_scope = "b::c::"
                  // We want to match:
                  //  match_type_scope "a::b::c::"
                  // But not:
                  //  match_type_scope "a::bb::c::"
                  // So below we make sure what comes before "b::c::" in
                  // match_type_scope is "::", or the namespace boundary
                  if (match_type_scope[type_scope_pos - 1] == ':' &&
                      match_type_scope[type_scope_pos - 2] == ':') {
                    keep_match = true;
                  }
                }
              }
            }
          }
        }
      } else {
        // The type we are currently looking at doesn't exists in a namespace
        // or class, so it only matches if there is no type scope...
        keep_match = type_scope.empty() && type_basename == match_type_name;
      }
    }

    if (keep_match) {
      matching_types.insert(*pos);
    }
  }
  m_types.swap(matching_types);
}

void TypeMap::RemoveMismatchedTypes(TypeClass type_class) {
  if (type_class == eTypeClassAny)
    return;

  // Our "collection" type currently is a std::map which doesn't have any good
  // way to iterate and remove items from the map so we currently just make a
  // new list and add all of the matching types to it, and then swap it into
  // m_types at the end
  collection matching_types;

  iterator pos, end = m_types.end();

  for (pos = m_types.begin(); pos != end; ++pos) {
    Type *the_type = pos->second.get();
    TypeClass match_type_class =
        the_type->GetForwardCompilerType().GetTypeClass();
    if (match_type_class & type_class)
      matching_types.insert(*pos);
  }
  m_types.swap(matching_types);
}
