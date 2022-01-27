//===-- SBValueList.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBVALUELIST_H
#define LLDB_API_SBVALUELIST_H

#include "lldb/API/SBDefines.h"

class ValueListImpl;

namespace lldb {

class LLDB_API SBValueList {
public:
  SBValueList();

  SBValueList(const lldb::SBValueList &rhs);

  ~SBValueList();

  explicit operator bool() const;

  bool IsValid() const;

  void Clear();

  void Append(const lldb::SBValue &val_obj);

  void Append(const lldb::SBValueList &value_list);

  uint32_t GetSize() const;

  lldb::SBValue GetValueAtIndex(uint32_t idx) const;

  lldb::SBValue GetFirstValueByName(const char *name) const;

  lldb::SBValue FindValueObjectByUID(lldb::user_id_t uid);

  const lldb::SBValueList &operator=(const lldb::SBValueList &rhs);

protected:
  // only useful for visualizing the pointer or comparing two SBValueLists to
  // see if they are backed by the same underlying Impl.
  void *opaque_ptr();

private:
  friend class SBFrame;

  SBValueList(const ValueListImpl *lldb_object_ptr);

  void Append(lldb::ValueObjectSP &val_obj_sp);

  void CreateIfNeeded();

  ValueListImpl *operator->();

  ValueListImpl &operator*();

  const ValueListImpl *operator->() const;

  const ValueListImpl &operator*() const;

  ValueListImpl &ref();

  std::unique_ptr<ValueListImpl> m_opaque_up;
};

} // namespace lldb

#endif // LLDB_API_SBVALUELIST_H
