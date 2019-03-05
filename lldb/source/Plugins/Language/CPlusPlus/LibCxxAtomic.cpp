//===-- LibCxxAtomic.cpp ------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibCxxAtomic.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

bool lldb_private::formatters::LibCxxAtomicSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {

  ValueObjectSP non_sythetic = valobj.GetNonSyntheticValue();
  if (!non_sythetic)
    return false;

  ValueObjectSP index_zero = non_sythetic->GetChildAtIndex(0, true);
  if (!index_zero)
    return false;

  ValueObjectSP member__a_ =
      index_zero->GetChildMemberWithName(ConstString("__a_"), true);
  if (!member__a_)
    return false;

  if (ValueObjectSP member__a_value =
          member__a_->GetChildMemberWithName(ConstString("__a_value"), true)) {
    std::string summary;
    if (member__a_value->GetSummaryAsCString(summary, options) &&
        summary.size() > 0) {
      stream.Printf("%s", summary.c_str());
      return true;
    }
  }

  return false;
}

namespace lldb_private {
namespace formatters {
class LibcxxStdAtomicSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  LibcxxStdAtomicSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  ~LibcxxStdAtomicSyntheticFrontEnd() override = default;

  size_t CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override;

  bool Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(const ConstString &name) override;

  lldb::ValueObjectSP GetSyntheticValue() override;

private:
  ValueObject *m_real_child;
};
} // namespace formatters
} // namespace lldb_private

lldb_private::formatters::LibcxxStdAtomicSyntheticFrontEnd::
    LibcxxStdAtomicSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_real_child(nullptr) {}

bool lldb_private::formatters::LibcxxStdAtomicSyntheticFrontEnd::Update() {
  ValueObjectSP index_zero = m_backend.GetChildAtIndex(0, true);
  if (!index_zero)
    return false;

  ValueObjectSP member__a_ =
      index_zero->GetChildMemberWithName(ConstString("__a_"), true);
  if (!member__a_)
    return false;

  m_real_child =
      member__a_->GetChildMemberWithName(ConstString("__a_value"), true).get();

  return false;
}

bool lldb_private::formatters::LibcxxStdAtomicSyntheticFrontEnd::
    MightHaveChildren() {
  return true;
}

size_t lldb_private::formatters::LibcxxStdAtomicSyntheticFrontEnd::
    CalculateNumChildren() {
  return m_real_child ? m_real_child->GetNumChildren() : 0;
}

lldb::ValueObjectSP
lldb_private::formatters::LibcxxStdAtomicSyntheticFrontEnd::GetChildAtIndex(
    size_t idx) {
  return m_real_child ? m_real_child->GetChildAtIndex(idx, true) : nullptr;
}

size_t lldb_private::formatters::LibcxxStdAtomicSyntheticFrontEnd::
    GetIndexOfChildWithName(const ConstString &name) {
  return m_real_child ? m_real_child->GetIndexOfChildWithName(name)
                      : UINT32_MAX;
}

lldb::ValueObjectSP lldb_private::formatters::LibcxxStdAtomicSyntheticFrontEnd::
    GetSyntheticValue() {
  if (m_real_child && m_real_child->CanProvideValue())
    return m_real_child->GetSP();
  return nullptr;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::LibcxxAtomicSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (valobj_sp)
    return new LibcxxStdAtomicSyntheticFrontEnd(valobj_sp);
  return nullptr;
}
