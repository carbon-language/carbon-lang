//===-- DebuggerEvents.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DebuggerEvents.h"
#include "llvm/Support/WithColor.h"

using namespace lldb_private;

template <typename T>
static const T *GetEventDataFromEventImpl(const Event *event_ptr) {
  if (event_ptr)
    if (const EventData *event_data = event_ptr->GetData())
      if (event_data->GetFlavor() == T::GetFlavorString())
        return static_cast<const T *>(event_ptr->GetData());
  return nullptr;
}

ConstString ProgressEventData::GetFlavorString() {
  static ConstString g_flavor("ProgressEventData");
  return g_flavor;
}

ConstString ProgressEventData::GetFlavor() const {
  return ProgressEventData::GetFlavorString();
}

void ProgressEventData::Dump(Stream *s) const {
  s->Printf(" id = %" PRIu64 ", message = \"%s\"", m_id, m_message.c_str());
  if (m_completed == 0 || m_completed == m_total)
    s->Printf(", type = %s", m_completed == 0 ? "start" : "end");
  else
    s->PutCString(", type = update");
  // If m_total is UINT64_MAX, there is no progress to report, just "start"
  // and "end". If it isn't we will show the completed and total amounts.
  if (m_total != UINT64_MAX)
    s->Printf(", progress = %" PRIu64 " of %" PRIu64, m_completed, m_total);
}

const ProgressEventData *
ProgressEventData::GetEventDataFromEvent(const Event *event_ptr) {
  return GetEventDataFromEventImpl<ProgressEventData>(event_ptr);
}

llvm::StringRef DiagnosticEventData::GetPrefix() const {
  switch (m_type) {
  case Type::Warning:
    return "warning";
  case Type::Error:
    return "error";
  }
  llvm_unreachable("Fully covered switch above!");
}

void DiagnosticEventData::Dump(Stream *s) const {
  llvm::HighlightColor color = m_type == Type::Warning
                                   ? llvm::HighlightColor::Warning
                                   : llvm::HighlightColor::Error;
  llvm::WithColor(s->AsRawOstream(), color, llvm::ColorMode::Enable)
      << GetPrefix();
  *s << ": " << GetMessage() << '\n';
  s->Flush();
}

ConstString DiagnosticEventData::GetFlavorString() {
  static ConstString g_flavor("DiagnosticEventData");
  return g_flavor;
}

ConstString DiagnosticEventData::GetFlavor() const {
  return DiagnosticEventData::GetFlavorString();
}

const DiagnosticEventData *
DiagnosticEventData::GetEventDataFromEvent(const Event *event_ptr) {
  return GetEventDataFromEventImpl<DiagnosticEventData>(event_ptr);
}
