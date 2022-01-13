//===-- SBSymbolContext.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBSymbolContext.h"
#include "lldb/Utility/ReproducerInstrumentation.h"
#include "Utils.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"

using namespace lldb;
using namespace lldb_private;

SBSymbolContext::SBSymbolContext() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBSymbolContext);
}

SBSymbolContext::SBSymbolContext(const SymbolContext &sc)
    : m_opaque_up(std::make_unique<SymbolContext>(sc)) {
  LLDB_RECORD_CONSTRUCTOR(SBSymbolContext,
                          (const lldb_private::SymbolContext &), sc);
}

SBSymbolContext::SBSymbolContext(const SBSymbolContext &rhs) {
  LLDB_RECORD_CONSTRUCTOR(SBSymbolContext, (const lldb::SBSymbolContext &),
                          rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

SBSymbolContext::~SBSymbolContext() = default;

const SBSymbolContext &SBSymbolContext::operator=(const SBSymbolContext &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBSymbolContext &,
                     SBSymbolContext, operator=,(const lldb::SBSymbolContext &),
                     rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

bool SBSymbolContext::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBSymbolContext, IsValid);
  return this->operator bool();
}
SBSymbolContext::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBSymbolContext, operator bool);

  return m_opaque_up != nullptr;
}

SBModule SBSymbolContext::GetModule() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBModule, SBSymbolContext, GetModule);

  SBModule sb_module;
  ModuleSP module_sp;
  if (m_opaque_up) {
    module_sp = m_opaque_up->module_sp;
    sb_module.SetSP(module_sp);
  }

  return sb_module;
}

SBCompileUnit SBSymbolContext::GetCompileUnit() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBCompileUnit, SBSymbolContext,
                             GetCompileUnit);

  return SBCompileUnit(m_opaque_up ? m_opaque_up->comp_unit : nullptr);
}

SBFunction SBSymbolContext::GetFunction() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBFunction, SBSymbolContext, GetFunction);

  Function *function = nullptr;

  if (m_opaque_up)
    function = m_opaque_up->function;

  SBFunction sb_function(function);

  return sb_function;
}

SBBlock SBSymbolContext::GetBlock() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBBlock, SBSymbolContext, GetBlock);

  return SBBlock(m_opaque_up ? m_opaque_up->block : nullptr);
}

SBLineEntry SBSymbolContext::GetLineEntry() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBLineEntry, SBSymbolContext, GetLineEntry);

  SBLineEntry sb_line_entry;
  if (m_opaque_up)
    sb_line_entry.SetLineEntry(m_opaque_up->line_entry);

  return sb_line_entry;
}

SBSymbol SBSymbolContext::GetSymbol() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBSymbol, SBSymbolContext, GetSymbol);

  Symbol *symbol = nullptr;

  if (m_opaque_up)
    symbol = m_opaque_up->symbol;

  SBSymbol sb_symbol(symbol);

  return sb_symbol;
}

void SBSymbolContext::SetModule(lldb::SBModule module) {
  LLDB_RECORD_METHOD(void, SBSymbolContext, SetModule, (lldb::SBModule),
                     module);

  ref().module_sp = module.GetSP();
}

void SBSymbolContext::SetCompileUnit(lldb::SBCompileUnit compile_unit) {
  LLDB_RECORD_METHOD(void, SBSymbolContext, SetCompileUnit,
                     (lldb::SBCompileUnit), compile_unit);

  ref().comp_unit = compile_unit.get();
}

void SBSymbolContext::SetFunction(lldb::SBFunction function) {
  LLDB_RECORD_METHOD(void, SBSymbolContext, SetFunction, (lldb::SBFunction),
                     function);

  ref().function = function.get();
}

void SBSymbolContext::SetBlock(lldb::SBBlock block) {
  LLDB_RECORD_METHOD(void, SBSymbolContext, SetBlock, (lldb::SBBlock), block);

  ref().block = block.GetPtr();
}

void SBSymbolContext::SetLineEntry(lldb::SBLineEntry line_entry) {
  LLDB_RECORD_METHOD(void, SBSymbolContext, SetLineEntry, (lldb::SBLineEntry),
                     line_entry);

  if (line_entry.IsValid())
    ref().line_entry = line_entry.ref();
  else
    ref().line_entry.Clear();
}

void SBSymbolContext::SetSymbol(lldb::SBSymbol symbol) {
  LLDB_RECORD_METHOD(void, SBSymbolContext, SetSymbol, (lldb::SBSymbol),
                     symbol);

  ref().symbol = symbol.get();
}

lldb_private::SymbolContext *SBSymbolContext::operator->() const {
  return m_opaque_up.get();
}

const lldb_private::SymbolContext &SBSymbolContext::operator*() const {
  assert(m_opaque_up.get());
  return *m_opaque_up;
}

lldb_private::SymbolContext &SBSymbolContext::operator*() {
  if (m_opaque_up == nullptr)
    m_opaque_up = std::make_unique<SymbolContext>();
  return *m_opaque_up;
}

lldb_private::SymbolContext &SBSymbolContext::ref() {
  if (m_opaque_up == nullptr)
    m_opaque_up = std::make_unique<SymbolContext>();
  return *m_opaque_up;
}

lldb_private::SymbolContext *SBSymbolContext::get() const {
  return m_opaque_up.get();
}

bool SBSymbolContext::GetDescription(SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBSymbolContext, GetDescription, (lldb::SBStream &),
                     description);

  Stream &strm = description.ref();

  if (m_opaque_up) {
    m_opaque_up->GetDescription(&strm, lldb::eDescriptionLevelFull, nullptr);
  } else
    strm.PutCString("No value");

  return true;
}

SBSymbolContext
SBSymbolContext::GetParentOfInlinedScope(const SBAddress &curr_frame_pc,
                                         SBAddress &parent_frame_addr) const {
  LLDB_RECORD_METHOD_CONST(lldb::SBSymbolContext, SBSymbolContext,
                           GetParentOfInlinedScope,
                           (const lldb::SBAddress &, lldb::SBAddress &),
                           curr_frame_pc, parent_frame_addr);

  SBSymbolContext sb_sc;
  if (m_opaque_up.get() && curr_frame_pc.IsValid()) {
    if (m_opaque_up->GetParentOfInlinedScope(curr_frame_pc.ref(), sb_sc.ref(),
                                             parent_frame_addr.ref()))
      return sb_sc;
  }
  return SBSymbolContext();
}
