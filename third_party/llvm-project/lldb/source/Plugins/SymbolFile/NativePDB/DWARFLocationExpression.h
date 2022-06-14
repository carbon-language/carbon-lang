//===-- DWARFLocationExpression.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_NATIVEPDB_DWARFLOCATIONEXPRESSION_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_NATIVEPDB_DWARFLOCATIONEXPRESSION_H

#include "lldb/lldb-forward.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include <map>

namespace llvm {
class APSInt;
class StringRef;
namespace codeview {
class TypeIndex;
}
namespace pdb {
class TpiStream;
}
} // namespace llvm
namespace lldb_private {
namespace npdb {
DWARFExpression
MakeEnregisteredLocationExpression(llvm::codeview::RegisterId reg,
                                   lldb::ModuleSP module);

DWARFExpression MakeRegRelLocationExpression(llvm::codeview::RegisterId reg,
                                             int32_t offset,
                                             lldb::ModuleSP module);
DWARFExpression MakeVFrameRelLocationExpression(llvm::StringRef fpo_program,
                                                int32_t offset,
                                                lldb::ModuleSP module);
DWARFExpression MakeGlobalLocationExpression(uint16_t section, uint32_t offset,
                                             lldb::ModuleSP module);
DWARFExpression MakeConstantLocationExpression(
    llvm::codeview::TypeIndex underlying_ti, llvm::pdb::TpiStream &tpi,
    const llvm::APSInt &constant, lldb::ModuleSP module);
DWARFExpression MakeEnregisteredLocationExpressionForClass(
    std::map<uint64_t, std::pair<llvm::codeview::RegisterId, uint32_t>>
        &members_info,
    lldb::ModuleSP module);
} // namespace npdb
} // namespace lldb_private

#endif
