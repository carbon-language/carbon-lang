//===-- CommandOptionValidators.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandOptionValidators_h_
#define liblldb_CommandOptionValidators_h_

#include "lldb/lldb-private-types.h"

namespace lldb_private {

class Platform;
class ExecutionContext;

class PosixPlatformCommandOptionValidator : public OptionValidator
{
    virtual bool IsValid(Platform &platform, const ExecutionContext &target) const;
    virtual const char* ShortConditionString() const;
    virtual const char* LongConditionString() const;
};

} // namespace lldb_private


#endif  // liblldb_CommandOptionValidators_h_
