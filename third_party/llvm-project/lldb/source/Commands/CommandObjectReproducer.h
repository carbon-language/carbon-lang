//===-- CommandObjectReproducer.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_COMMANDS_COMMANDOBJECTREPRODUCER_H
#define LLDB_SOURCE_COMMANDS_COMMANDOBJECTREPRODUCER_H

#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {

// CommandObjectReproducer

class CommandObjectReproducer : public CommandObjectMultiword {
public:
  CommandObjectReproducer(CommandInterpreter &interpreter);

  ~CommandObjectReproducer() override;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_COMMANDS_COMMANDOBJECTREPRODUCER_H
