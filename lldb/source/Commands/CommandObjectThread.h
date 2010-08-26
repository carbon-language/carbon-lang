//===-- CommandObjectThread.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectThread_h_
#define liblldb_CommandObjectThread_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {

class CommandObjectMultiwordThread : public CommandObjectMultiword
{
public:

    CommandObjectMultiwordThread (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectMultiwordThread ();

};


bool
DisplayThreadInfo (CommandInterpreter &interpreter,
                   Stream &strm,
                   Thread *thread,
                   bool only_threads_with_stop_reason,
                   bool show_source);

size_t
DisplayThreadsInfo (CommandInterpreter &interpreter,
                    ExecutionContext *exe_ctx,
                    CommandReturnObject &result,
                    bool only_threads_with_stop_reason,
                    bool show_source);

size_t
DisplayFramesForExecutionContext (Thread *thread,
                                  CommandInterpreter &interpreter,
                                  Stream& strm,
                                  uint32_t first_frame,
                                  uint32_t num_frames,
                                  bool show_frame_info,
                                  uint32_t num_frames_with_source,
                                  uint32_t source_lines_before,
                                  uint32_t source_lines_after);

bool
DisplayFrameForExecutionContext (Thread *thread,
                                 StackFrame *frame,
                                 CommandInterpreter &interpreter,
                                 Stream& strm,
                                 bool show_frame_info,
                                 bool show_source,
                                 uint32_t source_lines_before,
                                 uint32_t source_lines_after);

} // namespace lldb_private

#endif  // liblldb_CommandObjectThread_h_
