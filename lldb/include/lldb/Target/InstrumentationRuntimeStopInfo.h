//===-- InstrumentationRuntimeStopInfo.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_InstrumentationRuntimeStopInfo_h_
#define liblldb_InstrumentationRuntimeStopInfo_h_

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/Target/StopInfo.h"
#include "lldb/Core/StructuredData.h"

namespace lldb_private {

class InstrumentationRuntimeStopInfo : public StopInfo
{
public:
    
    virtual ~InstrumentationRuntimeStopInfo()
    {
    }
    
    virtual lldb::StopReason
    GetStopReason () const
    {
        return lldb::eStopReasonInstrumentation;
    }
    
    virtual const char *
    GetDescription ();
    
    static lldb::StopInfoSP
    CreateStopReasonWithInstrumentationData (Thread &thread, std::string description, StructuredData::ObjectSP additional_data);
    
private:
    
    InstrumentationRuntimeStopInfo(Thread &thread, std::string description, StructuredData::ObjectSP additional_data);
    
};

} // namespace lldb_private

#endif  // liblldb_InstrumentationRuntimeStopInfo_h_
