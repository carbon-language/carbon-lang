//===-- StreamTee.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StreamTee_h_
#define liblldb_StreamTee_h_

#include "lldb/Core/Stream.h"

namespace lldb_private {

class StreamTee : public Stream
{
public:
    StreamTee () :
        Stream()
    {
    }

    StreamTee (lldb::StreamSP &stream_1_sp, lldb::StreamSP &stream_2_sp):
        m_stream_1_sp (stream_1_sp),
        m_stream_2_sp (stream_2_sp)
    {
    }

    StreamTee (lldb::StreamSP &stream_1_sp):
        m_stream_1_sp (stream_1_sp),
        m_stream_2_sp ()
    {
    }

    virtual
    ~StreamTee ()
    {
    }

    virtual void
    Flush ()
    {
        if (m_stream_1_sp)
            m_stream_1_sp->Flush ();
            
        if (m_stream_2_sp)
            m_stream_2_sp->Flush ();
    }

    virtual int
    Write (const void *s, size_t length)
    {
        int ret_1;
        int ret_2;
        if (m_stream_1_sp)
            ret_1 = m_stream_1_sp->Write (s, length);
            
        if (m_stream_2_sp)
            ret_2 = m_stream_2_sp->Write (s, length);
        
        return ret_1 < ret_2 ? ret_1 : ret_2;
    }

    void
    SetStream1 (lldb::StreamSP &stream_1_sp)
    {
        m_stream_1_sp = stream_1_sp;
    }
    
    void
    SetStream2 (lldb::StreamSP &stream_2_sp)
    {
        m_stream_2_sp = stream_2_sp;
    }
    
    lldb::StreamSP &
    GetStream1 ()
    {
        return m_stream_1_sp;
    }
    
    lldb::StreamSP &
    GetStream2 ()
    {
        return m_stream_2_sp;
    }
    
protected:
    lldb::StreamSP m_stream_1_sp;
    lldb::StreamSP m_stream_2_sp;

};

} // namespace lldb_private
#endif  // #ifndef liblldb_StreamTee_h_
