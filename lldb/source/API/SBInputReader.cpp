//===-- SBInputReader.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lldb/lldb-enumerations.h"

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBInputReader.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Core/Log.h"


using namespace lldb;
using namespace lldb_private;

SBInputReader::SBInputReader ()  :
    m_opaque_sp (),
    m_callback_function (NULL),
    m_callback_baton (NULL)

{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBInputReader::SBInputReader () ==> this = %p", this);
}

SBInputReader::SBInputReader (const lldb::InputReaderSP &reader_sp) :
    m_opaque_sp (reader_sp)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBInputReader::SBInputReader (const lldb::InputReaderSP &reader_sp) reader_sp.get = %p"
                     " ==> this = %p", this);
}

SBInputReader::SBInputReader (const SBInputReader &rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf("SBInputReader::SBInputReader (const SBInputReader &rhs) rhs.m_opaque_sp.get() = %p ==> this = %p",
                    rhs.m_opaque_sp.get(), this);
}

SBInputReader::~SBInputReader ()
{
}

size_t
SBInputReader::PrivateCallback 
(
    void *baton, 
    InputReader &reader, 
    lldb::InputReaderAction notification,
    const char *bytes, 
    size_t bytes_len
)
{
    SBInputReader *sb_reader = (SBInputReader *)baton;
    return sb_reader->m_callback_function (sb_reader->m_callback_baton, 
                                           sb_reader, 
                                           notification,
                                           bytes,
                                           bytes_len);
}

SBError
SBInputReader::Initialize 
(
    SBDebugger &debugger,
    Callback callback_function,
    void *callback_baton,
    lldb::InputReaderGranularity granularity,
    const char *end_token,
    const char *prompt,
    bool echo
)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
    {
        log->Printf("SBInputReader::Initialize (SBDebugger &debugger, Callback callback_function, void *baton, "
                    "lldb::InputReaderGranularity granularity, const char *end_token, const char *prompt, bool echo)");
        log->Printf("    debugger (this = %p), callback_function, callback_baton = %p, granularity = %s, "
                    "end_token = '%s', prompt = '%s', echo = %s", &debugger, callback_baton, 
                    InputReader::GranularityAsCString (granularity), end_token, prompt, (echo ? "true" : "false"));
    }

    SBError sb_error;
    m_opaque_sp.reset (new InputReader (debugger.ref()));
    
    m_callback_function = callback_function;
    m_callback_baton = callback_baton;

    if (m_opaque_sp)
    {
        sb_error.SetError (m_opaque_sp->Initialize (SBInputReader::PrivateCallback,
                                                    this,
                                                    granularity,
                                                    end_token,
                                                    prompt,
                                                    echo));
    }

    if (sb_error.Fail())
    {
        m_opaque_sp.reset ();
        m_callback_function = NULL;
        m_callback_baton = NULL;
    }

    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBInputReader::Initialize ==> SBError (this = %p, '%s')", &sb_error, sstr.GetData());
    }

    return sb_error;
}

bool
SBInputReader::IsValid () const
{
    return (m_opaque_sp.get() != NULL);
}

const SBInputReader &
SBInputReader::operator = (const SBInputReader &rhs)
{
    if (this != &rhs)
        m_opaque_sp = rhs.m_opaque_sp;
    return *this;
}

InputReader *
SBInputReader::operator->() const
{
    return m_opaque_sp.get();
}

lldb::InputReaderSP &
SBInputReader::operator *()
{
    return m_opaque_sp;
}

const lldb::InputReaderSP &
SBInputReader::operator *() const
{
    return m_opaque_sp;
}

InputReader *
SBInputReader::get() const
{
    return m_opaque_sp.get();
}

InputReader &
SBInputReader::ref() const
{
    assert (m_opaque_sp.get());
    return *m_opaque_sp;
}

bool
SBInputReader::IsDone () const
{
    if (m_opaque_sp)
        return m_opaque_sp->IsDone();
    else
        return true;
}

void
SBInputReader::SetIsDone (bool value)
{
    if (m_opaque_sp)
        m_opaque_sp->SetIsDone (value);
}

bool
SBInputReader::IsActive () const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBInputReader::IsActive ()");

    bool ret_value = false;
    if (m_opaque_sp)
        ret_value = m_opaque_sp->IsActive();
    
    if (log)
        log->Printf ("SBInputReader::IsActive ==> %s", (ret_value ? "true" : "false"));

    return ret_value;
}

InputReaderGranularity
SBInputReader::GetGranularity ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetGranularity();
    else
        return eInputReaderGranularityInvalid;
}
