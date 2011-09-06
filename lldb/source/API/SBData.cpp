//===-- SBData.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBData.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBStream.h"

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"

using namespace lldb;
using namespace lldb_private;

SBData::SBData ()
{
}

SBData::SBData (const lldb::DataExtractorSP& data_sp) :
    m_opaque_sp (data_sp)
{
}

SBData::SBData(const SBData &rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
}

const SBData &
SBData::operator = (const SBData &rhs)
{
    if (this != &rhs)
        m_opaque_sp = rhs.m_opaque_sp;
    return *this;
}

SBData::~SBData ()
{
}

void
SBData::SetOpaque (const lldb::DataExtractorSP &data_sp)
{
    m_opaque_sp = data_sp;
}

lldb_private::DataExtractor *
SBData::get() const
{
    return m_opaque_sp.get();
}

lldb_private::DataExtractor *
SBData::operator->() const
{
    return m_opaque_sp.operator->();
}

lldb::DataExtractorSP &
SBData::operator*()
{
    return m_opaque_sp;
}

const lldb::DataExtractorSP &
SBData::operator*() const
{
    return m_opaque_sp;
}

bool
SBData::IsValid()
{
    return m_opaque_sp.get() != NULL;
}

uint8_t
SBData::GetAddressByteSize ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    uint8_t value = 0;
    if (m_opaque_sp.get())
        value = m_opaque_sp->GetAddressByteSize();
    if (log)
        log->Printf ("SBData::GetAddressByteSize () => "
                     "(%i)", value);
    return value;
}

void
SBData::Clear ()
{
    if (m_opaque_sp.get())
        m_opaque_sp->Clear();
}

size_t
SBData::GetByteSize ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    size_t value = 0;
    if (m_opaque_sp.get())
        value = m_opaque_sp->GetByteSize();
    if (log)
        log->Printf ("SBData::GetByteSize () => "
                     "(%i)", value);
    return value;
}

lldb::ByteOrder
SBData::GetByteOrder ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    lldb::ByteOrder value = eByteOrderInvalid;
    if (m_opaque_sp.get())
        value = m_opaque_sp->GetByteOrder();
    if (log)
        log->Printf ("SBData::GetByteOrder () => "
                     "(%i)", value);
    return value;
}

float
SBData::GetFloat (lldb::SBError& error, uint32_t offset)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    float value = 0;
    if (!m_opaque_sp.get())
    {
        error.SetErrorString("no value to read from");
    }
    else
    {
        uint32_t old_offset = offset;
        value = m_opaque_sp->GetFloat(&offset);
        if (offset == old_offset)
            error.SetErrorString("unable to read data");
    }
    if (log)
        log->Printf ("SBData::GetFloat (error=%p,offset=%d) => "
                     "(%f)", error.get(), offset, value);
    return value;
}

double
SBData::GetDouble (lldb::SBError& error, uint32_t offset)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    double value = 0;
    if (!m_opaque_sp.get())
    {
        error.SetErrorString("no value to read from");
    }
    else
    {
        uint32_t old_offset = offset;
        value = m_opaque_sp->GetDouble(&offset);
        if (offset == old_offset)
            error.SetErrorString("unable to read data");
    }
    if (log)
        log->Printf ("SBData::GetDouble (error=%p,offset=%d) => "
                     "(%f)", error.get(), offset, value);
    return value;
}

long double
SBData::GetLongDouble (lldb::SBError& error, uint32_t offset)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    long double value = 0;
    if (!m_opaque_sp.get())
    {
        error.SetErrorString("no value to read from");
    }
    else
    {
        uint32_t old_offset = offset;
        value = m_opaque_sp->GetLongDouble(&offset);
        if (offset == old_offset)
            error.SetErrorString("unable to read data");
    }
    if (log)
        log->Printf ("SBData::GetLongDouble (error=%p,offset=%d) => "
                     "(%lf)", error.get(), offset, value);
    return value;
}

lldb::addr_t
SBData::GetAddress (lldb::SBError& error, uint32_t offset)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    lldb::addr_t value = 0;
    if (!m_opaque_sp.get())
    {
        error.SetErrorString("no value to read from");
    }
    else
    {
        uint32_t old_offset = offset;
        value = m_opaque_sp->GetAddress(&offset);
        if (offset == old_offset)
            error.SetErrorString("unable to read data");
    }
    if (log)
        log->Printf ("SBData::GetAddress (error=%p,offset=%d) => "
                     "(%p)", error.get(), offset, (void*)value);
    return value;
}

uint8_t
SBData::GetUnsignedInt8 (lldb::SBError& error, uint32_t offset)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    uint8_t value = 0;
    if (!m_opaque_sp.get())
    {
        error.SetErrorString("no value to read from");
    }
    else
    {
        uint32_t old_offset = offset;
        value = m_opaque_sp->GetU8(&offset);
        if (offset == old_offset)
            error.SetErrorString("unable to read data");
    }
    if (log)
        log->Printf ("SBData::GetUnsignedInt8 (error=%p,offset=%d) => "
                     "(%c)", error.get(), offset, value);
    return value;
}

uint16_t
SBData::GetUnsignedInt16 (lldb::SBError& error, uint32_t offset)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    uint16_t value = 0;
    if (!m_opaque_sp.get())
    {
        error.SetErrorString("no value to read from");
    }
    else
    {
        uint32_t old_offset = offset;
        value = m_opaque_sp->GetU16(&offset);
        if (offset == old_offset)
            error.SetErrorString("unable to read data");
    }
    if (log)
        log->Printf ("SBData::GetUnsignedInt16 (error=%p,offset=%d) => "
                     "(%hd)", error.get(), offset, value);
    return value;
}

uint32_t
SBData::GetUnsignedInt32 (lldb::SBError& error, uint32_t offset)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    uint32_t value = 0;
    if (!m_opaque_sp.get())
    {
        error.SetErrorString("no value to read from");
    }
    else
    {
        uint32_t old_offset = offset;
        value = m_opaque_sp->GetU32(&offset);
        if (offset == old_offset)
            error.SetErrorString("unable to read data");
    }
    if (log)
        log->Printf ("SBData::GetUnsignedInt32 (error=%p,offset=%d) => "
                     "(%d)", error.get(), offset, value);
    return value;
}

uint64_t
SBData::GetUnsignedInt64 (lldb::SBError& error, uint32_t offset)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    uint64_t value = 0;
    if (!m_opaque_sp.get())
    {
        error.SetErrorString("no value to read from");
    }
    else
    {
        uint32_t old_offset = offset;
        value = m_opaque_sp->GetU64(&offset);
        if (offset == old_offset)
            error.SetErrorString("unable to read data");
    }
    if (log)
        log->Printf ("SBData::GetUnsignedInt64 (error=%p,offset=%d) => "
                     "(%q)", error.get(), offset, value);
    return value;
}

int8_t
SBData::GetSignedInt8 (lldb::SBError& error, uint32_t offset)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    int8_t value = 0;
    if (!m_opaque_sp.get())
    {
        error.SetErrorString("no value to read from");
    }
    else
    {
        uint32_t old_offset = offset;
        value = (int8_t)m_opaque_sp->GetMaxS64(&offset, 1);
        if (offset == old_offset)
            error.SetErrorString("unable to read data");
    }
    if (log)
        log->Printf ("SBData::GetSignedInt8 (error=%p,offset=%d) => "
                     "(%c)", error.get(), offset, value);
    return value;
}

int16_t
SBData::GetSignedInt16 (lldb::SBError& error, uint32_t offset)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    int16_t value = 0;
    if (!m_opaque_sp.get())
    {
        error.SetErrorString("no value to read from");
    }
    else
    {
        uint32_t old_offset = offset;
        value = (int16_t)m_opaque_sp->GetMaxS64(&offset, 2);
        if (offset == old_offset)
            error.SetErrorString("unable to read data");
    }
    if (log)
        log->Printf ("SBData::GetSignedInt16 (error=%p,offset=%d) => "
                     "(%hd)", error.get(), offset, value);
    return value;
}

int32_t
SBData::GetSignedInt32 (lldb::SBError& error, uint32_t offset)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    int32_t value = 0;
    if (!m_opaque_sp.get())
    {
        error.SetErrorString("no value to read from");
    }
    else
    {
        uint32_t old_offset = offset;
        value = (int32_t)m_opaque_sp->GetMaxS64(&offset, 4);
        if (offset == old_offset)
            error.SetErrorString("unable to read data");
    }
    if (log)
        log->Printf ("SBData::GetSignedInt32 (error=%p,offset=%d) => "
                     "(%d)", error.get(), offset, value);
    return value;
}

int64_t
SBData::GetSignedInt64 (lldb::SBError& error, uint32_t offset)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    int64_t value = 0;
    if (!m_opaque_sp.get())
    {
        error.SetErrorString("no value to read from");
    }
    else
    {
        uint32_t old_offset = offset;
        value = (int64_t)m_opaque_sp->GetMaxS64(&offset, 8);
        if (offset == old_offset)
            error.SetErrorString("unable to read data");
    }
    if (log)
        log->Printf ("SBData::GetSignedInt64 (error=%p,offset=%d) => "
                     "(%q)", error.get(), offset, value);
    return value;
}

const char*
SBData::GetString (lldb::SBError& error, uint32_t offset)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    const char* value = 0;
    if (!m_opaque_sp.get())
    {
        error.SetErrorString("no value to read from");
    }
    else
    {
        uint32_t old_offset = offset;
        value = m_opaque_sp->GetCStr(&offset);
        if (offset == old_offset || (value == NULL))
            error.SetErrorString("unable to read data");
    }
    if (log)
        log->Printf ("SBData::GetString (error=%p,offset=%d) => "
                     "(%p)", error.get(), offset, value);
    return value;
}

bool
SBData::GetDescription (lldb::SBStream &description)
{
    if (m_opaque_sp)
    {
        description.ref();
        m_opaque_sp->Dump(description.get(),
                          0,
                          lldb::eFormatBytesWithASCII,
                          1,
                          m_opaque_sp->GetByteSize(),
                          16,
                          LLDB_INVALID_ADDRESS,
                          0,
                          0);
    }
    else
        description.Printf ("No Value");
    
    return true;
}

size_t
SBData::ReadRawData (lldb::SBError& error,
                     uint32_t offset,
                     void *buf,
                     size_t size)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    void* ok = NULL;
    if (!m_opaque_sp.get())
    {
        error.SetErrorString("no value to read from");
    }
    else
    {
        uint32_t old_offset = offset;
        ok = m_opaque_sp->GetU8(&offset, buf, size);
        if ((offset == old_offset) || (ok == NULL))
            error.SetErrorString("unable to read data");
    }
    if (log)
        log->Printf ("SBData::ReadRawData (error=%p,offset=%d,buf=%p,size=%d) => "
                     "(%p)", error.get(), offset, buf, size, ok);
    return ok ? size : 0;
}

void
SBData::SetData(lldb::SBError& error,
                const void *buf,
                size_t size,
                lldb::ByteOrder endian,
                uint8_t addr_size)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (!m_opaque_sp.get())
        m_opaque_sp.reset(new DataExtractor(buf, size, endian, addr_size));
    else
        m_opaque_sp->SetData(buf, size, endian);
    if (log)
        log->Printf ("SBData::SetData (error=%p,buf=%p,size=%d,endian=%d,addr_size=%c) => "
                     "(%p)", error.get(), buf, size, endian, addr_size, m_opaque_sp.get());
}

bool
SBData::Append(const SBData& rhs)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    bool value = false;
    if (m_opaque_sp.get() && rhs.m_opaque_sp.get())
        value = m_opaque_sp.get()->Append(*rhs.m_opaque_sp);
    if (log)
        log->Printf ("SBData::Append (rhs=%p) => "
                     "(%s)", rhs.get(), value ? "true" : "false");
    return value;
}