//===-- SBData.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBData_h_
#define LLDB_SBData_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class SBData
{
public:

    SBData ();

    SBData (const SBData &rhs);
    
#ifndef SWIG
    const SBData &
    operator = (const SBData &rhs);
#endif

    ~SBData ();
    
    uint8_t
    GetAddressByteSize ();
    
    void
    Clear ();
    
    bool
    IsValid();
    
    size_t
    GetByteSize ();
    
    lldb::ByteOrder
    GetByteOrder();
    
    float
    GetFloat (lldb::SBError& error, uint32_t offset);
    
    double
    GetDouble (lldb::SBError& error, uint32_t offset);

    long double
    GetLongDouble (lldb::SBError& error, uint32_t offset);
    
    lldb::addr_t
    GetAddress (lldb::SBError& error, uint32_t offset);
    
    uint8_t
    GetUnsignedInt8 (lldb::SBError& error, uint32_t offset);

    uint16_t
    GetUnsignedInt16 (lldb::SBError& error, uint32_t offset);

    uint32_t
    GetUnsignedInt32 (lldb::SBError& error, uint32_t offset);

    uint64_t
    GetUnsignedInt64 (lldb::SBError& error, uint32_t offset);
    
    int8_t
    GetSignedInt8 (lldb::SBError& error, uint32_t offset);
    
    int16_t
    GetSignedInt16 (lldb::SBError& error, uint32_t offset);
    
    int32_t
    GetSignedInt32 (lldb::SBError& error, uint32_t offset);
    
    int64_t
    GetSignedInt64 (lldb::SBError& error, uint32_t offset);
    
    const char*
    GetString (lldb::SBError& error, uint32_t offset);
    
    size_t
    ReadRawData (lldb::SBError& error,
                 uint32_t offset,
                 void *buf,
                 size_t size);
    
    bool
    GetDescription (lldb::SBStream &description);
    
    // it would be nice to have SetData(SBError, const void*, size_t) when endianness and address size can be
    // inferred from the existing DataExtractor, but having two SetData() signatures triggers a SWIG bug where
    // the typemap isn't applied before resolving the overload, and thus the right function never gets called
    void
    SetData(lldb::SBError& error, const void *buf, size_t size, lldb::ByteOrder endian, uint8_t addr_size);
    
    // see SetData() for why we don't have Append(const void* buf, size_t size)
    bool
    Append(const SBData& rhs);
    
protected:
    
#ifndef SWIG
    // Mimic shared pointer...
    lldb_private::DataExtractor *
    get() const;
    
    lldb_private::DataExtractor *
    operator->() const;
    
    lldb::DataExtractorSP &
    operator*();
    
    const lldb::DataExtractorSP &
    operator*() const;
#endif

    SBData (const lldb::DataExtractorSP &data_sp);

    void
    SetOpaque (const lldb::DataExtractorSP &data_sp);

private:
    friend class SBValue;
    
    lldb::DataExtractorSP  m_opaque_sp;
};


} // namespace lldb

#endif // LLDB_SBData_h_
