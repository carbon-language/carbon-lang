//===-- SWIG Interface for SBData -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


namespace lldb {

class SBData
{
public:

    SBData ();

    SBData (const SBData &rhs);

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

    bool
    GetDescription (lldb::SBStream &description);

    size_t
    ReadRawData (lldb::SBError& error,
                 uint32_t offset,
                 void *buf,
                 size_t size);

    void
    SetData(lldb::SBError& error, const void *buf, size_t size, lldb::ByteOrder endian, uint8_t addr_size);

    bool
    Append(const SBData& rhs);


};

} // namespace lldb
