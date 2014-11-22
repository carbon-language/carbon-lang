//===-- DWARFFormValue.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFFormValue_h_
#define SymbolFileDWARF_DWARFFormValue_h_

#include <stddef.h> // for NULL
#include "DWARFDataExtractor.h"

class DWARFCompileUnit;

class DWARFFormValue
{
public:
    typedef struct ValueTypeTag
    {
        ValueTypeTag() :
            value(),
            data(NULL)
        {
            value.uval = 0;
        }

        union
        {
            uint64_t uval;
            int64_t sval;
            const char* cstr;
        } value;
        const uint8_t* data;
    } ValueType;

    enum
    {
        eValueTypeInvalid = 0,
        eValueTypeUnsigned,
        eValueTypeSigned,
        eValueTypeCStr,
        eValueTypeBlock
    };

    DWARFFormValue();
    DWARFFormValue(const DWARFCompileUnit* cu, dw_form_t form);
    const DWARFCompileUnit* GetCompileUnit () const { return m_cu; }
    void                SetCompileUnit (const DWARFCompileUnit* cu) { m_cu = cu; }
    dw_form_t           Form()  const { return m_form; }
    void                SetForm(dw_form_t form) { m_form = form; }
    const ValueType&    Value() const { return m_value; }
    void                Dump(lldb_private::Stream &s, const lldb_private::DWARFDataExtractor* debug_str_data) const;
    bool                ExtractValue(const lldb_private::DWARFDataExtractor& data,
                                     lldb::offset_t* offset_ptr);
    bool                IsInlinedCStr() const { return (m_value.data != NULL) && m_value.data == (const uint8_t*)m_value.value.cstr; }
    const uint8_t*      BlockData() const;
    uint64_t            Reference() const;
    uint64_t            Reference (dw_offset_t offset) const;
    bool                Boolean() const { return m_value.value.uval != 0; }
    uint64_t            Unsigned() const { return m_value.value.uval; }
    void                SetUnsigned(uint64_t uval) { m_value.value.uval = uval; }
    int64_t             Signed() const { return m_value.value.sval; }
    void                SetSigned(int64_t sval) { m_value.value.sval = sval; }
    const char*         AsCString(const lldb_private::DWARFDataExtractor* debug_str_data_ptr) const;
    bool                SkipValue(const lldb_private::DWARFDataExtractor& debug_info_data, lldb::offset_t *offset_ptr) const;
    static bool         SkipValue(const dw_form_t form, const lldb_private::DWARFDataExtractor& debug_info_data, lldb::offset_t *offset_ptr, const DWARFCompileUnit* cu);
    static bool         IsBlockForm(const dw_form_t form);
    static bool         IsDataForm(const dw_form_t form);
    static const uint8_t * GetFixedFormSizesForAddressSize (uint8_t addr_size, bool is_dwarf64);
    static int          Compare (const DWARFFormValue& a, const DWARFFormValue& b, const lldb_private::DWARFDataExtractor* debug_str_data_ptr);
protected:
    const DWARFCompileUnit* m_cu; // Compile unit for this form
    dw_form_t   m_form;     // Form for this value
    ValueType   m_value;    // Contains all data for the form
};


#endif  // SymbolFileDWARF_DWARFFormValue_h_
