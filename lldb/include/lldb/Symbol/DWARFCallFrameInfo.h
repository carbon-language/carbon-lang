//===-- DWARFCallFrameInfo.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DWARFCallFrameInfo_h_
#define liblldb_DWARFCallFrameInfo_h_

// C Includes
// C++ Includes
#include <map>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/VMRange.h"
#include "lldb/Core/dwarf.h"

namespace lldb_private {
//----------------------------------------------------------------------
// DWARFCallFrameInfo
//
// State that describes all register locations for a given address
// range.
//----------------------------------------------------------------------

class DWARFCallFrameInfo
{
public:
    enum
    {
        CFI_AUG_MAX_SIZE = 8,
        CFI_HEADER_SIZE = 8
    };

    class Row;

    class RegisterLocation
    {
    public:

        typedef enum Type
            {
                unspecified,    // not specified, we may be able to assume this is the same register.
                                // gcc doesn't specify all initial values so we really don't know...
                isUndefined,    // reg is not available
                isSame,         // reg is unchanged
                atCFAPlusOffset,// reg = deref(CFA + offset)
                isCFAPlusOffset,// reg = CFA + offset
                inOtherRegister,// reg = other reg
                atDWARFExpression,  // reg = deref(eval(dwarf_expr))
                isDWARFExpression       // reg = eval(dwarf_expr)
            };

        RegisterLocation();

        bool
        operator == (const RegisterLocation& rhs) const;

        void
        Dump(Stream *s, const DWARFCallFrameInfo &cfi, Thread *thread, const Row *row, uint32_t reg_num) const;

        void
        SetUnspecified();

        void
        SetUndefined();

        void
        SetSame() ;

        void
        SetAtCFAPlusOffset (int64_t offset);

        void
        SetIsCFAPlusOffset (int64_t offset);

        void
        SetInRegister (uint32_t reg_num);

        void
        SetAtDWARFExpression (const uint8_t *opcodes, uint32_t len);

        void
        SetIsDWARFExpression (const uint8_t *opcodes, uint32_t len);

    protected:
        Type m_type;            // How do we locate this register?
        union
        {
            // For m_type == atCFAPlusOffset or m_type == isCFAPlusOffset
            int32_t offset;
            // For m_type == inOtherRegister
            uint32_t reg_num; // The register number
            // For m_type == atDWARFExpression or m_type == isDWARFExpression
            struct {
                const uint8_t *opcodes;
                uint32_t length;
            } expr;
        } m_location;
    };

    class Row
    {
    public:

        Row ();

        ~Row ();

        void
        Clear();

        void
        Dump(Stream* s, const DWARFCallFrameInfo &cfi, Thread *thread, lldb::addr_t base_addr) const;

        bool
        GetRegisterInfo (uint32_t reg_num, RegisterLocation& register_location) const;

        void
        SetRegisterInfo (uint32_t reg_num, const RegisterLocation& register_location);

        lldb::addr_t
        GetOffset() const
        {
            return m_offset;
        }

        void
        SetOffset(lldb::addr_t offset)
        {
            m_offset = offset;
        }

        void
        SlideOffset (lldb::addr_t slide)
        {
            m_offset += slide;
        }

        uint32_t
        GetCFARegister () const
        {
            return m_cfa_reg_num;
        }

        void
        SetCFARegister (uint32_t reg_num)
        {
            m_cfa_reg_num = reg_num;
        }

        int32_t
        GetCFAOffset () const
        {
            return m_cfa_offset;
        }

        void
        SetCFAOffset (int32_t offset)
        {
            m_cfa_offset = offset;
        }

    protected:
        typedef std::map<uint32_t, RegisterLocation> collection;
        lldb::addr_t m_offset;          // The an offset into the DBAddressRange that owns this row.
        uint32_t m_cfa_reg_num;     // The Call Frame Address register number
        int32_t  m_cfa_offset;      // The offset from the CFA for this row
        collection m_register_locations;
    };

    //------------------------------------------------------------------
    // Common Information Entry (CIE)
    //------------------------------------------------------------------
protected:

    struct CIE
    {
        typedef lldb::SharedPtr<CIE>::Type shared_ptr;
        dw_offset_t cie_offset;
        uint8_t     version;
        char        augmentation[CFI_AUG_MAX_SIZE];  // This is typically empty or very short. If we ever run into the limit, make this a NSData pointer
        uint32_t    code_align;
        int32_t     data_align;
        uint32_t    return_addr_reg_num;
        dw_offset_t inst_offset;        // offset of CIE instructions in mCFIData
        uint32_t    inst_length;        // length of CIE instructions in mCFIData
        uint8_t     ptr_encoding;

        CIE(dw_offset_t offset);
        ~CIE();

        void
        Dump(Stream *s, Thread* threadState, const ArchSpec *arch, uint32_t reg_kind) const;
    };

    //------------------------------------------------------------------
    // Frame Description Entry (FDE)
    //------------------------------------------------------------------
public:

    class FDE
    {
    public:
        typedef lldb::SharedPtr<FDE>::Type shared_ptr;

        FDE (uint32_t offset, const AddressRange &range);
        ~FDE();

        const AddressRange &
        GetAddressRange() const;

        void
        AppendRow (const Row &row);

        bool
        IsValidRowIndex (uint32_t idx) const;

        void
        Dump (Stream *s, const DWARFCallFrameInfo &cfi, Thread* thread) const;

        const Row&
        GetRowAtIndex (uint32_t idx);

    protected:
        typedef std::vector<Row> collection;
        uint32_t m_fde_offset;
        AddressRange m_range;
        collection m_row_list;
    private:
        DISALLOW_COPY_AND_ASSIGN (FDE);
    };

    DWARFCallFrameInfo(ObjectFile *objfile, lldb_private::Section *section, uint32_t reg_kind);

    ~DWARFCallFrameInfo();

    bool
    IsEHFrame() const;

    const ArchSpec *
    GetArchitecture() const;

    uint32_t
    GetRegisterKind () const;

    void
    SetRegisterKind (uint32_t reg_kind);

    void
    Index ();

//  bool        UnwindRegister (const uint32_t reg_num, const Thread* currState, const Row* row, Thread* unwindState);
//  uint32_t    UnwindThreadState(const Thread* curr_state, bool is_first_frame, Thread* unwound_state);
    const FDE *
    FindFDE(const Address &addr);

    void
    Dump(Stream *s, Thread *thread) const;

    void
    ParseAll();
protected:

    enum
    {
        eFlagParsedIndex = (1 << 0)
    };

    typedef std::map<off_t, CIE::shared_ptr> cie_map_t;
    struct FDEInfo
    {
        off_t fde_offset;
        FDE::shared_ptr fde_sp;
        FDEInfo (off_t offset);
        FDEInfo ();

    };
    typedef std::map<VMRange, FDEInfo> fde_map_t;

    ObjectFile *    m_objfile;
    lldb_private::Section *     m_section;
    uint32_t        m_reg_kind;
    Flags           m_flags;
    DataExtractor   m_cfi_data;
    cie_map_t       m_cie_map;
    fde_map_t       m_fde_map;

    const CIE*
    GetCIE (uint32_t offset);

    void
    ParseInstructions(const CIE *cie, FDE *fde, uint32_t instr_offset, uint32_t instr_length);

    CIE::shared_ptr
    ParseCIE (const uint32_t cie_offset);

    FDE::shared_ptr
    ParseFDE (const uint32_t fde_offset);
};

} // namespace lldb_private

#endif  // liblldb_DWARFCallFrameInfo_h_
