//===-- DWARFCallFrameInfo.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// C Includes
// C++ Includes
#include <list>

// Other libraries and framework includes
// Project includes
#include "lldb/Symbol/DWARFCallFrameInfo.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Core/Section.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

static void
DumpRegisterName (Stream *s, Thread *thread, const ArchSpec *arch, uint32_t reg_kind, uint32_t reg_num)
{
    const char *reg_name = NULL;
    RegisterContext *reg_ctx = NULL;
    if (thread)
    {
        reg_ctx = thread->GetRegisterContext();
        if (reg_ctx)
            reg_name = reg_ctx->GetRegisterName (reg_ctx->ConvertRegisterKindToRegisterNumber (reg_kind, reg_num));
    }

    if (reg_name == NULL && arch != NULL)
    {
        switch (reg_kind)
        {
        case eRegisterKindDWARF: reg_name = arch->GetRegisterName(reg_num, eRegisterKindDWARF); break;
        case eRegisterKindGCC: reg_name = arch->GetRegisterName(reg_num, eRegisterKindGCC); break;
        default:
            break;
        }
    }

    if (reg_name)
        s->PutCString(reg_name);
    else
    {
        const char *reg_kind_name = NULL;
        switch (reg_kind)
        {
        case eRegisterKindDWARF: reg_kind_name = "dwarf-reg"; break;
        case eRegisterKindGCC: reg_kind_name = "compiler-reg"; break;
        case eRegisterKindGeneric: reg_kind_name = "generic-reg"; break;
        default:
            break;
        }
        if (reg_kind_name)
            s->Printf("%s(%u)", reg_kind_name, reg_num);
        else
            s->Printf("reg(%d.%u)", reg_kind, reg_num);
    }
}


#pragma mark DWARFCallFrameInfo::RegisterLocation

DWARFCallFrameInfo::RegisterLocation::RegisterLocation() :
    m_type(isSame)
{
}


bool
DWARFCallFrameInfo::RegisterLocation::operator == (const DWARFCallFrameInfo::RegisterLocation& rhs) const
{
    if (m_type != rhs.m_type)
        return false;
    switch (m_type)
    {
        case unspecified:
        case isUndefined:
        case isSame:
            return true;

        case atCFAPlusOffset:
            return m_location.offset == rhs.m_location.offset;

        case isCFAPlusOffset:
            return m_location.offset == rhs.m_location.offset;

        case inOtherRegister:
            return m_location.reg_num == rhs.m_location.reg_num;

        default:
            break;
    }
    return false;
}

void
DWARFCallFrameInfo::RegisterLocation::SetUnspecified()
{
    m_type = unspecified;
}

void
DWARFCallFrameInfo::RegisterLocation::SetUndefined()
{
    m_type = isUndefined;
}

void
DWARFCallFrameInfo::RegisterLocation::SetSame()
{
    m_type = isSame;
}

void
DWARFCallFrameInfo::RegisterLocation::SetAtCFAPlusOffset(int64_t offset)
{
    m_type = atCFAPlusOffset;
    m_location.offset = offset;
}

void
DWARFCallFrameInfo::RegisterLocation::SetIsCFAPlusOffset(int64_t offset)
{
    m_type = isCFAPlusOffset;
    m_location.offset = offset;
}

void
DWARFCallFrameInfo::RegisterLocation::SetInRegister (uint32_t reg_num)
{
    m_type = inOtherRegister;
    m_location.reg_num = reg_num;
}

void
DWARFCallFrameInfo::RegisterLocation::SetAtDWARFExpression(const uint8_t *opcodes, uint32_t len)
{
    m_type = atDWARFExpression;
    m_location.expr.opcodes = opcodes;
    m_location.expr.length = len;
}

void
DWARFCallFrameInfo::RegisterLocation::SetIsDWARFExpression(const uint8_t *opcodes, uint32_t len)
{
    m_type = isDWARFExpression;
    m_location.expr.opcodes = opcodes;
    m_location.expr.length = len;
}

void
DWARFCallFrameInfo::RegisterLocation::Dump(Stream *s, const DWARFCallFrameInfo &cfi, Thread *thread, const Row *row, uint32_t reg_num) const
{
    const ArchSpec *arch = cfi.GetArchitecture();
    const uint32_t reg_kind = cfi.GetRegisterKind();

    DumpRegisterName (s, thread, arch, reg_kind, reg_num);
    s->PutChar('=');

    switch (m_type)
    {
    case unspecified:
        s->PutChar('?');
        break;

    case isUndefined:
        s->PutCString("undefined");
        break;

    case isSame:
        s->PutCString("same");
        break;

    case atCFAPlusOffset:
        s->PutChar('[');
        // Fall through to isCFAPlusOffset...
    case isCFAPlusOffset:
        {
            DumpRegisterName (s, thread, arch, reg_kind, row->GetCFARegister());
            int32_t offset = row->GetCFAOffset() + m_location.offset;
            if (offset != 0)
                s->Printf("%-+d", offset);
            if (m_type == atCFAPlusOffset)
                s->PutChar(']');
        }
        break;

    case inOtherRegister:
        DumpRegisterName (s, thread, arch, reg_kind, m_location.reg_num);
        break;

    case atDWARFExpression:
        s->PutCString("[EXPR] ");
        break;

    case isDWARFExpression:
        s->PutCString("EXPR ");
        break;
    }
}


#pragma mark DWARFCallFrameInfo::Row

DWARFCallFrameInfo::Row::Row() :
    m_offset(0),
    m_cfa_reg_num(0),
    m_cfa_offset(0),
    m_register_locations()
{
}

DWARFCallFrameInfo::Row::~Row()
{
}

void
DWARFCallFrameInfo::Row::Clear()
{
    m_register_locations.clear();
}
bool
DWARFCallFrameInfo::Row::GetRegisterInfo (uint32_t reg_num, DWARFCallFrameInfo::RegisterLocation& register_location) const
{
    collection::const_iterator pos = m_register_locations.find(reg_num);
    if (pos != m_register_locations.end())
    {
        register_location = pos->second;
        return true;
    }
    return false;
}

void
DWARFCallFrameInfo::Row::SetRegisterInfo (uint32_t reg_num, const RegisterLocation& register_location)
{
    m_register_locations[reg_num] = register_location;
}


void
DWARFCallFrameInfo::Row::Dump(Stream* s, const DWARFCallFrameInfo &cfi, Thread *thread, lldb::addr_t base_addr) const
{
    const ArchSpec *arch = cfi.GetArchitecture();
    const uint32_t reg_kind = cfi.GetRegisterKind();
    collection::const_iterator pos, end = m_register_locations.end();
    s->Indent();
    s->Printf("0x%16.16llx: CFA=", m_offset + base_addr);
    DumpRegisterName(s, thread, arch, reg_kind, m_cfa_reg_num);
    if (m_cfa_offset != 0)
        s->Printf("%-+lld", m_cfa_offset);

    for (pos = m_register_locations.begin(); pos != end; ++pos)
    {
        s->PutChar(' ');
        pos->second.Dump(s, cfi, thread, this, pos->first);
    }
    s->EOL();
}


#pragma mark DWARFCallFrameInfo::FDE


DWARFCallFrameInfo::FDE::FDE (dw_offset_t offset, const AddressRange &range) :
    m_fde_offset (offset),
    m_range (range),
    m_row_list ()
{
}

DWARFCallFrameInfo::FDE::~FDE()
{
}

void
DWARFCallFrameInfo::FDE::AppendRow (const Row &row)
{
    if (m_row_list.empty() || m_row_list.back().GetOffset() != row.GetOffset())
        m_row_list.push_back(row);
    else
        m_row_list.back() = row;
}

void
DWARFCallFrameInfo::FDE::Dump (Stream *s, const DWARFCallFrameInfo &cfi, Thread* thread) const
{
    s->Indent();
    s->Printf("FDE{0x%8.8x} ", m_fde_offset);
    m_range.Dump(s, NULL, Address::DumpStyleFileAddress);
    lldb::addr_t fde_base_addr = m_range.GetBaseAddress().GetFileAddress();
    s->EOL();
    s->IndentMore();
    collection::const_iterator pos, end = m_row_list.end();
    for (pos = m_row_list.begin(); pos != end; ++pos)
    {
        pos->Dump(s, cfi, thread, fde_base_addr);
    }
    s->IndentLess();
}

const AddressRange &
DWARFCallFrameInfo::FDE::GetAddressRange() const
{
    return m_range;
}

bool
DWARFCallFrameInfo::FDE::IsValidRowIndex (uint32_t idx) const
{
    return idx < m_row_list.size();
}

const DWARFCallFrameInfo::Row&
DWARFCallFrameInfo::FDE::GetRowAtIndex (uint32_t idx)
{
    // You must call IsValidRowIndex(idx) first before calling this!!!
    return m_row_list[idx];
}
#pragma mark DWARFCallFrameInfo::FDEInfo

DWARFCallFrameInfo::FDEInfo::FDEInfo () :
    fde_offset (0),
    fde_sp()
{
}

DWARFCallFrameInfo::FDEInfo::FDEInfo (off_t offset) :
    fde_offset(offset),
    fde_sp()
{
}

#pragma mark DWARFCallFrameInfo::CIE

DWARFCallFrameInfo::CIE::CIE(dw_offset_t offset) :
    cie_offset (offset),
    version (0),
    augmentation(),
    code_align (0),
    data_align (0),
    return_addr_reg_num (0),
    inst_offset (0),
    inst_length (0),
    ptr_encoding (DW_GNU_EH_PE_absptr)
{
}


DWARFCallFrameInfo::CIE::~CIE()
{
}

void
DWARFCallFrameInfo::CIE::Dump(Stream *s, Thread* thread, const ArchSpec *arch, uint32_t reg_kind) const
{
    s->Indent();
    s->Printf("CIE{0x%8.8x} version=%u, code_align=%u, data_align=%d, return_addr_reg=", cie_offset, version, code_align, data_align);
    DumpRegisterName(s, thread, arch, reg_kind, return_addr_reg_num);
    s->Printf(", instr_offset=0x%8.8x, instr_length=%u, ptr_encoding=0x%02x\n",
            inst_offset,
            inst_length,
            ptr_encoding);
}

#pragma mark DWARFCallFrameInfo::CIE

DWARFCallFrameInfo::DWARFCallFrameInfo(ObjectFile *objfile, Section *section, uint32_t reg_kind) :
    m_objfile (objfile),
    m_section (section),
    m_reg_kind (reg_kind),  // The flavor of registers that the CFI data uses (One of the defines that starts with "LLDB_REGKIND_")
    m_cfi_data (),
    m_cie_map (),
    m_fde_map ()
{
    if (objfile && section)
    {
        section->ReadSectionDataFromObjectFile (objfile, m_cfi_data);
    }
}

DWARFCallFrameInfo::~DWARFCallFrameInfo()
{
}

bool
DWARFCallFrameInfo::IsEHFrame() const
{
    return (m_reg_kind == eRegisterKindGCC);
}

const ArchSpec *
DWARFCallFrameInfo::GetArchitecture() const
{
    if (m_objfile && m_objfile->GetModule())
        return &m_objfile->GetModule()->GetArchitecture();
    return NULL;
}

uint32_t
DWARFCallFrameInfo::GetRegisterKind () const
{
    return m_reg_kind;
}

void
DWARFCallFrameInfo::SetRegisterKind (uint32_t reg_kind)
{
    m_reg_kind = reg_kind;
}




const DWARFCallFrameInfo::CIE*
DWARFCallFrameInfo::GetCIE(dw_offset_t cie_offset)
{
    Index ();

    cie_map_t::iterator pos = m_cie_map.find(cie_offset);

    if (pos != m_cie_map.end())
    {
        // Parse and cache the CIE
        if (pos->second.get() == NULL)
            pos->second = ParseCIE (cie_offset);

        return pos->second.get();
    }
    return NULL;
}

DWARFCallFrameInfo::CIE::shared_ptr
DWARFCallFrameInfo::ParseCIE (const dw_offset_t cie_offset)
{
    CIE::shared_ptr cie_sp(new CIE(cie_offset));
    const bool for_eh_frame = IsEHFrame();
    dw_offset_t offset = cie_offset;
    const uint32_t length = m_cfi_data.GetU32(&offset);
    const dw_offset_t cie_id = m_cfi_data.GetU32(&offset);
    const dw_offset_t end_offset = cie_offset + length + 4;
    if (length > 0 && (!for_eh_frame && cie_id == 0xfffffffful) || (for_eh_frame && cie_id == 0ul))
    {
        size_t i;
        //    cie.offset = cie_offset;
        //    cie.length = length;
        //    cie.cieID = cieID;
        cie_sp->ptr_encoding = DW_GNU_EH_PE_absptr;
        cie_sp->version = m_cfi_data.GetU8(&offset);

        for (i=0; i<CFI_AUG_MAX_SIZE; ++i)
        {
            cie_sp->augmentation[i] = m_cfi_data.GetU8(&offset);
            if (cie_sp->augmentation[i] == '\0')
            {
                // Zero out remaining bytes in augmentation string
                for (size_t j = i+1; j<CFI_AUG_MAX_SIZE; ++j)
                    cie_sp->augmentation[j] = '\0';

                break;
            }
        }

        if (i == CFI_AUG_MAX_SIZE && cie_sp->augmentation[CFI_AUG_MAX_SIZE-1] != '\0')
        {
            fprintf(stderr, "CIE parse error: CIE augmentation string was too large for the fixed sized buffer of %d bytes.\n", CFI_AUG_MAX_SIZE);
            return cie_sp;
        }
        cie_sp->code_align = (uint32_t)m_cfi_data.GetULEB128(&offset);
        cie_sp->data_align = (int32_t)m_cfi_data.GetSLEB128(&offset);
        cie_sp->return_addr_reg_num = m_cfi_data.GetU8(&offset);

        if (cie_sp->augmentation[0])
        {
            // Get the length of the eh_frame augmentation data
            // which starts with a ULEB128 length in bytes
            const size_t aug_data_len = (size_t)m_cfi_data.GetULEB128(&offset);
            const size_t aug_data_end = offset + aug_data_len;
            const size_t aug_str_len = strlen(cie_sp->augmentation);
            // A 'z' may be present as the first character of the string.
            // If present, the Augmentation Data field shall be present.
            // The contents of the Augmentation Data shall be intepreted
            // according to other characters in the Augmentation String.
            if (cie_sp->augmentation[0] == 'z')
            {
                // Extract the Augmentation Data
                size_t aug_str_idx = 0;
                for (aug_str_idx = 1; aug_str_idx < aug_str_len; aug_str_idx++)
                {
                    char aug = cie_sp->augmentation[aug_str_idx];
                    switch (aug)
                    {
                        case 'L':
                            // Indicates the presence of one argument in the
                            // Augmentation Data of the CIE, and a corresponding
                            // argument in the Augmentation Data of the FDE. The
                            // argument in the Augmentation Data of the CIE is
                            // 1-byte and represents the pointer encoding used
                            // for the argument in the Augmentation Data of the
                            // FDE, which is the address of a language-specific
                            // data area (LSDA). The size of the LSDA pointer is
                            // specified by the pointer encoding used.
                            m_cfi_data.GetU8(&offset);
                            break;

                        case 'P':
                            // Indicates the presence of two arguments in the
                            // Augmentation Data of the cie_sp-> The first argument
                            // is 1-byte and represents the pointer encoding
                            // used for the second argument, which is the
                            // address of a personality routine handler. The
                            // size of the personality routine pointer is
                            // specified by the pointer encoding used.
                        {
                            uint8_t arg_ptr_encoding = m_cfi_data.GetU8(&offset);
                            m_cfi_data.GetGNUEHPointer(&offset, arg_ptr_encoding, LLDB_INVALID_ADDRESS, LLDB_INVALID_ADDRESS, LLDB_INVALID_ADDRESS);
                        }
                            break;

                        case 'R':
                            // A 'R' may be present at any position after the
                            // first character of the string. The Augmentation
                            // Data shall include a 1 byte argument that
                            // represents the pointer encoding for the address
                            // pointers used in the FDE.
                            cie_sp->ptr_encoding = m_cfi_data.GetU8(&offset);
                            break;
                    }
                }
            }
            else if (strcmp(cie_sp->augmentation, "eh") == 0)
            {
                // If the Augmentation string has the value "eh", then
                // the EH Data field shall be present
            }

            // Set the offset to be the end of the augmentation data just in case
            // we didn't understand any of the data.
            offset = (uint32_t)aug_data_end;
        }

        if (end_offset > offset)
        {
            cie_sp->inst_offset = offset;
            cie_sp->inst_length = end_offset - offset;
        }
    }

    return cie_sp;
}

DWARFCallFrameInfo::FDE::shared_ptr
DWARFCallFrameInfo::ParseFDE(const dw_offset_t fde_offset)
{
    const bool for_eh_frame = IsEHFrame();
    FDE::shared_ptr fde_sp;

    dw_offset_t offset = fde_offset;
    const uint32_t length = m_cfi_data.GetU32(&offset);
    dw_offset_t cie_offset = m_cfi_data.GetU32(&offset);
    const dw_offset_t end_offset = fde_offset + length + 4;

    // Translate the CIE_id from the eh_frame format, which
    // is relative to the FDE offset, into a __eh_frame section
    // offset
    if (for_eh_frame)
        cie_offset = offset - (cie_offset + 4);

    const CIE* cie = GetCIE(cie_offset);
    if (cie)
    {
        const lldb::addr_t pc_rel_addr = m_section->GetFileAddress();
        const lldb::addr_t text_addr = LLDB_INVALID_ADDRESS;
        const lldb::addr_t data_addr = LLDB_INVALID_ADDRESS;
        lldb::addr_t range_base = m_cfi_data.GetGNUEHPointer(&offset, cie->ptr_encoding, pc_rel_addr, text_addr, data_addr);
        lldb::addr_t range_len = m_cfi_data.GetGNUEHPointer(&offset, cie->ptr_encoding & DW_GNU_EH_PE_MASK_ENCODING, pc_rel_addr, text_addr, data_addr);

        if (cie->augmentation[0] == 'z')
        {
            uint32_t aug_data_len = (uint32_t)m_cfi_data.GetULEB128(&offset);
            offset += aug_data_len;
        }

        AddressRange fde_range (range_base, range_len, m_objfile->GetSectionList ());
        fde_sp.reset(new FDE(fde_offset, fde_range));
        if (offset < end_offset)
        {
            dw_offset_t fde_instr_offset = offset;
            uint32_t fde_instr_length = end_offset - offset;
            if (cie->inst_length > 0)
                ParseInstructions(cie, fde_sp.get(), cie->inst_offset, cie->inst_length);
            ParseInstructions(cie, fde_sp.get(), fde_instr_offset, fde_instr_length);
        }
    }
    return fde_sp;
}

const DWARFCallFrameInfo::FDE *
DWARFCallFrameInfo::FindFDE(const Address &addr)
{
    Index ();

    VMRange find_range(addr.GetFileAddress(), 0);
    fde_map_t::iterator pos = m_fde_map.lower_bound (find_range);
    fde_map_t::iterator end = m_fde_map.end();

    if (pos != end)
    {
        if (pos->first.Contains(find_range.GetBaseAddress()))
        {
            // Parse and cache the FDE if we already haven't
            if (pos->second.fde_sp.get() == NULL)
                pos->second.fde_sp = ParseFDE(pos->second.fde_offset);

            return pos->second.fde_sp.get();
        }
    }
    return NULL;
}


void
DWARFCallFrameInfo::Index ()
{
    if (m_flags.IsClear(eFlagParsedIndex))
    {
        m_flags.Set (eFlagParsedIndex);
        const bool for_eh_frame = IsEHFrame();
        CIE::shared_ptr empty_cie_sp;
        dw_offset_t offset = 0;
        // Parse all of the CIEs first since we will need them to be able to
        // properly parse the FDE addresses due to them possibly having
        // GNU pointer encodings in their augmentations...
        while (m_cfi_data.ValidOffsetForDataOfSize(offset, 8))
        {
            const dw_offset_t curr_offset = offset;
            const uint32_t length = m_cfi_data.GetU32(&offset);
            const dw_offset_t next_offset = offset + length;
            const dw_offset_t cie_id = m_cfi_data.GetU32(&offset);

            bool is_cie = for_eh_frame ?  cie_id == 0 : cie_id == UINT32_MAX;
            if (is_cie)
                m_cie_map[curr_offset]= ParseCIE(curr_offset);

            offset = next_offset;
        }

        // Now go back through and index all FDEs
        offset = 0;
        const lldb::addr_t pc_rel_addr = m_section->GetFileAddress();
        const lldb::addr_t text_addr = LLDB_INVALID_ADDRESS;
        const lldb::addr_t data_addr = LLDB_INVALID_ADDRESS;
        while (m_cfi_data.ValidOffsetForDataOfSize(offset, 8))
        {
            const dw_offset_t curr_offset = offset;
            const uint32_t length = m_cfi_data.GetU32(&offset);
            const dw_offset_t next_offset = offset + length;
            const dw_offset_t cie_id = m_cfi_data.GetU32(&offset);

            bool is_fde = for_eh_frame ?  cie_id != 0 : cie_id != UINT32_MAX;
            if (is_fde)
            {
                dw_offset_t cie_offset;
                if (for_eh_frame)
                    cie_offset = offset - (cie_id + 4);
                else
                    cie_offset = cie_id;

                const CIE* cie = GetCIE(cie_offset);
                assert(cie);
                lldb::addr_t addr = m_cfi_data.GetGNUEHPointer(&offset, cie->ptr_encoding, pc_rel_addr, text_addr, data_addr);
                lldb::addr_t length = m_cfi_data.GetGNUEHPointer(&offset, cie->ptr_encoding & DW_GNU_EH_PE_MASK_ENCODING, pc_rel_addr, text_addr, data_addr);
                m_fde_map[VMRange(addr, addr + length)] = FDEInfo(curr_offset);
            }

            offset = next_offset;
        }
    }
}

//----------------------------------------------------------------------
// Parse instructions for a FDE. The initial instruction for the CIE
// are parsed first, then the instructions for the FDE are parsed
//----------------------------------------------------------------------
void
DWARFCallFrameInfo::ParseInstructions(const CIE *cie, FDE *fde, dw_offset_t instr_offset, uint32_t instr_length)
{
    if (cie != NULL && fde == NULL)
        return;

    uint32_t reg_num = 0;
    int32_t op_offset = 0;
    uint32_t tmp_uval32;
    uint32_t code_align = cie->code_align;
    int32_t data_align = cie->data_align;
    typedef std::list<Row> RowStack;

    RowStack row_stack;
    Row row;
    if (fde->IsValidRowIndex(0))
        row = fde->GetRowAtIndex(0);

    dw_offset_t offset = instr_offset;
    const dw_offset_t end_offset = instr_offset + instr_length;
    RegisterLocation reg_location;
    while (m_cfi_data.ValidOffset(offset) && offset < end_offset)
    {
        uint8_t inst = m_cfi_data.GetU8(&offset);
        uint8_t primary_opcode  = inst & 0xC0;
        uint8_t extended_opcode = inst & 0x3F;

        if (primary_opcode)
        {
            switch (primary_opcode)
            {
                case DW_CFA_advance_loc :   // (Row Creation Instruction)
                    {   // 0x40 - high 2 bits are 0x1, lower 6 bits are delta
                        // takes a single argument that represents a constant delta. The
                        // required action is to create a new table row with a location
                        // value that is computed by taking the current entry's location
                        // value and adding (delta * code_align). All other
                        // values in the new row are initially identical to the current row.
                        fde->AppendRow(row);
                        row.SlideOffset(extended_opcode * code_align);
                    }
                    break;

                case DW_CFA_offset      :
                    {   // 0x80 - high 2 bits are 0x2, lower 6 bits are register
                        // takes two arguments: an unsigned LEB128 constant representing a
                        // factored offset and a register number. The required action is to
                        // change the rule for the register indicated by the register number
                        // to be an offset(N) rule with a value of
                        // (N = factored offset * data_align).
                        reg_num = extended_opcode;
                        op_offset = (int32_t)m_cfi_data.GetULEB128(&offset) * data_align;
                        reg_location.SetAtCFAPlusOffset(op_offset);
                        row.SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_restore     :
                    {   // 0xC0 - high 2 bits are 0x3, lower 6 bits are register
                        // takes a single argument that represents a register number. The
                        // required action is to change the rule for the indicated register
                        // to the rule assigned it by the initial_instructions in the CIE.
                        reg_num = extended_opcode;
                        // We only keep enough register locations around to
                        // unwind what is in our thread, and these are organized
                        // by the register index in that state, so we need to convert our
                        // GCC register number from the EH frame info, to a registe index

                        if (fde->IsValidRowIndex(0) && fde->GetRowAtIndex(0).GetRegisterInfo(reg_num, reg_location))
                            row.SetRegisterInfo (reg_num, reg_location);
                    }
                    break;
            }
        }
        else
        {
            switch (extended_opcode)
            {
                case DW_CFA_nop                 : // 0x0
                    break;

                case DW_CFA_set_loc             : // 0x1 (Row Creation Instruction)
                    {
                        // DW_CFA_set_loc takes a single argument that represents an address.
                        // The required action is to create a new table row using the
                        // specified address as the location. All other values in the new row
                        // are initially identical to the current row. The new location value
                        // should always be greater than the current one.
                        fde->AppendRow(row);
                        row.SetOffset(m_cfi_data.GetPointer(&offset) - fde->GetAddressRange().GetBaseAddress().GetFileAddress());
                    }
                    break;

                case DW_CFA_advance_loc1        : // 0x2 (Row Creation Instruction)
                    {
                        // takes a single uword argument that represents a constant delta.
                        // This instruction is identical to DW_CFA_advance_loc except for the
                        // encoding and size of the delta argument.
                        fde->AppendRow(row);
                        row.SlideOffset (m_cfi_data.GetU8(&offset) * code_align);
                    }
                    break;

                case DW_CFA_advance_loc2        : // 0x3 (Row Creation Instruction)
                    {
                        // takes a single uword argument that represents a constant delta.
                        // This instruction is identical to DW_CFA_advance_loc except for the
                        // encoding and size of the delta argument.
                        fde->AppendRow(row);
                        row.SlideOffset (m_cfi_data.GetU16(&offset) * code_align);
                    }
                    break;

                case DW_CFA_advance_loc4        : // 0x4 (Row Creation Instruction)
                    {
                        // takes a single uword argument that represents a constant delta.
                        // This instruction is identical to DW_CFA_advance_loc except for the
                        // encoding and size of the delta argument.
                        fde->AppendRow(row);
                        row.SlideOffset (m_cfi_data.GetU32(&offset) * code_align);
                    }
                    break;

                case DW_CFA_offset_extended     : // 0x5
                    {
                        // takes two unsigned LEB128 arguments representing a register number
                        // and a factored offset. This instruction is identical to DW_CFA_offset
                        // except for the encoding and size of the register argument.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        op_offset = (int32_t)m_cfi_data.GetULEB128(&offset) * data_align;
                        reg_location.SetAtCFAPlusOffset(op_offset);
                        row.SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_restore_extended    : // 0x6
                    {
                        // takes a single unsigned LEB128 argument that represents a register
                        // number. This instruction is identical to DW_CFA_restore except for
                        // the encoding and size of the register argument.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        if (fde->IsValidRowIndex(0) && fde->GetRowAtIndex(0).GetRegisterInfo(reg_num, reg_location))
                            row.SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_undefined           : // 0x7
                    {
                        // takes a single unsigned LEB128 argument that represents a register
                        // number. The required action is to set the rule for the specified
                        // register to undefined.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        reg_location.SetUndefined();
                        row.SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_same_value          : // 0x8
                    {
                        // takes a single unsigned LEB128 argument that represents a register
                        // number. The required action is to set the rule for the specified
                        // register to same value.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        reg_location.SetSame();
                        row.SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_register            : // 0x9
                    {
                        // takes two unsigned LEB128 arguments representing register numbers.
                        // The required action is to set the rule for the first register to be
                        // the second register.

                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        uint32_t other_reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        reg_location.SetInRegister(other_reg_num);
                        row.SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_remember_state      : // 0xA
                    // These instructions define a stack of information. Encountering the
                    // DW_CFA_remember_state instruction means to save the rules for every
                    // register on the current row on the stack. Encountering the
                    // DW_CFA_restore_state instruction means to pop the set of rules off
                    // the stack and place them in the current row. (This operation is
                    // useful for compilers that move epilogue code into the body of a
                    // function.)
                    row_stack.push_back(row);
                    break;

                case DW_CFA_restore_state       : // 0xB
                    // These instructions define a stack of information. Encountering the
                    // DW_CFA_remember_state instruction means to save the rules for every
                    // register on the current row on the stack. Encountering the
                    // DW_CFA_restore_state instruction means to pop the set of rules off
                    // the stack and place them in the current row. (This operation is
                    // useful for compilers that move epilogue code into the body of a
                    // function.)
                    {
                        row = row_stack.back();
                        row_stack.pop_back();
                    }
                    break;

                case DW_CFA_def_cfa             : // 0xC    (CFA Definition Instruction)
                    {
                        // Takes two unsigned LEB128 operands representing a register
                        // number and a (non-factored) offset. The required action
                        // is to define the current CFA rule to use the provided
                        // register and offset.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        op_offset = (int32_t)m_cfi_data.GetULEB128(&offset);
                        row.SetCFARegister (reg_num);
                        row.SetCFAOffset (op_offset);
                    }
                    break;

                case DW_CFA_def_cfa_register    : // 0xD    (CFA Definition Instruction)
                    {
                        // takes a single unsigned LEB128 argument representing a register
                        // number. The required action is to define the current CFA rule to
                        // use the provided register (but to keep the old offset).
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        row.SetCFARegister (reg_num);
                    }
                    break;

                case DW_CFA_def_cfa_offset      : // 0xE    (CFA Definition Instruction)
                    {
                        // Takes a single unsigned LEB128 operand representing a
                        // (non-factored) offset. The required action is to define
                        // the current CFA rule to use the provided offset (but
                        // to keep the old register).
                        op_offset = (int32_t)m_cfi_data.GetULEB128(&offset);
                        row.SetCFAOffset (op_offset);
                    }
                    break;

                case DW_CFA_def_cfa_expression  : // 0xF    (CFA Definition Instruction)
                    {
                        size_t block_len = (size_t)m_cfi_data.GetULEB128(&offset);
                        offset += (uint32_t)block_len;
                    }
                    break;

                case DW_CFA_expression          : // 0x10
                    {
                        // Takes two operands: an unsigned LEB128 value representing
                        // a register number, and a DW_FORM_block value representing a DWARF
                        // expression. The required action is to change the rule for the
                        // register indicated by the register number to be an expression(E)
                        // rule where E is the DWARF expression. That is, the DWARF
                        // expression computes the address. The value of the CFA is
                        // pushed on the DWARF evaluation stack prior to execution of
                        // the DWARF expression.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        uint32_t block_len = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        const uint8_t *block_data = (uint8_t *)m_cfi_data.GetData(&offset, block_len);

                        reg_location.SetAtDWARFExpression(block_data, block_len);
                        row.SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_offset_extended_sf  : // 0x11
                    {
                        // takes two operands: an unsigned LEB128 value representing a
                        // register number and a signed LEB128 factored offset. This
                        // instruction is identical to DW_CFA_offset_extended except
                        //that the second operand is signed and factored.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        op_offset = (int32_t)m_cfi_data.GetSLEB128(&offset) * data_align;
                        reg_location.SetAtCFAPlusOffset(op_offset);
                        row.SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_def_cfa_sf          : // 0x12   (CFA Definition Instruction)
                    {
                        // Takes two operands: an unsigned LEB128 value representing
                        // a register number and a signed LEB128 factored offset.
                        // This instruction is identical to DW_CFA_def_cfa except
                        // that the second operand is signed and factored.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        op_offset = (int32_t)m_cfi_data.GetSLEB128(&offset) * data_align;
                        row.SetCFARegister (reg_num);
                        row.SetCFAOffset (op_offset);
                    }
                    break;

                case DW_CFA_def_cfa_offset_sf   : // 0x13   (CFA Definition Instruction)
                    {
                        // takes a signed LEB128 operand representing a factored
                        // offset. This instruction is identical to  DW_CFA_def_cfa_offset
                        // except that the operand is signed and factored.
                        op_offset = (int32_t)m_cfi_data.GetSLEB128(&offset) * data_align;
                        row.SetCFAOffset (op_offset);
                    }
                    break;

                case DW_CFA_val_expression      :   // 0x16
                    {
                        // takes two operands: an unsigned LEB128 value representing a register
                        // number, and a DW_FORM_block value representing a DWARF expression.
                        // The required action is to change the rule for the register indicated
                        // by the register number to be a val_expression(E) rule where E is the
                        // DWARF expression. That is, the DWARF expression computes the value of
                        // the given register. The value of the CFA is pushed on the DWARF
                        // evaluation stack prior to execution of the DWARF expression.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        uint32_t block_len = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        const uint8_t* block_data = (uint8_t*)m_cfi_data.GetData(&offset, block_len);
//#if defined(__i386__) || defined(__x86_64__)
//                      // The EH frame info for EIP and RIP contains code that looks for traps to
//                      // be a specific type and increments the PC.
//                      // For i386:
//                      // DW_CFA_val_expression where:
//                      // eip = DW_OP_breg6(+28), DW_OP_deref, DW_OP_dup, DW_OP_plus_uconst(0x34),
//                      //       DW_OP_deref, DW_OP_swap, DW_OP_plus_uconst(0), DW_OP_deref,
//                      //       DW_OP_dup, DW_OP_lit3, DW_OP_ne, DW_OP_swap, DW_OP_lit4, DW_OP_ne,
//                      //       DW_OP_and, DW_OP_plus
//                      // This basically does a:
//                      // eip = ucontenxt.mcontext32->gpr.eip;
//                      // if (ucontenxt.mcontext32->exc.trapno != 3 && ucontenxt.mcontext32->exc.trapno != 4)
//                      //   eip++;
//                      //
//                      // For x86_64:
//                      // DW_CFA_val_expression where:
//                      // rip =  DW_OP_breg3(+48), DW_OP_deref, DW_OP_dup, DW_OP_plus_uconst(0x90), DW_OP_deref,
//                      //          DW_OP_swap, DW_OP_plus_uconst(0), DW_OP_deref_size(4), DW_OP_dup, DW_OP_lit3,
//                      //          DW_OP_ne, DW_OP_swap, DW_OP_lit4, DW_OP_ne, DW_OP_and, DW_OP_plus
//                      // This basically does a:
//                      // rip = ucontenxt.mcontext64->gpr.rip;
//                      // if (ucontenxt.mcontext64->exc.trapno != 3 && ucontenxt.mcontext64->exc.trapno != 4)
//                      //   rip++;
//                      // The trap comparisons and increments are not needed as it hoses up the unwound PC which
//                      // is expected to point at least past the instruction that causes the fault/trap. So we
//                      // take it out by trimming the expression right at the first "DW_OP_swap" opcodes
//                      if (block_data != NULL && thread->GetPCRegNum(Thread::GCC) == reg_num)
//                      {
//                          if (thread->Is64Bit())
//                          {
//                              if (block_len > 9 && block_data[8] == DW_OP_swap && block_data[9] == DW_OP_plus_uconst)
//                                  block_len = 8;
//                          }
//                          else
//                          {
//                              if (block_len > 8 && block_data[7] == DW_OP_swap && block_data[8] == DW_OP_plus_uconst)
//                                  block_len = 7;
//                          }
//                      }
//#endif
                        reg_location.SetIsDWARFExpression(block_data, block_len);
                        row.SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_val_offset          :   // 0x14
                case DW_CFA_val_offset_sf       :   // 0x15
                default:
                    tmp_uval32 = extended_opcode;
                    break;
            }
        }
    }
    fde->AppendRow(row);
}

void
DWARFCallFrameInfo::ParseAll()
{
    Index();
    fde_map_t::iterator pos, end = m_fde_map.end();
    for (pos = m_fde_map.begin(); pos != end; ++ pos)
    {
        if (pos->second.fde_sp.get() == NULL)
            pos->second.fde_sp = ParseFDE(pos->second.fde_offset);
    }
}


//bool
//DWARFCallFrameInfo::UnwindRegisterAtIndex
//(
//  const uint32_t reg_idx,
//  const Thread* currState,
//  const DWARFCallFrameInfo::Row* row,
//  mapped_memory_t * memCache,
//  Thread* unwindState
//)
//{
//    bool get_reg_success = false;
//
//    const RegLocation* regLocation = row->regs.GetRegisterInfo(reg_idx);
//
//  // On some systems, we may not get unwind info for the program counter,
//  // but the return address register can be used to get that information.
//    if (reg_idx == currState->GetPCRegNum(Thread::Index))
//    {
//      const RegLocation* returnAddrRegLocation = row->regs.GetRegisterInfo(currState->GetRARegNum(Thread::Index));
//      if (regLocation == NULL)
//      {
//          // We have nothing to the program counter, so lets see if this
//          // thread state has a return address (link register) that can
//          // help us track down the previous PC
//          regLocation = returnAddrRegLocation;
//      }
//      else if (regLocation->type == RegLocation::unspecified)
//      {
//          // We did have a location that didn't specify a value for unwinding
//          // the PC, so if there is a info for the return return address
//          // register (link register) lets use that
//          if (returnAddrRegLocation)
//              regLocation = returnAddrRegLocation;
//      }
//    }
//
//    if (regLocation)
//    {
//      mach_vm_address_t unwoundRegValue = INVALID_VMADDR;
//      switch (regLocation->type)
//      {
//          case RegLocation::undefined:
//              // Register is not available, mark it as invalid
//              unwindState->SetRegisterIsValid(reg_idx, Thread::Index, false);
//              return true;
//
//          case RegLocation::unspecified:
//              // Nothing to do if it is the same
//              return true;
//
//          case RegLocation::same:
//              // Nothing to do if it is the same
//              return true;
//
//          case RegLocation::atFPPlusOffset:
//          case RegLocation::isFPPlusOffset:
//          {
//              uint64_t unwindAddress = currState->GetRegisterValue(row->cfa_register, Thread::GCC, INVALID_VMADDR, &get_reg_success);
//
//              if (get_reg_success)
//              {
//                  unwindAddress += row->cfa_offset + regLocation->location.offset;
//
//                  if (regLocation->type == RegLocation::isFPPlusOffset)
//                  {
//                      unwindState->SetRegisterValue(reg_idx, Thread::Index, unwindAddress);
//                      return true;
//                  }
//                  else
//                  {
//                      kern_return_t err = mapped_memory_read_pointer(memCache, unwindAddress, &unwoundRegValue);
//                      if (err != KERN_SUCCESS)
//                      {
//                          unwindState->SetRegisterIsValid(reg_idx, Thread::Index, false);
//                          return false;
//                      }
//                      unwindState->SetRegisterValue(reg_idx, Thread::Index, unwoundRegValue);
//                      return true;
//                  }
//              }
//              else
//              {
//                  unwindState->SetRegisterIsValid(reg_idx, Thread::Index, false);
//              }
//              return false;
//          }
//              break;
//
//          case RegLocation::atDWARFExpression:
//          case RegLocation::isDWARFExpression:
//          {
//              bool swap = false;
//              DWARFExpressionBaton baton = { currState, memCache, swap };
//              uint64_t expr_result = 0;
//              CSBinaryDataRef opcodes(regLocation->location.expr.opcodes, regLocation->location.expr.length, swap);
//              opcodes.SetPointerSize(currState->Is64Bit() ? 8 : 4);
//              const char * expr_err = CSDWARFExpression::Evaluate(DWARFExpressionReadMemoryDCScriptInterpreter::Type,
//                                                                  DWARFExpressionReadRegisterDCScriptInterpreter::Type,
//                                                                  &baton,
//                                                                  opcodes,
//                                                                  0,
//                                                                  regLocation->location.expr.length,
//                                                                  NULL,
//                                                                  expr_result);
//              if (expr_err == NULL)
//              {
//                  // SUCCESS!
//                  if (regLocation->type == RegLocation::isDWARFExpression)
//                  {
//                      unwindState->SetRegisterValue(reg_idx, Thread::Index, expr_result);
//                      return true;
//                  }
//                  else
//                  {
//                      kern_return_t err = mapped_memory_read_pointer(memCache, expr_result, &unwoundRegValue);
//                      if (err != KERN_SUCCESS)
//                      {
//                          unwindState->SetRegisterIsValid(reg_idx, Thread::Index, false);
//                          return false;
//                      }
//                      unwindState->SetRegisterValue(reg_idx, Thread::Index, unwoundRegValue);
//                      return true;
//                  }
//              }
//              else
//              {
//                  // FAIL
//                  unwindState->SetRegisterIsValid(reg_idx, Thread::Index, false);
//              }
//              return false;
//          }
//              break;
//
//
//          case RegLocation::inRegister:
//              // The value is in another register.
//              unwoundRegValue = currState->GetRegisterValue(regLocation->location.reg, Thread::GCC, 0, &get_reg_success);
//              if (get_reg_success)
//              {
//                  unwindState->SetRegisterValue(reg_idx, Thread::Index, unwoundRegValue);
//                  return true;
//              }
//              return false;
//
//          default:
//              break;
//      }
//    }
//
//    if (reg_idx == currState->GetSPRegNum(Thread::Index))
//    {
//      uint64_t cfa = currState->GetRegisterValue(row->cfa_register, Thread::GCC, 0, &get_reg_success);
//      if (get_reg_success)
//      {
//          return unwindState->SetSP(cfa + row->cfa_offset);
//      }
//      else
//      {
//          unwindState->SetRegisterIsValid(reg_idx, Thread::Index, false);
//          return false;
//      }
//    }
//
//    return false;
//}

void
DWARFCallFrameInfo::Dump(Stream *s, Thread *thread) const
{
    s->Indent();
    s->Printf("DWARFCallFrameInfo for ");
    *s << m_objfile->GetFileSpec();
    if (m_flags.IsSet(eFlagParsedIndex))
    {
        s->Printf(" (CIE[%zu], FDE[%zu])\n", m_cie_map.size(), m_fde_map.size());
        s->IndentMore();
        cie_map_t::const_iterator cie_pos, cie_end = m_cie_map.end();
        const ArchSpec *arch = &m_objfile->GetModule()->GetArchitecture();

        for (cie_pos = m_cie_map.begin(); cie_pos != cie_end; ++ cie_pos)
        {
            if (cie_pos->second.get() == NULL)
            {
                s->Indent();
                s->Printf("CIE{0x%8.8x} - unparsed\n", cie_pos->first);
            }
            else
            {
                cie_pos->second->Dump(s, thread, arch, m_reg_kind);
            }
        }

        fde_map_t::const_iterator fde_pos, fde_end = m_fde_map.end();
        for (fde_pos = m_fde_map.begin(); fde_pos != fde_end; ++ fde_pos)
        {
            if (fde_pos->second.fde_sp.get() == NULL)
            {
                s->Indent();
                s->Printf("FDE{0x%8.8x} - unparsed\n", fde_pos->second.fde_offset);
            }
            else
            {
                fde_pos->second.fde_sp->Dump(s, *this, thread);
            }
        }
        s->IndentLess();
    }
    else
    {
        s->PutCString(" (not indexed yet)\n");
    }
}


//uint32_t
//DWARFCallFrameInfo::UnwindThreadState(const Thread* currState, mapped_memory_t *memCache, bool is_first_frame, Thread* unwindState)
//{
//  if (currState == NULL || unwindState == NULL)
//      return 0;
//
//    *unwindState = *currState;
//    uint32_t numRegisterUnwound = 0;
//    uint64_t currPC = currState->GetPC(INVALID_VMADDR);
//
//    if (currPC != INVALID_VMADDR)
//    {
//      // If this is not the first frame, we care about the previous instruction
//      // since it will be at the instruction following the instruction that
//      // made the function call.
//      uint64_t unwindPC = currPC;
//      if (unwindPC > 0 && !is_first_frame)
//          --unwindPC;
//
//#if defined(__i386__) || defined(__x86_64__)
//      // Only on i386 do we have __IMPORT segments that contain trampolines
//      if (!currState->Is64Bit() && ImportRangesContainsAddress(unwindPC))
//      {
//          uint64_t curr_sp = currState->GetSP(INVALID_VMADDR);
//          mach_vm_address_t pc = INVALID_VMADDR;
//          unwindState->SetSP(curr_sp + 4);
//          kern_return_t err = mapped_memory_read_pointer(memCache, curr_sp, &pc);
//          if (err == KERN_SUCCESS)
//          {
//              unwindState->SetPC(pc);
//              return 2;
//          }
//      }
//#endif
//      FDE *fde = FindFDE(unwindPC);
//      if (fde)
//      {
//          FindRowUserData rowUserData (currState, unwindPC);
//          ParseInstructions (currState, fde, FindRowForAddress, &rowUserData);
//
//          const uint32_t numRegs = currState->NumRegisters();
//          for (uint32_t regNum = 0; regNum < numRegs; regNum++)
//          {
//              if (UnwindRegisterAtIndex(regNum, currState, &rowUserData.state, memCache, unwindState))
//                  numRegisterUnwound++;
//          }
//      }
//    }
//    return numRegisterUnwound;
//}


