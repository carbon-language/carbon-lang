#ifndef liblldb_UnwindPlan_h
#define liblldb_UnwindPlan_h

#include "lldb/lldb-private.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ConstString.h"

#include <map>
#include <vector>

namespace lldb_private {

// The UnwindPlan object specifies how to unwind out of a function - where
// this function saves the caller's register values before modifying them
// (for non-volatile aka saved registers) and how to find this frame's
// Canonical Frame Address (CFA).

// Most commonly, registers are saved on the stack, offset some bytes from
// the Canonical Frame Address, or CFA, which is the starting address of
// this function's stack frame (the CFA is same as the eh_frame's CFA,
// whatever that may be on a given architecture).
// The CFA address for the stack frame does not change during
// the lifetime of the function.

// Internally, the UnwindPlan is structured as a vector of register locations
// organized by code address in the function, showing which registers have been
// saved at that point and where they are saved.  
// It can be thought of as the expanded table form of the DWARF CFI 
// encoded information.

// Other unwind information sources will be converted into UnwindPlans before
// being added to a FuncUnwinders object.  The unwind source may be 
// an eh_frame FDE, a DWARF debug_frame FDE, or assembly language based 
// prologue analysis.
// The UnwindPlan is the canonical form of this information that the unwinder
// code will use when walking the stack.

class UnwindPlan {
public:

    class Row {
    public:
        class RegisterLocation
        {
        public:
    
            enum RestoreType
                {
                    unspecified,        // not specified, we may be able to assume this 
                                        // is the same register. gcc doesn't specify all 
                                        // initial values so we really don't know...
                    isUndefined,        // reg is not available, e.g. volatile reg
                    isSame,             // reg is unchanged
                    atCFAPlusOffset,    // reg = deref(CFA + offset)
                    isCFAPlusOffset,    // reg = CFA + offset
                    inOtherRegister,    // reg = other reg
                    atDWARFExpression,  // reg = deref(eval(dwarf_expr))
                    isDWARFExpression   // reg = eval(dwarf_expr)
                };
    
            RegisterLocation() : m_type(unspecified), m_location() { }
    
            bool
            operator == (const RegisterLocation& rhs) const;
    
            void SetUnspecified();

            void SetUndefined();
    
            void SetSame();
    
            bool IsSame () const { return m_type == isSame; }

            bool IsUnspecified () const { return m_type == unspecified; }

            bool IsCFAPlusOffset () const { return m_type == isCFAPlusOffset; }

            bool IsAtCFAPlusOffset () const { return m_type == atCFAPlusOffset; }

            bool IsInOtherRegister () const { return m_type == inOtherRegister; }

            bool IsAtDWARFExpression () const { return m_type == atDWARFExpression; }

            bool IsDWARFExpression () const { return m_type == isDWARFExpression; }

            void SetAtCFAPlusOffset (int32_t offset);
    
            void SetIsCFAPlusOffset (int32_t offset);
    
            void SetInRegister (uint32_t reg_num);
    
            uint32_t GetRegisterNumber () const { return m_location.reg_num; }

            RestoreType GetLocationType () const { return m_type; }

            int32_t GetOffset () const { return m_location.offset; }
            
            void GetDWARFExpr (const uint8_t **opcodes, uint16_t& len) const { *opcodes = m_location.expr.opcodes; len = m_location.expr.length; }

            void
            SetAtDWARFExpression (const uint8_t *opcodes, uint32_t len);
    
            void
            SetIsDWARFExpression (const uint8_t *opcodes, uint32_t len);

            const uint8_t *
            GetDWARFExpressionBytes () { return m_location.expr.opcodes; }

            int
            GetDWARFExpressionLength () { return m_location.expr.length; }

            void
            Dump (Stream &s) const;

        private:
            RestoreType m_type;            // How do we locate this register?
            union
            {
                // For m_type == atCFAPlusOffset or m_type == isCFAPlusOffset
                int32_t offset;
                // For m_type == inOtherRegister
                uint32_t reg_num; // The register number
                // For m_type == atDWARFExpression or m_type == isDWARFExpression
                struct {
                    const uint8_t *opcodes;
                    uint16_t length;
                } expr;
            } m_location;
        };
    
    public:
        Row ();
    
        bool
        GetRegisterInfo (uint32_t reg_num, RegisterLocation& register_location) const;
    
        void
        SetRegisterInfo (uint32_t reg_num, const RegisterLocation register_location);
    
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
        SlideOffset(lldb::addr_t offset)
        {
            m_offset += offset;
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
    
        // Return the number of registers we have locations for
        int
        GetRegisterCount () const
        {
            return m_register_locations.size();
        }

        void
        Clear ();

        void
        Dump (Stream& s, int register_kind, Thread* thread) const;

    protected:
        typedef std::map<uint32_t, RegisterLocation> collection;
        lldb::addr_t m_offset;      // Offset into the function for this row
        uint32_t m_cfa_reg_num;     // The Call Frame Address register number
        int32_t  m_cfa_offset;      // The offset from the CFA for this row
        collection m_register_locations;

    }; // class Row

public:

    UnwindPlan () : m_register_kind(-1), m_row_list(), m_plan_valid_address_range(), m_source_name()
    { 
        m_plan_valid_address_range.SetByteSize (0);
    }

    void Dump (Stream& s, Thread* thread) const;

    void 
    AppendRow (const Row& row);

    // Returns a pointer to the best row for the given offset into the function's instructions.
    // If offset is -1 it indicates that the function start is unknown - the final row in the UnwindPlan is returned.
    // In practice, the UnwindPlan for a function with no known start address will be the architectural default
    // UnwindPlan which will only have one row.
    const Row*
    GetRowForFunctionOffset (int offset) const;

    void
    SetRegisterKind (uint32_t rk);

    uint32_t
    GetRegisterKind (void) const;

    // This UnwindPlan may not be valid at every address of the function span.  
    // For instance, a FastUnwindPlan will not be valid at the prologue setup 
    // instructions - only in the body of the function.
    void
    SetPlanValidAddressRange (const AddressRange& range);

    bool
    PlanValidAtAddress (Address addr);

    bool
    IsValidRowIndex (uint32_t idx) const;

    const UnwindPlan::Row&
    GetRowAtIndex (uint32_t idx) const;

    lldb_private::ConstString
    GetSourceName () const;

    void
    SetSourceName (const char *);

    int
    GetRowCount () const;

private:

    typedef std::vector<Row> collection;
    collection m_row_list;
    AddressRange m_plan_valid_address_range;
    uint32_t m_register_kind;   // The RegisterKind these register numbers are in terms of - will need to be
                                // translated to lldb native reg nums at unwind time
    lldb_private::ConstString m_source_name;  // for logging, where this UnwindPlan originated from
}; // class UnwindPlan

} // namespace lldb_private

#endif //liblldb_UnwindPlan_h
