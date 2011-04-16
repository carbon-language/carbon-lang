//===-- ValueObject.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObject_h_
#define liblldb_ValueObject_h_

// C Includes
// C++ Includes
#include <map>
#include <vector>
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/Value.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/StackID.h"

namespace lldb_private {

/// ValueObject:
/// This abstract class provides an interface to a particular value, be it a register, a local or global variable,
/// that is evaluated in some particular scope.  The ValueObject also has the capibility of being the "child" of
/// some other variable object, and in turn of having children.  
/// If a ValueObject is a root variable object - having no parent - then it must be constructed with respect to some
/// particular ExecutionContextScope.  If it is a child, it inherits the ExecutionContextScope from its parent.
/// The ValueObject will update itself if necessary before fetching its value, summary, object description, etc.
/// But it will always update itself in the ExecutionContextScope with which it was originally created.
class ValueObject : public UserID
{
public:

    class EvaluationPoint 
    {
    public:
        
        EvaluationPoint ();
        
        EvaluationPoint (ExecutionContextScope *exe_scope, bool use_selected = false);
        
        EvaluationPoint (const EvaluationPoint &rhs);
        
        ~EvaluationPoint ();
        
        ExecutionContextScope *
        GetExecutionContextScope ();
        
        Target *
        GetTarget () const
        {
            return m_target_sp.get();
        }
        
        Process *
        GetProcess () const
        {
            return m_process_sp.get();
        }
                
        // Set the EvaluationPoint to the values in exe_scope,
        // Return true if the Evaluation Point changed.
        // Since the ExecutionContextScope is always going to be valid currently, 
        // the Updated Context will also always be valid.
        
        bool
        SetContext (ExecutionContextScope *exe_scope);
        
        void
        SetIsConstant ()
        {
            SetUpdated();
            m_stop_id = LLDB_INVALID_UID;
        }
        
        bool
        IsConstant () const
        {
            return m_stop_id == LLDB_INVALID_UID;
        }
        
        lldb::user_id_t
        GetUpdateID () const
        {
            return m_stop_id;
        }

        void
        SetUpdateID (lldb::user_id_t new_id)
        {
            m_stop_id = new_id;
        }
        
        bool
        IsFirstEvaluation () const
        {
            return m_first_update;
        }
        
        void
        SetNeedsUpdate ()
        {
            m_needs_update = true;
        }
        
        void
        SetUpdated ()
        {
            m_first_update = false;
            m_needs_update = false;
        }
        
        bool
        NeedsUpdating()
        {
            SyncWithProcessState();
            return m_needs_update;
        }
        
        bool
        IsValid ()
        {
            if (m_stop_id == LLDB_INVALID_UID)
                return false;
            else if (SyncWithProcessState ())
            {
                if (m_stop_id == LLDB_INVALID_UID)
                    return false;
            }
            return true;
        }
        
        void
        SetInvalid ()
        {
            // Use the stop id to mark us as invalid, leave the thread id and the stack id around for logging and
            // history purposes.
            m_stop_id = LLDB_INVALID_UID;
            
            // Can't update an invalid state.
            m_needs_update = false;
            
//            m_thread_id = LLDB_INVALID_THREAD_ID;
//            m_stack_id.Clear();
        }
        
    private:
        bool
        SyncWithProcessState ();
                
        ExecutionContextScope *m_exe_scope;   // This is not the way to store the evaluation point state, it is just
                                            // a cache of the lookup, and gets thrown away when we update.
        bool             m_needs_update;
        bool             m_first_update;

        lldb::TargetSP   m_target_sp;
        lldb::ProcessSP  m_process_sp;
        lldb::user_id_t  m_thread_id;
        StackID          m_stack_id;
        lldb::user_id_t  m_stop_id; // This is the stop id when this ValueObject was last evaluated.
    };

    const EvaluationPoint &
    GetUpdatePoint () const
    {
        return m_update_point;
    }
    
    EvaluationPoint &
    GetUpdatePoint ()
    {
        return m_update_point;
    }
    
    ExecutionContextScope *
    GetExecutionContextScope ()
    {
        return m_update_point.GetExecutionContextScope();
    }
    
    friend class ValueObjectList;

    virtual ~ValueObject();

    //------------------------------------------------------------------
    // Sublasses must implement the functions below.
    //------------------------------------------------------------------
    virtual size_t
    GetByteSize() = 0;

    virtual clang::ASTContext *
    GetClangAST () = 0;

    virtual lldb::clang_type_t
    GetClangType () = 0;

    virtual lldb::ValueType
    GetValueType() const = 0;

    virtual ConstString
    GetTypeName() = 0;

    virtual lldb::LanguageType
    GetObjectRuntimeLanguage();

    virtual bool
    IsPointerType ();

    virtual bool
    IsPointerOrReferenceType ();

    virtual bool
    IsBaseClass ()
    {
        return false;
    }
    
    virtual bool
    IsDereferenceOfParent ()
    {
        return false;
    }
    
    bool
    IsIntegerType (bool &is_signed);
    
    virtual bool
    GetBaseClassPath (Stream &s);

    virtual void
    GetExpressionPath (Stream &s, bool qualify_cxx_base_classes);

    virtual bool
    IsInScope ()
    {
        return true;
    }

    virtual off_t
    GetByteOffset()
    {
        return 0;
    }

    virtual uint32_t
    GetBitfieldBitSize()
    {
        return 0;
    }

    virtual uint32_t
    GetBitfieldBitOffset()
    {
        return 0;
    }
    
    virtual bool
    SetClangAST (clang::ASTContext *ast)
    {
        return false;
    }

    virtual const char *
    GetValueAsCString ();

    virtual bool
    SetValueFromCString (const char *value_str);

    //------------------------------------------------------------------
    // The functions below should NOT be modified by sublasses
    //------------------------------------------------------------------
    const Error &
    GetError() const;

    const ConstString &
    GetName() const;

    lldb::ValueObjectSP
    GetChildAtIndex (uint32_t idx, bool can_create);

    virtual lldb::ValueObjectSP
    GetChildMemberWithName (const ConstString &name, bool can_create);

    virtual uint32_t
    GetIndexOfChildWithName (const ConstString &name);

    uint32_t
    GetNumChildren ();

    const Value &
    GetValue() const;

    Value &
    GetValue();

    bool
    ResolveValue (Scalar &scalar);
    
    const char *
    GetLocationAsCString ();

    const char *
    GetSummaryAsCString ();
    
    const char *
    GetObjectDescription ();

    bool
    GetValueIsValid () const;

    bool
    GetValueDidChange ();

    bool
    UpdateValueIfNeeded ();

    DataExtractor &
    GetDataExtractor ();

    bool
    Write ();

    void
    AddSyntheticChild (const ConstString &key,
                       lldb::ValueObjectSP& valobj_sp);

    lldb::ValueObjectSP
    GetSyntheticChild (const ConstString &key) const;

    lldb::ValueObjectSP
    GetSyntheticArrayMemberFromPointer (int32_t index, bool can_create);
    
    lldb::ValueObjectSP
    GetDynamicValue (bool can_create);
    
    lldb::ValueObjectSP
    GetDynamicValue (bool can_create, lldb::ValueObjectSP &owning_valobj_sp);
    
    virtual lldb::ValueObjectSP
    CreateConstantValue (const ConstString &name);

    virtual lldb::ValueObjectSP
    Dereference (Error &error);
    
    virtual lldb::ValueObjectSP
    AddressOf (Error &error);

    // The backing bits of this value object were updated, clear any value
    // values, summaries or descriptions so we refetch them.
    virtual void
    ValueUpdated ()
    {
        m_value_str.clear();
        m_summary_str.clear();
        m_object_desc_str.clear();
    }

    virtual bool
    IsDynamic ()
    {
        return false;
    }

    static void
    DumpValueObject (Stream &s,
                     ValueObject *valobj,
                     const char *root_valobj_name,
                     uint32_t ptr_depth,
                     uint32_t curr_depth,
                     uint32_t max_depth,
                     bool show_types,
                     bool show_location,
                     bool use_objc,
                     bool use_dynamic,
                     bool scope_already_checked,
                     bool flat_output);

    bool
    GetIsConstant () const
    {
        return m_update_point.IsConstant();
    }
    
    void
    SetIsConstant ()
    {
        m_update_point.SetIsConstant();
    }

    lldb::Format
    GetFormat () const
    {
        return m_format;
    }
    
    void
    SetFormat (lldb::Format format)
    {
        if (format != m_format)
            m_value_str.clear();
        m_format = format;
    }

    // Use GetParent for display purposes, but if you want to tell the parent to update itself
    // then use m_parent.  The ValueObjectDynamicValue's parent is not the correct parent for
    // displaying, they are really siblings, so for display it needs to route through to its grandparent.
    virtual ValueObject *
    GetParent()
    {
        return m_parent;
    }

    virtual const ValueObject *
    GetParent() const
    {
        return m_parent;
    }

    ValueObject *
    GetNonBaseClassParent();

    void
    SetPointersPointToLoadAddrs (bool b)
    {
        m_pointers_point_to_load_addrs = b;
    }

protected:
    //------------------------------------------------------------------
    // Classes that inherit from ValueObject can see and modify these
    //------------------------------------------------------------------
    ValueObject  *m_parent;       // The parent value object, or NULL if this has no parent
    EvaluationPoint      m_update_point; // Stores both the stop id and the full context at which this value was last 
                                        // updated.  When we are asked to update the value object, we check whether
                                        // the context & stop id are the same before updating.
    ConstString         m_name;         // The name of this object
    DataExtractor       m_data;         // A data extractor that can be used to extract the value.
    Value               m_value;
    Error               m_error;        // An error object that can describe any errors that occur when updating values.
    std::string         m_value_str;    // Cached value string that will get cleared if/when the value is updated.
    std::string         m_old_value_str;// Cached old value string from the last time the value was gotten
    std::string         m_location_str; // Cached location string that will get cleared if/when the value is updated.
    std::string         m_summary_str;  // Cached summary string that will get cleared if/when the value is updated.
    std::string         m_object_desc_str; // Cached result of the "object printer".  This differs from the summary
                                              // in that the summary is consed up by us, the object_desc_string is builtin.
    std::vector<lldb::ValueObjectSP> m_children;
    std::map<ConstString, lldb::ValueObjectSP> m_synthetic_children;
    lldb::ValueObjectSP m_dynamic_value_sp;
    lldb::ValueObjectSP m_addr_of_valobj_sp; // These two shared pointers help root the ValueObject shared pointers that
    lldb::ValueObjectSP m_deref_valobj_sp;   // we hand out, so that we can use them in their dynamic types and ensure
                                             // they will last as long as this ValueObject...

    lldb::Format        m_format;
    bool                m_value_is_valid:1,
                        m_value_did_change:1,
                        m_children_count_valid:1,
                        m_old_value_valid:1,
                        m_pointers_point_to_load_addrs:1,
                        m_is_deref_of_parent:1;
    
    friend class CommandObjectExpression;
    friend class ClangExpressionVariable;
    friend class ClangExpressionDeclMap;  // For GetValue...
    friend class Target;
    friend class ValueObjectChild;
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    
    // Use the no-argument constructor to make a constant variable object (with no ExecutionContextScope.)
    
    ValueObject();
    
    // Use this constructor to create a "root variable object".  The ValueObject will be locked to this context
    // through-out its lifespan.
    
    ValueObject (ExecutionContextScope *exe_scope);
    
    // Use this constructor to create a ValueObject owned by another ValueObject.  It will inherit the ExecutionContext
    // of its parent.
    
    ValueObject (ValueObject &parent);

    virtual bool
    UpdateValue () = 0;

    virtual void
    CalculateDynamicValue ();
    
    // Should only be called by ValueObject::GetChildAtIndex()
    virtual lldb::ValueObjectSP
    CreateChildAtIndex (uint32_t idx, bool synthetic_array_member, int32_t synthetic_index);

    // Should only be called by ValueObject::GetNumChildren()
    virtual uint32_t
    CalculateNumChildren() = 0;

    void
    SetName (const char *name);

    void
    SetName (const ConstString &name);

    void
    SetNumChildren (uint32_t num_children);

    void
    SetValueDidChange (bool value_changed);

    void
    SetValueIsValid (bool valid);

public:
    lldb::addr_t
    GetPointerValue (AddressType &address_type, 
                     bool scalar_is_load_address);

    lldb::addr_t
    GetAddressOf (AddressType &address_type, 
                  bool scalar_is_load_address);
private:
    //------------------------------------------------------------------
    // For ValueObject only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ValueObject);

};

} // namespace lldb_private

#endif  // liblldb_ValueObject_h_
