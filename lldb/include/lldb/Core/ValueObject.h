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
#include "lldb/Utility/SharedCluster.h"

namespace lldb_private {

/// ValueObject:
///
/// This abstract class provides an interface to a particular value, be it a register, a local or global variable,
/// that is evaluated in some particular scope.  The ValueObject also has the capibility of being the "child" of
/// some other variable object, and in turn of having children.  
/// If a ValueObject is a root variable object - having no parent - then it must be constructed with respect to some
/// particular ExecutionContextScope.  If it is a child, it inherits the ExecutionContextScope from its parent.
/// The ValueObject will update itself if necessary before fetching its value, summary, object description, etc.
/// But it will always update itself in the ExecutionContextScope with which it was originally created.

/// A brief note on life cycle management for ValueObjects.  This is a little tricky because a ValueObject can contain
/// various other ValueObjects - the Dynamic Value, its children, the dereference value, etc.  Any one of these can be
/// handed out as a shared pointer, but for that contained value object to be valid, the root object and potentially other
/// of the value objects need to stay around.  
/// We solve this problem by handing out shared pointers to the Value Object and any of its dependents using a shared
/// ClusterManager.  This treats each shared pointer handed out for the entire cluster as a reference to the whole
/// cluster.  The whole cluster will stay around until the last reference is released.
///
/// The ValueObject mostly handle this automatically, if a value object is made with a Parent ValueObject, then it adds
/// itself to the ClusterManager of the parent.

/// It does mean that external to the ValueObjects we should only ever make available ValueObjectSP's, never ValueObjects 
/// or pointers to them.  So all the "Root level" ValueObject derived constructors should be private, and 
/// should implement a Create function that new's up object and returns a Shared Pointer that it gets from the GetSP() method.
///
/// However, if you are making an derived ValueObject that will be contained in a parent value object, you should just
/// hold onto a pointer to it internally, and by virtue of passing the parent ValueObject into its constructor, it will
/// be added to the ClusterManager for the parent.  Then if you ever hand out a Shared Pointer to the contained ValueObject,
/// just do so by calling GetSP() on the contained object.

class ValueObject : public UserID
{
public:
    
    enum GetExpressionPathFormat
    {
        eDereferencePointers = 1,
        eHonorPointers,
    };
    
    enum ValueObjectRepresentationStyle
    {
        eDisplayValue = 1,
        eDisplaySummary,
        eDisplayLanguageSpecific
    };
    
    enum ExpressionPathScanEndReason
    {
        eEndOfString = 1,           // out of data to parse
        eNoSuchChild,               // child element not found
        eEmptyRangeNotAllowed,      // [] only allowed for arrays
        eDotInsteadOfArrow,         // . used when -> should be used
        eArrowInsteadOfDot,         // -> used when . should be used
        eFragileIVarNotAllowed,     // ObjC ivar expansion not allowed
        eRangeOperatorNotAllowed,   // [] not allowed by options
        eRangeOperatorInvalid,      // [] not valid on objects other than scalars, pointers or arrays
        eArrayRangeOperatorMet,     // [] is good for arrays, but I cannot parse it
        eBitfieldRangeOperatorMet,  // [] is good for bitfields, but I cannot parse after it
        eUnexpectedSymbol,          // something is malformed in the expression
        eTakingAddressFailed,       // impossible to apply & operator
        eDereferencingFailed,       // impossible to apply * operator
        eUnknown = 0xFFFF
    };
    
    enum ExpressionPathEndResultType
    {
        ePlain = 1,                 // anything but...
        eBitfield,                  // a bitfield
        eBoundedRange,              // a range [low-high]
        eUnboundedRange,            // a range []
        eInvalid = 0xFFFF
    };
    
    enum ExpressionPathAftermath
    {
        eNothing = 1,               // just return it
        eDereference,               // dereference the target
        eTakeAddress                // take target's address
    };
    
    struct GetValueForExpressionPathOptions
    {
        bool m_check_dot_vs_arrow_syntax;
        bool m_no_fragile_ivar;
        bool m_allow_bitfields_syntax;
        
        GetValueForExpressionPathOptions(bool dot = false,
                                         bool no_ivar = false,
                                         bool bitfield = true) :
            m_check_dot_vs_arrow_syntax(dot),
            m_no_fragile_ivar(no_ivar),
            m_allow_bitfields_syntax(bitfield)
        {
        }
        
        GetValueForExpressionPathOptions&
        DoCheckDotVsArrowSyntax()
        {
            m_check_dot_vs_arrow_syntax = true;
            return *this;
        }
        
        GetValueForExpressionPathOptions&
        DontCheckDotVsArrowSyntax()
        {
            m_check_dot_vs_arrow_syntax = false;
            return *this;
        }
        
        GetValueForExpressionPathOptions&
        DoAllowFragileIVar()
        {
            m_no_fragile_ivar = false;
            return *this;
        }
        
        GetValueForExpressionPathOptions&
        DontAllowFragileIVar()
        {
            m_no_fragile_ivar = true;
            return *this;
        }

        GetValueForExpressionPathOptions&
        DoAllowBitfieldSyntax()
        {
            m_allow_bitfields_syntax = true;
            return *this;
        }
        
        GetValueForExpressionPathOptions&
        DontAllowBitfieldSyntax()
        {
            m_allow_bitfields_syntax = false;
            return *this;
        }
        
        static const GetValueForExpressionPathOptions
        DefaultOptions()
        {
            static GetValueForExpressionPathOptions g_default_options;
            
            return g_default_options;
        }

    };

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
        SetUpdated ();
        
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
    IsScalarType ();

    virtual bool
    IsPointerOrReferenceType ();
    
    virtual bool
    IsPossibleCPlusPlusDynamicType ();
    
    virtual bool
    IsPossibleDynamicType ();

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
    GetExpressionPath (Stream &s, bool qualify_cxx_base_classes, GetExpressionPathFormat = eDereferencePointers);
    
    lldb::ValueObjectSP
    GetValueForExpressionPath(const char* expression,
                              const char** first_unparsed = NULL,
                              ExpressionPathScanEndReason* reason_to_stop = NULL,
                              ExpressionPathEndResultType* final_value_type = NULL,
                              const GetValueForExpressionPathOptions& options = GetValueForExpressionPathOptions::DefaultOptions(),
                              ExpressionPathAftermath* final_task_on_target = NULL);
    
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
    IsArrayItemForPointer()
    {
        return m_is_array_item_for_pointer;
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

    // Return the module associated with this value object in case the
    // value is from an executable file and might have its data in
    // sections of the file. This can be used for variables.
    virtual Module *
    GetModule()
    {
        if (m_parent)
            return m_parent->GetModule();
        return NULL;
    }
    //------------------------------------------------------------------
    // The functions below should NOT be modified by sublasses
    //------------------------------------------------------------------
    const Error &
    GetError();

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
    
    const char *
    GetPrintableRepresentation(ValueObjectRepresentationStyle val_obj_display = eDisplaySummary,
                               lldb::Format custom_format = lldb::eFormatInvalid);

    bool
    DumpPrintableRepresentation(Stream& s,
                                ValueObjectRepresentationStyle val_obj_display = eDisplaySummary,
                                lldb::Format custom_format = lldb::eFormatInvalid);
    bool
    GetValueIsValid () const;

    bool
    GetValueDidChange ();

    bool
    UpdateValueIfNeeded (bool update_format = true);
    
    void
    UpdateFormatsIfNeeded();

    DataExtractor &
    GetDataExtractor ();

    bool
    Write ();

    lldb::ValueObjectSP
    GetSP ()
    {
        return m_manager->GetSharedPointer(this);
    }
    
protected:
    void
    AddSyntheticChild (const ConstString &key,
                       ValueObject *valobj);
public:
    lldb::ValueObjectSP
    GetSyntheticChild (const ConstString &key) const;

    lldb::ValueObjectSP
    GetSyntheticArrayMemberFromPointer (int32_t index, bool can_create);
    
    lldb::ValueObjectSP
    GetSyntheticBitFieldChild (uint32_t from, uint32_t to, bool can_create);
    
    lldb::ValueObjectSP
    GetDynamicValue (lldb::DynamicValueType valueType);
    
    virtual lldb::ValueObjectSP
    CreateConstantValue (const ConstString &name);

    virtual lldb::ValueObjectSP
    Dereference (Error &error);
    
    virtual lldb::ValueObjectSP
    AddressOf (Error &error);

    virtual lldb::ValueObjectSP
    CastPointerType (const char *name,
                     ClangASTType &ast_type);

    virtual lldb::ValueObjectSP
    CastPointerType (const char *name,
                     lldb::TypeSP &type_sp);

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
                     lldb::DynamicValueType use_dynamic,
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
        if (m_parent && m_format == lldb::eFormatDefault)
            return m_parent->GetFormat();
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
    typedef ClusterManager<ValueObject> ValueObjectManager;

    //------------------------------------------------------------------
    // Classes that inherit from ValueObject can see and modify these
    //------------------------------------------------------------------
    ValueObject  *      m_parent;       // The parent value object, or NULL if this has no parent
    EvaluationPoint     m_update_point; // Stores both the stop id and the full context at which this value was last 
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

    ValueObjectManager *m_manager;      // This object is managed by the root object (any ValueObject that gets created
                                        // without a parent.)  The manager gets passed through all the generations of
                                        // dependent objects, and will keep the whole cluster of objects alive as long
                                        // as a shared pointer to any of them has been handed out.  Shared pointers to
                                        // value objects must always be made with the GetSP method.

    std::vector<ValueObject *> m_children;
    std::map<ConstString, ValueObject *> m_synthetic_children;
    ValueObject *m_dynamic_value;
    lldb::ValueObjectSP m_addr_of_valobj_sp; // We have to hold onto a shared pointer to this one because it is created
                                             // as an independent ValueObjectConstResult, which isn't managed by us.
    ValueObject *m_deref_valobj;

    lldb::Format        m_format;
    uint32_t            m_last_format_mgr_revision;
    lldb::SummaryFormatSP m_last_summary_format;
    lldb::ValueFormatSP m_last_value_format;
    bool                m_value_is_valid:1,
                        m_value_did_change:1,
                        m_children_count_valid:1,
                        m_old_value_valid:1,
                        m_pointers_point_to_load_addrs:1,
                        m_is_deref_of_parent:1,
                        m_is_array_item_for_pointer:1,
                        m_is_bitfield_for_scalar:1;
    
    friend class ClangExpressionDeclMap;  // For GetValue
    friend class ClangExpressionVariable; // For SetName
    friend class Target;                  // For SetName

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

    ValueObjectManager *
    GetManager()
    {
        return m_manager;
    }
    
    virtual bool
    UpdateValue () = 0;

    virtual void
    CalculateDynamicValue (lldb::DynamicValueType use_dynamic);
    
    // Should only be called by ValueObject::GetChildAtIndex()
    // Returns a ValueObject managed by this ValueObject's manager.
    virtual ValueObject *
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
    
    lldb::ValueObjectSP
    GetValueForExpressionPath_Impl(const char* expression,
                                   const char** first_unparsed,
                                   ExpressionPathScanEndReason* reason_to_stop,
                                   ExpressionPathEndResultType* final_value_type,
                                   const GetValueForExpressionPathOptions& options,
                                   ExpressionPathAftermath* final_task_on_target);
    
    DISALLOW_COPY_AND_ASSIGN (ValueObject);

};

} // namespace lldb_private

#endif  // liblldb_ValueObject_h_
