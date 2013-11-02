//===-- Frame.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Frame_h_
#define liblldb_Frame_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/StackID.h"

namespace lldb_private {

/// @class Frame Frame.h "lldb/Target/Frame.h"
///
/// @brief This base class provides an interface to stack frames.
///
/// Frame is a pure virtual class, instances of the subclasses should be 
/// created depending on the type of frame -- a live frame on a thread's
/// stack, an inlined code frame, a historical frame.
///
/// Frames may have a Canonical Frame Address (CFA) or not.  A frame may
/// have a plain pc value or it may have a pc value + stop_id to indicate
/// a specific point in the debug session so the correct section load list
/// is used for symbolication.
///
/// Local variables may be available, or not.  A register context may be
/// available, or not.

class Frame :
    public ExecutionContextScope,
    public std::enable_shared_from_this<Frame>
{
public:
    enum ExpressionPathOption
    {
        eExpressionPathOptionCheckPtrVsMember       = (1u << 0),
        eExpressionPathOptionsNoFragileObjcIvar     = (1u << 1),
        eExpressionPathOptionsNoSyntheticChildren   = (1u << 2),
        eExpressionPathOptionsNoSyntheticArrayRange = (1u << 3),
        eExpressionPathOptionsAllowDirectIVarAccess = (1u << 4)
    };

    virtual
    ~Frame () {}

    virtual lldb::ThreadSP
    GetThread () const = 0;

    virtual StackID&
    GetStackID() = 0;

    //------------------------------------------------------------------
    /// Get an Address for the current pc value in this Frame.
    ///
    /// May not be the same as the actual PC value for inlined stack frames.
    ///
    /// @return
    ///   The Address object set to the current PC value.
    //------------------------------------------------------------------
    virtual const Address&
    GetFrameCodeAddress() = 0;

    //------------------------------------------------------------------
    /// Change the pc value for a given thread.
    ///
    /// Change the current pc value for the frame on this thread.
    ///
    /// @param[in] pc
    ///     The load address that the pc will be set to.
    ///
    /// @return
    ///     true if the pc was changed.  false if this failed -- possibly
    ///     because this frame is not a live Frame.
    //------------------------------------------------------------------
    virtual bool
    ChangePC (lldb::addr_t pc) = 0;

    //------------------------------------------------------------------
    /// Provide a SymbolContext for this Frame's current pc value.
    ///
    /// The Frame maintains this SymbolContext and adds additional information
    /// to it on an as-needed basis.  This helps to avoid different functions
    /// looking up symbolic information for a given pc value multple times.
    ///
    /// @params [in] resolve_scope
    ///   Flags from the SymbolContextItem enumerated type which specify what
    ///   type of symbol context is needed by this caller.
    ///
    /// @return
    ///   A SymbolContext reference which includes the types of information
    ///   requested by resolve_scope, if they are available.
    //------------------------------------------------------------------
    virtual const SymbolContext&
    GetSymbolContext (uint32_t resolve_scope) = 0;

    //------------------------------------------------------------------
    /// Return the Canonical Frame Address (DWARF term) for this frame.
    ///
    /// The CFA is typically the value of the stack pointer register before
    /// the call invocation is made.  It will not change during the lifetime
    /// of a stack frame.  It is often not the same thing as the frame pointer
    /// register value.
    ///
    /// Live Frames will always have a CFA but other types of frames may
    /// not be able to supply one.
    ///
    /// @param [out] value
    ///   The address of the CFA for this frame, if available.
    ///
    /// @param [out] error_ptr
    ///   If there is an error determining the CFA address, this may contain a
    ///   string explaining the failure.
    ///
    /// @return
    ///   Returns true if the CFA value was successfully set in value.  Some
    ///   frames may be unable to provide this value; they will return false.
    //------------------------------------------------------------------
    virtual bool
    GetFrameBaseValue(Scalar &value, Error *error_ptr) = 0;

    //------------------------------------------------------------------
    /// Get the current lexical scope block for this Frame, if possible.
    ///
    /// If debug information is available for this stack frame, return a
    /// pointer to the innermost lexical Block that the frame is currently
    /// executing.
    ///
    /// @return
    ///   A pointer to the current Block.  NULL is returned if this can
    ///   not be provided.
    //------------------------------------------------------------------
    virtual Block *
    GetFrameBlock () = 0;

    //------------------------------------------------------------------
    /// Get the RegisterContext for this frame, if possible.
    ///
    /// Returns a shared pointer to the RegisterContext for this stack frame.
    /// Only a live Frame object will be able to return a RegisterContext -
    /// callers must be prepared for an empty shared pointer being returned.
    ///
    /// Even a live Frame RegisterContext may not be able to provide all
    /// registers.  Only the currently executing frame (frame 0) can reliably
    /// provide every register in the register context.
    ///
    /// @return
    ///   The RegisterContext shared point for this frame.
    //------------------------------------------------------------------
    virtual lldb::RegisterContextSP
    GetRegisterContext () = 0;

    virtual const lldb::RegisterContextSP &
    GetRegisterContextSP () const = 0;

    //------------------------------------------------------------------
    /// Retrieve the list of variables that are in scope at this Frame's pc.
    ///
    /// A frame that is not live may return an empty VariableList for a given
    /// pc value even though variables would be available at this point if
    /// it were a live stack frame.
    ///
    /// @param[in] get_file_globals
    ///     Whether to also retrieve compilation-unit scoped variables
    ///     that are visisble to the entire compilation unit (e.g. file
    ///     static in C, globals that are homed in this CU).
    ///
    /// @return
    ///     A pointer to a list of variables.
    //------------------------------------------------------------------
    virtual VariableList *
    GetVariableList (bool get_file_globals) = 0;

    //------------------------------------------------------------------
    /// Retrieve the list of variables that are in scope at this Frame's pc.
    ///
    /// A frame that is not live may return an empty VariableListSP for a
    /// given pc value even though variables would be available at this point
    /// if it were a live stack frame.
    ///
    /// @param[in] get_file_globals
    ///     Whether to also retrieve compilation-unit scoped variables
    ///     that are visisble to the entire compilation unit (e.g. file
    ///     static in C, globals that are homed in this CU).
    ///
    /// @return
    ///     A pointer to a list of variables.
    //------------------------------------------------------------------
    virtual lldb::VariableListSP
    GetInScopeVariableList (bool get_file_globals) = 0;

    //------------------------------------------------------------------
    /// Create a ValueObject for a variable name / pathname, possibly
    /// including simple dereference/child selection syntax.
    ///
    /// @param[in] var_expr
    ///     The string specifying a variable to base the VariableObject off
    ///     of.
    ///
    /// @param[in] use_dynamic
    ///     Whether the correct dynamic type of an object pointer should be
    ///     determined before creating the object, or if the static type is
    ///     sufficient.  One of the DynamicValueType enumerated values.
    ///
    /// @param[in] options
    ///     An unsigned integer of flags, values from Frame::ExpressionPathOption
    ///     enum.
    /// @param[in] var_sp
    ///     A VariableSP that will be set to the variable described in the
    ///     var_expr path.
    ///
    /// @param[in] error
    ///     Record any errors encountered while evaluating var_expr.
    ///
    /// @return
    ///     A shared pointer to the ValueObject described by var_expr.
    //------------------------------------------------------------------
    virtual lldb::ValueObjectSP
    GetValueForVariableExpressionPath (const char *var_expr,
                                       lldb::DynamicValueType use_dynamic,
                                       uint32_t options,
                                       lldb::VariableSP &var_sp,
                                       Error &error) = 0;

    //------------------------------------------------------------------
    /// Determine whether this Frame has debug information available or not
    ///
    /// @return
    //    true if debug information is available for this frame (function,
    //    compilation unit, block, etc.)
    //------------------------------------------------------------------
    virtual bool
    HasDebugInformation () = 0;

    //------------------------------------------------------------------
    /// Return the disassembly for the instructions of this Frame's function
    /// as a single C string.
    ///
    /// @return
    //    C string with the assembly instructions for this function.
    //------------------------------------------------------------------
    virtual const char *
    Disassemble () = 0;

    //------------------------------------------------------------------
    /// Print a description for this frame using the frame-format formatter settings.
    ///
    /// @param [in] strm
    ///   The Stream to print the description to.
    ///
    /// @param [in] frame_marker
    ///   Optional string that will be prepended to the frame output description.
    //------------------------------------------------------------------
    virtual void
    DumpUsingSettingsFormat (Stream *strm, const char *frame_marker = NULL) = 0;

    //------------------------------------------------------------------
    /// Print a description for this frame using a default format.
    ///
    /// @param [in] strm
    ///   The Stream to print the description to.
    ///
    /// @param [in] show_frame_index
    ///   Whether to print the frame number or not.
    ///
    /// @param [in] show_fullpaths
    ///   Whether to print the full source paths or just the file base name.
    //------------------------------------------------------------------
    virtual void
    Dump (Stream *strm, bool show_frame_index, bool show_fullpaths) = 0;

    //------------------------------------------------------------------
    /// Print a description of this stack frame and/or the source context/assembly
    /// for this stack frame.
    ///
    /// @param[in] strm
    ///   The Stream to send the output to.
    ///
    /// @param[in] show_frame_info
    ///   If true, print the frame info by calling DumpUsingSettingsFormat().
    ///
    /// @param[in] show_source
    ///   If true, print source or disassembly as per the user's settings.
    ///
    /// @param[in] frame_marker 
    ///   Passed to DumpUsingSettingsFormat() for the frame info printing.
    ///
    /// @return
    ///   Returns true if successful.
    //------------------------------------------------------------------
    virtual bool
    GetStatus (Stream &strm,
               bool show_frame_info,
               bool show_source,
               const char *frame_marker = NULL) = 0;

    //------------------------------------------------------------------
    /// Query whether this frame is a concrete frame on the call stack,
    /// or if it is an inlined frame derived from the debug information
    /// and presented by the debugger.
    ///
    /// @return
    ///   true if this is an inlined frame.
    //------------------------------------------------------------------
    virtual bool
    IsInlined () = 0;

    //------------------------------------------------------------------
    /// Query this frame to find what frame it is in this Thread's StackFrameList.
    ///
    /// @return
    ///   Frame index 0 indicates the currently-executing function.  Inline
    ///   frames are included in this frame index count.
    //------------------------------------------------------------------
    virtual uint32_t
    GetFrameIndex () const = 0;

    //------------------------------------------------------------------
    /// Query this frame to find what frame it is in this Thread's StackFrameList,
    /// not counting inlined frames.
    ///
    /// @return
    ///   Frame index 0 indicates the currently-executing function.  Inline
    ///   frames are not included in this frame index count; their concrete
    ///   frame index will be the same as the concrete frame that they are
    ///   derived from.
    //------------------------------------------------------------------
    virtual uint32_t
    GetConcreteFrameIndex () const = 0;

    //------------------------------------------------------------------
    /// Create a ValueObject for a given Variable in this Frame.
    ///
    /// @params [in] variable_sp
    ///   The Variable to base this ValueObject on
    ///
    /// @params [in] use_dynamic
    ///     Whether the correct dynamic type of the variable should be
    ///     determined before creating the ValueObject, or if the static type
    ///     is sufficient.  One of the DynamicValueType enumerated values.
    ///
    /// @return
    //    A ValueObject for this variable.
    //------------------------------------------------------------------
    virtual lldb::ValueObjectSP
    GetValueObjectForFrameVariable (const lldb::VariableSP &variable_sp, lldb::DynamicValueType use_dynamic) = 0;

    //------------------------------------------------------------------
    /// Add an arbitrary Variable object (e.g. one that specifics a global or static)
    /// to a Frame's list of ValueObjects.
    ///
    /// @params [in] variable_sp
    ///   The Variable to base this ValueObject on
    ///
    /// @params [in] use_dynamic
    ///     Whether the correct dynamic type of the variable should be
    ///     determined before creating the ValueObject, or if the static type
    ///     is sufficient.  One of the DynamicValueType enumerated values.
    ///
    /// @return
    //    A ValueObject for this variable.
    //------------------------------------------------------------------
    virtual lldb::ValueObjectSP
    TrackGlobalVariable (const lldb::VariableSP &variable_sp, lldb::DynamicValueType use_dynamic) = 0;

    //------------------------------------------------------------------
    // lldb::ExecutionContextScope pure virtual functions
    //------------------------------------------------------------------
    virtual lldb::TargetSP
    CalculateTarget () = 0;

    virtual lldb::ProcessSP
    CalculateProcess () = 0;

    virtual lldb::ThreadSP
    CalculateThread () = 0;

    virtual lldb::FrameSP
    CalculateFrame () = 0;

    virtual void
    CalculateExecutionContext (ExecutionContext &exe_ctx) = 0;

protected:
    friend class StackFrameList;
    friend class StackFrame;

    virtual void
    SetSymbolContextScope (SymbolContextScope *symbol_scope) = 0;

    virtual void
    UpdateCurrentFrameFromPreviousFrame (Frame &prev_frame) = 0;

    virtual void
    UpdatePreviousFrameFromCurrentFrame (Frame &curr_frame) = 0;

    virtual const char *
    GetFrameType () 
    { 
        return "Frame"; 
    }
};

} // namespace lldb_private

#endif  // liblldb_Frame_h_
