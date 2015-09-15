//===-- UserExpression.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_UserExpression_h_
#define liblldb_UserExpression_h_

// C Includes
// C++ Includes
#include <string>
#include <map>
#include <vector>

// Other libraries and framework includes

#include "llvm/ADT/ArrayRef.h"

// Project includes

#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Expression/ClangExpressionHelper.h"
#include "lldb/Expression/Expression.h"
#include "Plugins/ExpressionParser/Clang/ClangExpressionVariable.h"
#include "lldb/Expression/IRForTarget.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Symbol/TaggedASTType.h"
#include "lldb/Target/ExecutionContext.h"

namespace lldb_private
{

//----------------------------------------------------------------------
/// @class UserExpression UserExpression.h "lldb/Expression/UserExpression.h"
/// @brief Encapsulates a one-time expression for use in lldb.
///
/// LLDB uses expressions for various purposes, notably to call functions
/// and as a backend for the expr command.  UserExpression is a virtual base
/// class that encapsulates the objects needed to parse and interpret or
/// JIT an expression.  The actual parsing part will be provided by the specific
/// implementations of UserExpression - which will be vended through the
/// appropriate TypeSystem.
//----------------------------------------------------------------------
class UserExpression : public Expression
{
public:

    enum { kDefaultTimeout = 500000u };
    //------------------------------------------------------------------
    /// Constructor
    ///
    /// @param[in] expr
    ///     The expression to parse.
    ///
    /// @param[in] expr_prefix
    ///     If non-NULL, a C string containing translation-unit level
    ///     definitions to be included when the expression is parsed.
    ///
    /// @param[in] language
    ///     If not eLanguageTypeUnknown, a language to use when parsing
    ///     the expression.  Currently restricted to those languages
    ///     supported by Clang.
    ///
    /// @param[in] desired_type
    ///     If not eResultTypeAny, the type to use for the expression
    ///     result.
    //------------------------------------------------------------------
    UserExpression (ExecutionContextScope &exe_scope,
                    const char *expr,
                    const char *expr_prefix,
                    lldb::LanguageType language,
                    ResultType desired_type);

    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    ~UserExpression() override;

    //------------------------------------------------------------------
    /// Parse the expression
    ///
    /// @param[in] error_stream
    ///     A stream to print parse errors and warnings to.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use when looking up entities that
    ///     are needed for parsing (locations of functions, types of
    ///     variables, persistent variables, etc.)
    ///
    /// @param[in] execution_policy
    ///     Determines whether interpretation is possible or mandatory.
    ///
    /// @param[in] keep_result_in_memory
    ///     True if the resulting persistent variable should reside in
    ///     target memory, if applicable.
    ///
    /// @return
    ///     True on success (no errors); false otherwise.
    //------------------------------------------------------------------
    virtual bool
    Parse (Stream &error_stream,
           ExecutionContext &exe_ctx,
           lldb_private::ExecutionPolicy execution_policy,
           bool keep_result_in_memory,
           bool generate_debug_info) = 0;

    bool
    CanInterpret ()
    {
        return m_can_interpret;
    }

    bool
    MatchesContext (ExecutionContext &exe_ctx);

    //------------------------------------------------------------------
    /// Execute the parsed expression
    ///
    /// @param[in] error_stream
    ///     A stream to print errors to.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use when looking up entities that
    ///     are needed for parsing (locations of variables, etc.)
    ///
    /// @param[in] options
    ///     Expression evaluation options.
    ///
    /// @param[in] shared_ptr_to_me
    ///     This is a shared pointer to this UserExpression.  This is
    ///     needed because Execute can push a thread plan that will hold onto
    ///     the UserExpression for an unbounded period of time.  So you
    ///     need to give the thread plan a reference to this object that can
    ///     keep it alive.
    ///
    /// @param[in] result
    ///     A pointer to direct at the persistent variable in which the
    ///     expression's result is stored.
    ///
    /// @return
    ///     A Process::Execution results value.
    //------------------------------------------------------------------
    lldb::ExpressionResults
    Execute (Stream &error_stream,
             ExecutionContext &exe_ctx,
             const EvaluateExpressionOptions& options,
             lldb::UserExpressionSP &shared_ptr_to_me,
             lldb::ExpressionVariableSP &result);

    //------------------------------------------------------------------
    /// Apply the side effects of the function to program state.
    ///
    /// @param[in] error_stream
    ///     A stream to print errors to.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use when looking up entities that
    ///     are needed for parsing (locations of variables, etc.)
    ///
    /// @param[in] result
    ///     A pointer to direct at the persistent variable in which the
    ///     expression's result is stored.
    ///
    /// @param[in] function_stack_pointer
    ///     A pointer to the base of the function's stack frame.  This
    ///     is used to determine whether the expression result resides in
    ///     memory that will still be valid, or whether it needs to be
    ///     treated as homeless for the purpose of future expressions.
    ///
    /// @return
    ///     A Process::Execution results value.
    //------------------------------------------------------------------
    bool
    FinalizeJITExecution (Stream &error_stream,
                          ExecutionContext &exe_ctx,
                          lldb::ExpressionVariableSP &result,
                          lldb::addr_t function_stack_bottom = LLDB_INVALID_ADDRESS,
                          lldb::addr_t function_stack_top = LLDB_INVALID_ADDRESS);

    //------------------------------------------------------------------
    /// Return the string that the parser should parse.  Must be a full
    /// translation unit.
    //------------------------------------------------------------------
    const char *
    Text() override
    {
        return m_transformed_text.c_str();
    }

    //------------------------------------------------------------------
    /// Return the string that the user typed.
    //------------------------------------------------------------------
    const char *
    GetUserText ()
    {
        return m_expr_text.c_str();
    }

    //------------------------------------------------------------------
    /// Return the function name that should be used for executing the
    /// expression.  Text() should contain the definition of this
    /// function.
    //------------------------------------------------------------------
    const char *
    FunctionName() override
    {
        return "$__lldb_expr";
    }

    //------------------------------------------------------------------
    /// Return the language that should be used when parsing.  To use
    /// the default, return eLanguageTypeUnknown.
    //------------------------------------------------------------------
    lldb::LanguageType
    Language() override
    {
        return m_language;
    }

    //------------------------------------------------------------------
    /// Return the desired result type of the function, or
    /// eResultTypeAny if indifferent.
    //------------------------------------------------------------------
    ResultType
    DesiredResultType() override
    {
        return m_desired_type;
    }

    //------------------------------------------------------------------
    /// Return true if validation code should be inserted into the
    /// expression.
    //------------------------------------------------------------------
    bool
    NeedsValidation() override
    {
        return true;
    }

    //------------------------------------------------------------------
    /// Return true if external variables in the expression should be
    /// resolved.
    //------------------------------------------------------------------
    bool
    NeedsVariableResolution() override
    {
        return true;
    }

    //------------------------------------------------------------------
    /// Evaluate one expression in the scratch context of the
    /// target passed in the exe_ctx and return its result.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use when evaluating the expression.
    ///
    /// @param[in] options
    ///     Expression evaluation options.  N.B. The language in the
    ///     evaluation options will be used to determine the language used for
    ///     expression evaluation.
    ///
    /// @param[in] expr_cstr
    ///     A C string containing the expression to be evaluated.
    ///
    /// @param[in] expr_prefix
    ///     If non-NULL, a C string containing translation-unit level
    ///     definitions to be included when the expression is parsed.
    ///
    /// @param[in,out] result_valobj_sp
    ///      If execution is successful, the result valobj is placed here.
    ///
    /// @param[out]
    ///     Filled in with an error in case the expression evaluation
    ///     fails to parse, run, or evaluated.
    ///
    /// @result
    ///      A Process::ExpressionResults value.  eExpressionCompleted for success.
    //------------------------------------------------------------------
    static lldb::ExpressionResults
    Evaluate (ExecutionContext &exe_ctx,
              const EvaluateExpressionOptions& options,
              const char *expr_cstr,
              const char *expr_prefix,
              lldb::ValueObjectSP &result_valobj_sp,
              Error &error);

    static const Error::ValueType kNoResult = 0x1001; ///< ValueObject::GetError() returns this if there is no result from the expression.
protected:
    static lldb::addr_t
    GetObjectPointer (lldb::StackFrameSP frame_sp,
                      ConstString &object_name,
                      Error &err);

    //------------------------------------------------------------------
    /// Populate m_in_cplusplus_method and m_in_objectivec_method based on the environment.
    //------------------------------------------------------------------

    virtual void
    ScanContext (ExecutionContext &exe_ctx,
                 lldb_private::Error &err) = 0;

    bool
    PrepareToExecuteJITExpression (Stream &error_stream,
                                   ExecutionContext &exe_ctx,
                                   lldb::addr_t &struct_address);
    
    virtual bool
    AddInitialArguments (ExecutionContext &exe_ctx,
                         std::vector<lldb::addr_t> &args,
                         Stream &error_stream)
    {
        return true;
    }

    void
    InstallContext (ExecutionContext &exe_ctx);

    bool
    LockAndCheckContext (ExecutionContext &exe_ctx,
                         lldb::TargetSP &target_sp,
                         lldb::ProcessSP &process_sp,
                         lldb::StackFrameSP &frame_sp);

    Address                                     m_address;              ///< The address the process is stopped in.
    lldb::addr_t                                m_stack_frame_bottom;   ///< The bottom of the allocated stack frame.
    lldb::addr_t                                m_stack_frame_top;      ///< The top of the allocated stack frame.

    std::string                                 m_expr_text;            ///< The text of the expression, as typed by the user
    std::string                                 m_expr_prefix;          ///< The text of the translation-level definitions, as provided by the user
    lldb::LanguageType                          m_language;             ///< The language to use when parsing (eLanguageTypeUnknown means use defaults)
    bool                                        m_allow_cxx;            ///< True if the language allows C++.
    bool                                        m_allow_objc;           ///< True if the language allows Objective-C.
    std::string                                 m_transformed_text;     ///< The text of the expression, as send to the parser
    ResultType                                  m_desired_type;         ///< The type to coerce the expression's result to.  If eResultTypeAny, inferred from the expression.

    std::shared_ptr<IRExecutionUnit>            m_execution_unit_sp;    ///< The execution unit the expression is stored in.
    std::unique_ptr<Materializer>               m_materializer_ap;      ///< The materializer to use when running the expression.
    lldb::ModuleWP                              m_jit_module_wp;
    bool                                        m_enforce_valid_object; ///< True if the expression parser should enforce the presence of a valid class pointer in order to generate the expression as a method.
    bool                                        m_in_cplusplus_method;  ///< True if the expression is compiled as a C++ member function (true if it was parsed when exe_ctx was in a C++ method).
    bool                                        m_in_objectivec_method; ///< True if the expression is compiled as an Objective-C method (true if it was parsed when exe_ctx was in an Objective-C method).
    bool                                        m_in_static_method;     ///< True if the expression is compiled as a static (or class) method (currently true if it was parsed when exe_ctx was in an Objective-C class method).
    bool                                        m_needs_object_ptr;     ///< True if "this" or "self" must be looked up and passed in.  False if the expression doesn't really use them and they can be NULL.
    bool                                        m_const_object;         ///< True if "this" is const.
    Target                                     *m_target;               ///< The target for storing persistent data like types and variables.

    bool                                        m_can_interpret;        ///< True if the expression could be evaluated statically; false otherwise.
    lldb::addr_t                                m_materialized_address; ///< The address at which the arguments to the expression have been materialized.
    Materializer::DematerializerSP              m_dematerializer_sp;    ///< The dematerializer.
};

} // namespace lldb_private

#endif // liblldb_UserExpression_h_
