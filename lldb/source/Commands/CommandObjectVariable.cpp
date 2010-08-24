//===-- CommandObjectVariable.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectVariable.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectVariable.h"

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

//void
//DumpValueObjectValues (Stream *sout, const char *root_valobj_name, ValueObjectSP& valobj_sp, bool follow_ptrs_and_refs, uint32_t curr_depth, uint32_t max_depth)
//{
//    ValueObject *valobj = valobj_sp.get();
//    if (valobj)
//    {
//        const char *name_cstr = valobj->GetName().AsCString(NULL);
//        const char *val_cstr = valobj->GetValueAsCString();
//        const char *loc_cstr = valobj->GetLocationAsCString();
//        const char *type_cstr = valobj->GetTypeName().AsCString();
//        const char *sum_cstr = valobj->GetSummaryAsCString();
//        const char *err_cstr = valobj->GetError().AsCString();
//        // Indent
//        sout->Indent();
//        if (root_valobj_name)
//        {
//            sout->Printf ("%s = ", root_valobj_name);
//        }
//
//        if (name_cstr)
//            sout->Printf ("%s => ", name_cstr);
//
//        sout->Printf ("ValueObject{%u}", valobj->GetID());
//        const uint32_t num_children = valobj->GetNumChildren();
//
//        if (type_cstr)
//            sout->Printf (", type = '%s'", type_cstr);
//
//        if (loc_cstr)
//            sout->Printf (", location = %s", loc_cstr);
//
//        sout->Printf (", num_children = %u", num_children);
//
//        if (val_cstr)
//            sout->Printf (", value = %s", val_cstr);
//
//        if (err_cstr)
//            sout->Printf (", error = %s", err_cstr);
//
//        if (sum_cstr)
//            sout->Printf (", summary = %s", sum_cstr);
//
//        sout->EOL();
//        bool is_ptr_or_ref = ClangASTContext::IsPointerOrReferenceType (valobj->GetOpaqueClangQualType());
//        if (!follow_ptrs_and_refs && is_ptr_or_ref)
//            return;
//
//        if (curr_depth < max_depth)
//        {
//            for (uint32_t idx=0; idx<num_children; ++idx)
//            {
//                ValueObjectSP child_sp(valobj->GetChildAtIndex(idx, true));
//                if (child_sp.get())
//                {
//                    sout->IndentMore();
//                    DumpValueObjectValues (sout, NULL, child_sp, follow_ptrs_and_refs, curr_depth + 1, max_depth);
//                    sout->IndentLess();
//                }
//            }
//        }
//    }
//}

//----------------------------------------------------------------------
// List images with associated information
//----------------------------------------------------------------------
class CommandObjectVariableList : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:

        CommandOptions () :
            Options()
        {
            ResetOptionValues ();
        }

        virtual
        ~CommandOptions ()
        {
        }

        virtual Error
        SetOptionValue (int option_idx, const char *option_arg)
        {
            Error error;
            bool success;
            char short_option = (char) m_getopt_table[option_idx].val;
            switch (short_option)
            {
            case 'o':   use_objc     = true;  break;
            case 'n':   name = option_arg;    break;
            case 'r':   use_regex    = true;  break;
            case 'a':   show_args    = false; break;
            case 'l':   show_locals  = false; break;
            case 'g':   show_globals = false; break;
            case 't':   show_types   = false; break;
            case 'y':   show_summary = false; break;
            case 'L':   show_location= true;  break;
            case 'D':   debug        = true;  break;
            case 'd':
                max_depth = Args::StringToUInt32 (option_arg, UINT32_MAX, 0, &success);
                if (!success)
                    error.SetErrorStringWithFormat("Invalid max depth '%s'.\n", option_arg);
                break;

            case 'p':
                ptr_depth = Args::StringToUInt32 (option_arg, 0, 0, &success);
                if (!success)
                    error.SetErrorStringWithFormat("Invalid pointer depth '%s'.\n", option_arg);
                break;

            case 'G':
                {
                    ConstString const_string (option_arg);
                    globals.push_back(const_string);
                }
                break;

            case 's':
                show_scope = true;
                break;

            default:
                error.SetErrorStringWithFormat("Invalid short option character '%c'.\n", short_option);
                break;
            }

            return error;
        }

        void
        ResetOptionValues ()
        {
            Options::ResetOptionValues();

            name.clear();
            use_objc      = false;
            use_regex     = false;
            show_args     = true;
            show_locals   = true;
            show_globals  = true;
            show_types    = true;
            show_scope    = false;
            show_summary  = true;
            show_location = false;
            debug         = false;
            max_depth     = UINT32_MAX;
            ptr_depth     = 0;
            globals.clear();
        }

        const lldb::OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static lldb::OptionDefinition g_option_table[];
        std::string name;
        bool use_objc;
        bool use_regex;
        bool show_args;
        bool show_locals;
        bool show_globals;
        bool show_types;
        bool show_scope; // local/arg/global/static
        bool show_summary;
        bool show_location;
        bool debug;
        uint32_t max_depth; // The depth to print when dumping concrete (not pointers) aggreate values
        uint32_t ptr_depth; // The default depth that is dumped when we find pointers
        std::vector<ConstString> globals;
        // Instance variables to hold the values for command options.
    };

    CommandObjectVariableList () :
        CommandObject (
                "variable list",
                "Show specified argument, local variable, static variable or global variable.  If none specified, list them all.",
                "variable list [<cmd-options>] [<var-name1> [<var-name2>...]]")
    {
    }

    virtual
    ~CommandObjectVariableList ()
    {
    }

    virtual
    Options *
    GetOptions ()
    {
        return &m_options;
    }

    void
    DumpVariable (CommandReturnObject &result, ExecutionContext *exe_ctx, Variable *variable)
    {
        if (variable)
        {
            Stream &s = result.GetOutputStream();
            DWARFExpression &expr = variable->LocationExpression();
            Value expr_result;
            Error expr_error;
            Type *variable_type = variable->GetType();
            bool expr_success = expr.Evaluate(exe_ctx, NULL, NULL, expr_result, &expr_error);

            if (m_options.debug)
                s.Printf ("Variable{0x%8.8x}: ", variable->GetID());

            if (!expr_success)
                s.Printf ("%s = ERROR: %s\n", variable->GetName().AsCString(NULL), expr_error.AsCString());
            else
            {
                Value::ValueType expr_value_type = expr_result.GetValueType();
                switch (expr_value_type)
                {
                case Value::eValueTypeScalar:
                    s.Printf ("%s = ", variable->GetName().AsCString(NULL));
                    if (variable_type)
                    {
                        DataExtractor data;
                        if (expr_result.ResolveValue (exe_ctx, NULL).GetData (data))
                            variable_type->DumpValue (exe_ctx, &s, data, 0, m_options.show_types, m_options.show_summary, m_options.debug);
                    }
                    break;

                    case Value::eValueTypeFileAddress:
                    case Value::eValueTypeLoadAddress:
                    case Value::eValueTypeHostAddress:
                    {
                        s.Printf ("%s = ", variable->GetName().AsCString(NULL));
                        lldb::addr_t addr = LLDB_INVALID_ADDRESS;
                        lldb::AddressType addr_type = eAddressTypeLoad;

                        if (expr_value_type == Value::eValueTypeFileAddress)
                        {
                            lldb::addr_t file_addr = expr_result.ResolveValue (exe_ctx, NULL).ULongLong(LLDB_INVALID_ADDRESS);
                            SymbolContext var_sc;
                            variable->CalculateSymbolContext(&var_sc);
                            if (var_sc.module_sp)
                            {
                                ObjectFile *objfile = var_sc.module_sp->GetObjectFile();
                                if (objfile)
                                {
                                    Address so_addr(file_addr, objfile->GetSectionList());
                                    addr = so_addr.GetLoadAddress(exe_ctx->process);
                                }
                                if (addr == LLDB_INVALID_ADDRESS)
                                {
                                    result.GetErrorStream().Printf ("error: %s is not loaded", var_sc.module_sp->GetFileSpec().GetFilename().AsCString());
                                }
                            }
                            else
                            {
                                result.GetErrorStream().Printf ("error: unable to resolve the variable address 0x%llx", file_addr);
                            }
                        }
                        else
                        {
                            if (expr_value_type == Value::eValueTypeHostAddress)
                                addr_type = eAddressTypeHost;
                            addr = expr_result.ResolveValue (exe_ctx, NULL).ULongLong(LLDB_INVALID_ADDRESS);
                        }

                        if (addr != LLDB_INVALID_ADDRESS)
                        {
                            if (m_options.debug)
                                s.Printf("@ 0x%8.8llx, value = ", addr);
                            variable_type->DumpValueInMemory (exe_ctx, &s, addr, addr_type, m_options.show_types, m_options.show_summary, m_options.debug);
                        }
                    }
                    break;
                }
                s.EOL();
            }
        }
    }

    void
    DumpValueObject (CommandReturnObject &result,
                     ExecutionContextScope *exe_scope,
                     ValueObject *valobj,
                     const char *root_valobj_name,
                     uint32_t ptr_depth,
                     uint32_t curr_depth,
                     uint32_t max_depth,
                     bool use_objc)
    {
        if (valobj)
        {
            Stream &s = result.GetOutputStream();

            //const char *loc_cstr = valobj->GetLocationAsCString();
            if (m_options.show_location)
            {
                s.Printf("@ %s: ", valobj->GetLocationAsCString(exe_scope));
            }
            if (m_options.debug)
                s.Printf ("%p ValueObject{%u} ", valobj, valobj->GetID());

            s.Indent();

            if (m_options.show_types)
                s.Printf("(%s) ", valobj->GetTypeName().AsCString());

            const char *name_cstr = root_valobj_name ? root_valobj_name : valobj->GetName().AsCString("");
            s.Printf ("%s = ", name_cstr);

            const char *val_cstr = valobj->GetValueAsCString(exe_scope);
            const char *err_cstr = valobj->GetError().AsCString();

            if (err_cstr)
            {
                s.Printf ("error: %s\n", err_cstr);
            }
            else
            {
                const char *sum_cstr = valobj->GetSummaryAsCString(exe_scope);

                const bool is_aggregate = ClangASTContext::IsAggregateType (valobj->GetOpaqueClangQualType());

                if (val_cstr)
                    s.PutCString(val_cstr);

                if (sum_cstr)
                    s.Printf(" %s", sum_cstr);
                
                if (use_objc)
                {
                    if (!ClangASTContext::IsPointerType (valobj->GetOpaqueClangQualType()))
                        return;
                    
                    if (!valobj->GetValueIsValid())
                        return;
                    
                    Process *process = exe_scope->CalculateProcess();
                    
                    if (!process)
                        return;
                    
                    Scalar scalar;
                    
                    if (!ClangASTType::GetValueAsScalar (valobj->GetClangAST(),
                                                        valobj->GetOpaqueClangQualType(),
                                                        valobj->GetDataExtractor(),
                                                        0,
                                                        valobj->GetByteSize(),
                                                        scalar))
                        return;
                                        
                    ConstString po_output;
                    
                    ExecutionContext exe_ctx;
                    exe_scope->Calculate(exe_ctx);
                    
                    Value val(scalar);
                    val.SetContext(Value::eContextTypeOpaqueClangQualType, 
                                   ClangASTContext::GetVoidPtrType(valobj->GetClangAST(), false));
                    
                    if (!process->GetObjCObjectPrinter().PrintObject(po_output, val, exe_ctx))
                        return;
                    
                    s.Printf("\n%s\n", po_output.GetCString());
                                        
                    return;
                }


                if (curr_depth < max_depth)
                {
                    if (is_aggregate)
                        s.PutChar('{');

                    bool is_ptr_or_ref = ClangASTContext::IsPointerOrReferenceType (valobj->GetOpaqueClangQualType());
                    
                    if (is_ptr_or_ref && ptr_depth == 0)
                        return;

                    const uint32_t num_children = valobj->GetNumChildren();
                    if (num_children)
                    {
                        s.IndentMore();
                        for (uint32_t idx=0; idx<num_children; ++idx)
                        {
                            ValueObjectSP child_sp(valobj->GetChildAtIndex(idx, true));
                            if (child_sp.get())
                            {
                                s.EOL();
                                DumpValueObject (result,
                                                 exe_scope,
                                                 child_sp.get(),
                                                 NULL,
                                                 is_ptr_or_ref ? ptr_depth - 1 : ptr_depth,
                                                 curr_depth + 1,
                                                 max_depth,
                                                 false);
                                if (idx + 1 < num_children)
                                    s.PutChar(',');
                            }
                        }
                        s.IndentLess();
                    }
                    if (is_aggregate)
                    {
                        s.EOL();
                        s.Indent("}");
                    }
                }
                else
                {
                    if (is_aggregate)
                    {
                        s.PutCString("{...}");
                    }
                }

            }
        }
    }

    virtual bool
    Execute
    (
        CommandInterpreter &interpreter,
        Args& command,
        CommandReturnObject &result
    )
    {
        ExecutionContext exe_ctx(interpreter.GetDebugger().GetExecutionContext());
        if (exe_ctx.frame == NULL)
        {
            result.AppendError ("invalid frame");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            VariableList variable_list;

            bool show_inlined = true;   // TODO: Get this from the process
            SymbolContext frame_sc = exe_ctx.frame->GetSymbolContext (eSymbolContextEverything);
            if (exe_ctx.frame && frame_sc.block)
                frame_sc.block->AppendVariables(true, true, show_inlined, &variable_list);
            VariableSP var_sp;
            ValueObjectSP valobj_sp;
            //ValueObjectList &valobj_list = exe_ctx.frame->GetValueObjectList();
            const char *name_cstr = NULL;
            size_t idx;
            if (!m_options.globals.empty())
            {
                uint32_t fail_count = 0;
                if (exe_ctx.target)
                {
                    const size_t num_globals = m_options.globals.size();
                    for (idx = 0; idx < num_globals; ++idx)
                    {
                        VariableList global_var_list;
                        const uint32_t num_matching_globals = exe_ctx.target->GetImages().FindGlobalVariables (m_options.globals[idx], true, UINT32_MAX, global_var_list);

                        if (num_matching_globals == 0)
                        {
                            ++fail_count;
                            result.GetErrorStream().Printf ("error: can't find global variable '%s'\n", m_options.globals[idx].AsCString());
                        }
                        else
                        {
                            for (uint32_t global_idx=0; global_idx<num_matching_globals; ++global_idx)
                            {
                                var_sp = global_var_list.GetVariableAtIndex(global_idx);
                                if (var_sp)
                                {
                                    valobj_sp = exe_ctx.frame->GetValueObjectList().FindValueObjectByValueName (m_options.globals[idx].AsCString());
                                    if (!valobj_sp)
                                        valobj_sp.reset (new ValueObjectVariable (var_sp));

                                    if (valobj_sp)
                                    {
                                        exe_ctx.frame->GetValueObjectList().Append (valobj_sp);
                                        DumpValueObject (result, exe_ctx.frame, valobj_sp.get(), name_cstr, m_options.ptr_depth, 0, m_options.max_depth, false);
                                        result.GetOutputStream().EOL();
                                    }
                                }
                            }
                        }
                    }
                }
                if (fail_count)
                {
                    result.SetStatus (eReturnStatusFailed);
                }
            }
            
            if (command.GetArgumentCount() > 0)
            {
                // If we have any args to the variable command, we will make
                // variable objects from them...
                for (idx = 0; (name_cstr = command.GetArgumentAtIndex(idx)) != NULL; ++idx)
                {
                    uint32_t ptr_depth = m_options.ptr_depth;
                    // If first character is a '*', then show pointer contents
                    if (name_cstr[0] == '*')
                    {
                        ++ptr_depth;
                        name_cstr++; // Skip the '*'
                    }

                    std::string var_path (name_cstr);
                    size_t separator_idx = var_path.find_first_of(".-[");

                    ConstString name_const_string;
                    if (separator_idx == std::string::npos)
                        name_const_string.SetCString (var_path.c_str());
                    else
                        name_const_string.SetCStringWithLength (var_path.c_str(), separator_idx);

                    var_sp = variable_list.FindVariable(name_const_string);
                    if (var_sp)
                    {
                        //DumpVariable (result, &exe_ctx, var_sp.get());
                        // TODO: redo history variables using a different map
//                        if (var_path[0] == '$')
//                            valobj_sp = valobj_list.FindValueObjectByValueObjectName (name_const_string.GetCString());
//                        else
                            valobj_sp = exe_ctx.frame->GetValueObjectList().FindValueObjectByValueName (name_const_string.GetCString());

                        if (!valobj_sp)
                        {
                            valobj_sp.reset (new ValueObjectVariable (var_sp));
                            exe_ctx.frame->GetValueObjectList().Append (valobj_sp);
                        }

                        var_path.erase (0, name_const_string.GetLength ());
                        // We are dumping at least one child
                        while (separator_idx != std::string::npos)
                        {
                            // Calculate the next separator index ahead of time
                            ValueObjectSP child_valobj_sp;
                            const char separator_type = var_path[0];
                            switch (separator_type)
                            {

                            case '-':
                                if (var_path.size() >= 2 && var_path[1] != '>')
                                {
                                    result.GetErrorStream().Printf ("error: invalid character in variable path starting at '%s'\n",
                                                                    var_path.c_str());
                                    var_path.clear();
                                    valobj_sp.reset();
                                    break;
                                }
                                var_path.erase (0, 1); // Remove the '-'
                                // Fall through
                            case '.':
                                {
                                    var_path.erase (0, 1); // Remove the '.' or '>'
                                    separator_idx = var_path.find_first_of(".-[");
                                    ConstString child_name;
                                    if (separator_idx == std::string::npos)
                                        child_name.SetCString (var_path.c_str());
                                    else
                                        child_name.SetCStringWithLength(var_path.c_str(), separator_idx);

                                    child_valobj_sp = valobj_sp->GetChildMemberWithName (child_name, true);
                                    if (!child_valobj_sp)
                                    {
                                        result.GetErrorStream().Printf ("error: can't find child of '%s' named '%s'\n",
                                                                        valobj_sp->GetName().AsCString(),
                                                                        child_name.GetCString());
                                        var_path.clear();
                                        valobj_sp.reset();
                                        break;
                                    }
                                    // Remove the child name from the path
                                    var_path.erase(0, child_name.GetLength());
                                }
                                break;

                            case '[':
                                // Array member access, or treating pointer as an array
                                if (var_path.size() > 2) // Need at least two brackets and a number
                                {
                                    char *end = NULL;
                                    int32_t child_index = ::strtol (&var_path[1], &end, 0);
                                    if (end && *end == ']')
                                    {

                                        if (valobj_sp->IsPointerType ())
                                        {
                                            child_valobj_sp = valobj_sp->GetSyntheticArrayMemberFromPointer (child_index, true);
                                        }
                                        else
                                        {
                                            child_valobj_sp = valobj_sp->GetChildAtIndex (child_index, true);
                                        }

                                        if (!child_valobj_sp)
                                        {
                                            result.GetErrorStream().Printf ("error: invalid array index %u in '%s'\n",
                                                                            child_index,
                                                                            valobj_sp->GetName().AsCString());
                                            var_path.clear();
                                            valobj_sp.reset();
                                            break;
                                        }

                                        // Erase the array member specification '[%i]' where %i is the array index
                                        var_path.erase(0, (end - var_path.c_str()) + 1);
                                        separator_idx = var_path.find_first_of(".-[");

                                        // Break out early from the switch since we were able to find the child member
                                        break;
                                    }
                                }
                                result.GetErrorStream().Printf ("error: invalid array member specification for '%s' starting at '%s'\n",
                                                                valobj_sp->GetName().AsCString(),
                                                                var_path.c_str());
                                var_path.clear();
                                valobj_sp.reset();
                                break;

                                break;

                            default:
                                result.GetErrorStream().Printf ("error: invalid character in variable path starting at '%s'\n",
                                                                    var_path.c_str());
                                var_path.clear();
                                valobj_sp.reset();
                                separator_idx = std::string::npos;
                                break;
                            }

                            if (child_valobj_sp)
                                valobj_sp = child_valobj_sp;

                            if (var_path.empty())
                                break;

                        }

                        if (valobj_sp)
                        {
                            DumpValueObject (result, exe_ctx.frame, valobj_sp.get(), name_cstr, ptr_depth, 0, m_options.max_depth, m_options.use_objc);
                            result.GetOutputStream().EOL();
                        }
                    }
                    else
                    {
                        result.GetErrorStream().Printf ("error: unable to find any variables named '%s'\n", name_cstr);
                        var_path.clear();
                    }
                }
            }
            else
            {

                if (m_options.show_globals)
                {
                    if (frame_sc.comp_unit)
                    {
                        variable_list.AddVariables (frame_sc.comp_unit->GetVariableList(true).get());
                    }
                }

                const uint32_t num_variables = variable_list.GetSize();
    
                if (num_variables > 0)
                {
                    for (uint32_t i=0; i<num_variables; i++)
                    {
                        Variable *variable = variable_list.GetVariableAtIndex(i).get();
                        bool dump_variable = true;
                        
                        switch (variable->GetScope())
                        {
                        case eValueTypeVariableGlobal:
                            dump_variable = m_options.show_globals;
                            if (dump_variable && m_options.show_scope)
                                result.GetOutputStream().PutCString("GLOBAL: ");
                            break;

                        case eValueTypeVariableStatic:
                            dump_variable = m_options.show_globals;
                            if (dump_variable && m_options.show_scope)
                                result.GetOutputStream().PutCString("STATIC: ");
                            break;
                            
                        case eValueTypeVariableArgument:
                            dump_variable = m_options.show_args;
                            if (dump_variable && m_options.show_scope)
                                result.GetOutputStream().PutCString("   ARG: ");
                            break;
                            
                        case eValueTypeVariableLocal:
                            dump_variable = m_options.show_locals;
                            if (dump_variable && m_options.show_scope)
                                result.GetOutputStream().PutCString(" LOCAL: ");
                            break;

                        default:
                            break;
                        }
                        
                        if (dump_variable)
                            DumpVariable (result, &exe_ctx, variable);
                    }
                }
            }
            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        return result.Succeeded();
    }
protected:

    CommandOptions m_options;
};

lldb::OptionDefinition
CommandObjectVariableList::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "debug",      'D', no_argument,       NULL, 0, NULL,        "Show verbose debug information."},
{ LLDB_OPT_SET_1, false, "depth",      'd', required_argument, NULL, 0, "<count>",   "Set the max recurse depth when dumping aggregate types (default is infinity)."},
{ LLDB_OPT_SET_1, false, "globals",    'g', no_argument,       NULL, 0, NULL,        "List global and static variables for the current stack frame source file."},
{ LLDB_OPT_SET_1, false, "global",     'G', required_argument, NULL, 0, NULL,        "Find a global variable by name (which might not be in the current stack frame source file)."},
{ LLDB_OPT_SET_1, false, "location",   'L', no_argument,       NULL, 0, NULL,        "Show variable location information."},
{ LLDB_OPT_SET_1, false, "name",       'n', required_argument, NULL, 0, "<name>",    "Lookup a variable by name or regex (--regex) for the current execution context."},
{ LLDB_OPT_SET_1, false, "no-args",    'a', no_argument,       NULL, 0, NULL,        "Omit function arguments."},
{ LLDB_OPT_SET_1, false, "no-locals",  'l', no_argument,       NULL, 0, NULL,        "Omit local variables."},
{ LLDB_OPT_SET_1, false, "no-types",   't', no_argument,       NULL, 0, NULL,        "Omit variable type names."},
{ LLDB_OPT_SET_1, false, "no-summary", 'y', no_argument,       NULL, 0, NULL,        "Omit summary information."},
{ LLDB_OPT_SET_1, false, "scope",      's', no_argument,       NULL, 0, NULL,        "Show variable scope (argument, local, global, static)."},
{ LLDB_OPT_SET_1, false, "objc",       'o', no_argument,       NULL, 0, NULL,        "When looking up a variable by name (--name), print as an Objective-C object."},
{ LLDB_OPT_SET_1, false, "ptr-depth",  'p', required_argument, NULL, 0, "<count>",   "The number of pointers to be traversed when dumping values (default is zero)."},
{ LLDB_OPT_SET_1, false, "regex",      'r', no_argument,       NULL, 0, NULL,        "The <name> argument for name lookups are regular expressions."},
{ 0, false, NULL, 0, 0, NULL, NULL, NULL, NULL }
};

//----------------------------------------------------------------------
// CommandObjectVariable constructor
//----------------------------------------------------------------------
CommandObjectVariable::CommandObjectVariable(CommandInterpreter &interpreter) :
    CommandObjectMultiword ("variable",
                            "Access program arguments, locals, static and global variables.",
                            "variable [list] ...")
{
    LoadSubCommand (interpreter, "list", CommandObjectSP (new CommandObjectVariableList ()));
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
CommandObjectVariable::~CommandObjectVariable()
{
}




