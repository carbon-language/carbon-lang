//===-- CommandObjectRegister.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectRegister.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/NamedOptionValue.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/RegisterContext.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// "register read"
//----------------------------------------------------------------------
class CommandObjectRegisterRead : public CommandObject
{
public:
    CommandObjectRegisterRead (CommandInterpreter &interpreter) :
        CommandObject (interpreter, 
                       "register read",
                       "Dump the contents of one or more register values from the current frame.  If no register is specified, dumps them all.",
                       //"register read [<reg-name1> [<reg-name2> [...]]]",
                       NULL,
                       eFlagProcessMustBeLaunched | eFlagProcessMustBePaused),
        m_options (interpreter)
    {
        CommandArgumentEntry arg;
        CommandArgumentData register_arg;
        
        // Define the first (and only) variant of this arg.
        register_arg.arg_type = eArgTypeRegisterName;
        register_arg.arg_repetition = eArgRepeatStar;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (register_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    virtual
    ~CommandObjectRegisterRead ()
    {
    }

    Options *
    GetOptions ()
    {
        return &m_options;
    }

    bool
    DumpRegister (const ExecutionContext &exe_ctx,
                  Stream &strm,
                  RegisterContext *reg_ctx,
                  const RegisterInfo *reg_info)
    {
        if (reg_info)
        {
            uint32_t reg = reg_info->kinds[eRegisterKindLLDB];

            DataExtractor reg_data;

            if (reg_ctx->ReadRegisterBytes(reg, reg_data))
            {
                strm.Indent ();
                strm.Printf ("%-12s = ", reg_info ? reg_info->name : "<INVALID REGINFO>");
                Format format;
                if (m_options.format == eFormatDefault)
                    format = reg_info->format;
                else
                    format = m_options.format;

                reg_data.Dump(&strm, 0, format, reg_info->byte_size, 1, UINT32_MAX, LLDB_INVALID_ADDRESS, 0, 0);
                if (m_options.lookup_addresses && ((reg_info->encoding == eEncodingUint) || (reg_info->encoding == eEncodingSint)))
                {
                    addr_t reg_addr = reg_ctx->ReadRegisterAsUnsigned (reg, 0);
                    if (reg_addr)
                    {
                        Address so_reg_addr;
                        if (exe_ctx.target->GetSectionLoadList().ResolveLoadAddress(reg_addr, so_reg_addr))
                        {
                            strm.PutCString ("  ");
                            so_reg_addr.Dump(&strm, exe_ctx.GetBestExecutionContextScope(), Address::DumpStyleResolvedDescription);
                        }
                        else
                        {
                        }
                    }
                }
                strm.EOL();
                return true;
            }
        }
        return false;
    }

    bool
    DumpRegisterSet (const ExecutionContext &exe_ctx,
                     Stream &strm,
                     RegisterContext *reg_ctx,
                     uint32_t set_idx)
    {
        uint32_t unavailable_count = 0;
        uint32_t available_count = 0;
        const RegisterSet * const reg_set = reg_ctx->GetRegisterSet(set_idx);
        if (reg_set)
        {
            strm.Printf ("%s:\n", reg_set->name);
            strm.IndentMore ();
            const uint32_t num_registers = reg_set->num_registers;
            for (uint32_t reg_idx = 0; reg_idx < num_registers; ++reg_idx)
            {
                const uint32_t reg = reg_set->registers[reg_idx];
                if (DumpRegister (exe_ctx, strm, reg_ctx, reg_ctx->GetRegisterInfoAtIndex(reg)))
                    ++available_count;
                else
                    ++unavailable_count;
            }
            strm.IndentLess ();
            if (unavailable_count)
            {
                strm.Indent ();
                strm.Printf("%u registers were unavailable.\n", unavailable_count);
            }
            strm.EOL();
        }
        return available_count > 0;
    }

    virtual bool
    Execute 
    (
        Args& command,
        CommandReturnObject &result
    )
    {
        Stream &strm = result.GetOutputStream();
        ExecutionContext exe_ctx(m_interpreter.GetExecutionContext());
        RegisterContext *reg_ctx = exe_ctx.GetRegisterContext ();

        if (reg_ctx)
        {
            const RegisterInfo *reg_info = NULL;
            if (command.GetArgumentCount() == 0)
            {
                uint32_t set_idx;
                
                uint32_t num_register_sets = 1;
                const uint32_t set_array_size = m_options.set_indexes.GetSize();
                if (set_array_size > 0)
                {
                    for (uint32_t i=0; i<set_array_size; ++i)
                    {
                        set_idx = m_options.set_indexes[i]->GetUInt64Value (UINT32_MAX, NULL);
                        if (set_idx != UINT32_MAX)
                        {
                            if (!DumpRegisterSet (exe_ctx, strm, reg_ctx, set_idx))
                            {
                                result.AppendErrorWithFormat ("invalid register set index: %u\n", set_idx);
                                result.SetStatus (eReturnStatusFailed);
                                break;
                            }
                        }
                        else
                        {
                            result.AppendError ("invalid register set index\n");
                            result.SetStatus (eReturnStatusFailed);
                            break;
                        }
                    }
                }
                else
                {
                    if (m_options.dump_all_sets)
                        num_register_sets = reg_ctx->GetRegisterSetCount();

                    for (set_idx = 0; set_idx < num_register_sets; ++set_idx)
                    {
                        DumpRegisterSet (exe_ctx, strm, reg_ctx, set_idx);
                    }
                }
            }
            else
            {
                if (m_options.dump_all_sets)
                {
                    result.AppendError ("the --all option can't be used when registers names are supplied as arguments\n");
                    result.SetStatus (eReturnStatusFailed);
                }
                else if (m_options.set_indexes.GetSize() > 0)
                {
                    result.AppendError ("the --set <set> option can't be used when registers names are supplied as arguments\n");
                    result.SetStatus (eReturnStatusFailed);
                }
                else
                {
                    const char *arg_cstr;
                    for (int arg_idx = 0; (arg_cstr = command.GetArgumentAtIndex(arg_idx)) != NULL; ++arg_idx)
                    {
                        reg_info = reg_ctx->GetRegisterInfoByName(arg_cstr);

                        if (reg_info)
                        {
                            if (!DumpRegister (exe_ctx, strm, reg_ctx, reg_info))
                                strm.Printf("%-12s = error: unavailable\n", reg_info->name);
                        }
                        else
                        {
                            result.AppendErrorWithFormat ("Invalid register name '%s'.\n", arg_cstr);
                        }
                    }
                }
            }
        }
        else
        {
            result.AppendError ("no current frame");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }

protected:
    class CommandOptions : public Options
    {
    public:
        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter),
            set_indexes (OptionValue::ConvertTypeToMask (OptionValue::eTypeUInt64)),
            dump_all_sets (false, false), // Initial and default values are false
            lookup_addresses (false, false)         // Initial and default values are false
        {
            OptionParsingStarting();
        }
        
        virtual
        ~CommandOptions ()
        {
        }
        
        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;
            switch (short_option)
            {
                case 'f':
                    error = Args::StringToFormat (option_arg, format, NULL);
                    break;

                case 's':
                    {
                        OptionValueSP value_sp (OptionValueUInt64::Create (option_arg, error));
                        if (value_sp)
                            set_indexes.AppendValue (value_sp);
                    }
                    break;

                case 'a':
                    dump_all_sets.SetCurrentValue(true);
                    break;

                case 'l':
                    lookup_addresses.SetCurrentValue(true);
                    break;

                default:
                    error.SetErrorStringWithFormat("Unrecognized short option '%c'\n", short_option);
                    break;
            }
            return error;
        }
        
        void
        OptionParsingStarting ()
        {
            format = eFormatDefault;
            set_indexes.Clear();
            dump_all_sets.Clear();
            lookup_addresses.Clear();
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        lldb::Format format;
        OptionValueArray set_indexes;
        OptionValueBoolean dump_all_sets;
        OptionValueBoolean lookup_addresses;
    };

    CommandOptions m_options;
};

OptionDefinition
CommandObjectRegisterRead::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "format", 'f', required_argument, NULL, 0, eArgTypeExprFormat,  "Specify the format to use when dumping register values."},
    { LLDB_OPT_SET_ALL, false, "lookup", 'l', no_argument      , NULL, 0, eArgTypeNone      , "Lookup the register values as addresses and show that each value maps to in the address space."},
    { LLDB_OPT_SET_1  , false, "set"   , 's', required_argument, NULL, 0, eArgTypeIndex     , "Specify which register sets to dump by index."},
    { LLDB_OPT_SET_2  , false, "all"   , 'a', no_argument      , NULL, 0, eArgTypeNone      , "Show all register sets."},
    { 0, false, NULL, 0, 0, NULL, NULL, eArgTypeNone, NULL }
};



//----------------------------------------------------------------------
// "register write"
//----------------------------------------------------------------------
class CommandObjectRegisterWrite : public CommandObject
{
public:
    CommandObjectRegisterWrite (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "register write",
                       "Modify a single register value.",
                       //"register write <reg-name> <value>",
                       NULL,
                       eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData register_arg;
        CommandArgumentData value_arg;
        
        // Define the first (and only) variant of this arg.
        register_arg.arg_type = eArgTypeRegisterName;
        register_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (register_arg);
        
        // Define the first (and only) variant of this arg.
        value_arg.arg_type = eArgTypeValue;
        value_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg2.push_back (value_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
    }

    virtual
    ~CommandObjectRegisterWrite ()
    {
    }

    virtual bool
    Execute 
    (
        Args& command,
        CommandReturnObject &result
    )
    {
        DataExtractor reg_data;
        ExecutionContext exe_ctx(m_interpreter.GetExecutionContext());
        RegisterContext *reg_ctx = exe_ctx.GetRegisterContext ();

        if (reg_ctx)
        {
            if (command.GetArgumentCount() != 2)
            {
                result.AppendError ("register write takes exactly 2 arguments: <reg-name> <value>");
                result.SetStatus (eReturnStatusFailed);
            }
            else
            {
                const char *reg_name = command.GetArgumentAtIndex(0);
                const char *value_str = command.GetArgumentAtIndex(1);
                const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName(reg_name);

                if (reg_info)
                {
                    Scalar scalar;
                    Error error(scalar.SetValueFromCString (value_str, reg_info->encoding, reg_info->byte_size));
                    if (error.Success())
                    {
                        if (reg_ctx->WriteRegisterValue(reg_info->kinds[eRegisterKindLLDB], scalar))
                        {
                            result.SetStatus (eReturnStatusSuccessFinishNoResult);
                            return true;
                        }
                    }
                    else
                    {
                        result.AppendErrorWithFormat ("Failed to write register '%s' with value '%s': %s\n",
                                                     reg_name,
                                                     value_str,
                                                     error.AsCString());
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
                else
                {
                    result.AppendErrorWithFormat ("Register not found for '%s'.\n", reg_name);
                    result.SetStatus (eReturnStatusFailed);
                }
            }
        }
        else
        {
            result.AppendError ("no current frame");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};


//----------------------------------------------------------------------
// CommandObjectRegister constructor
//----------------------------------------------------------------------
CommandObjectRegister::CommandObjectRegister(CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "register",
                            "A set of commands to access thread registers.",
                            "register [read|write] ...")
{
    LoadSubCommand ("read",  CommandObjectSP (new CommandObjectRegisterRead (interpreter)));
    LoadSubCommand ("write", CommandObjectSP (new CommandObjectRegisterWrite (interpreter)));
}


//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
CommandObjectRegister::~CommandObjectRegister()
{
}
