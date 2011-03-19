//===-- CommandObjectPlatform.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectPlatform.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// "platform create <platform-name>"
//----------------------------------------------------------------------
class CommandObjectPlatformCreate : public CommandObject
{
public:
    CommandObjectPlatformCreate (CommandInterpreter &interpreter) :
        CommandObject (interpreter, 
                       "platform create",
                       "Create a platform instance by name and select it as the current platform.",
                       "platform create <platform-name>",
                       0)
    {
    }

    virtual
    ~CommandObjectPlatformCreate ()
    {
    }

    virtual bool
    Execute (Args& args, CommandReturnObject &result)
    {
        Error error;
        if (args.GetArgumentCount() == 1)
        {
            PlatformSP platform_sp (Platform::Create (args.GetArgumentAtIndex (0), error));
            
            if (platform_sp)
            {
                m_interpreter.GetDebugger().GetPlatformList().Append (platform_sp, true);
                if (m_options.os_version_major != UINT32_MAX)
                {
                    platform_sp->SetOSVersion (m_options.os_version_major,
                                               m_options.os_version_minor,
                                               m_options.os_version_update);
                }
                
                platform_sp->GetStatus (result.GetOutputStream());
            }
        }
        else
        {
            result.AppendError ("command not implemented");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }

protected:

    class CommandOptions : public Options
    {
    public:

        CommandOptions () :
            os_version_major (UINT32_MAX),
            os_version_minor (UINT32_MAX),
            os_version_update (UINT32_MAX)
        {
        }

        virtual
        ~CommandOptions ()
        {
        }

        virtual Error
        SetOptionValue (int option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
            case 'v':
                if (Args::StringToVersion (option_arg, 
                                           os_version_major,
                                           os_version_minor,
                                           os_version_update) == option_arg)
                {
                    error.SetErrorStringWithFormat ("invalid version string '%s'", option_arg);
                }
                break;

            default:
                error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
                break;
            }

            return error;
        }

        void
        ResetOptionValues ()
        {
            os_version_major = UINT32_MAX;
            os_version_minor = UINT32_MAX;
            os_version_update = UINT32_MAX;
        }

        const lldb::OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static lldb::OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        uint32_t os_version_major;
        uint32_t os_version_minor;
        uint32_t os_version_update;
    };
    CommandOptions m_options;
};

lldb::OptionDefinition
CommandObjectPlatformCreate::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "sdk-version", 'v', required_argument, NULL, 0, eArgTypeNone, "Specify the initial SDK version to use prior to connecting." },
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

//----------------------------------------------------------------------
// "platform list"
//----------------------------------------------------------------------
class CommandObjectPlatformList : public CommandObject
{
public:
    CommandObjectPlatformList (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "platform list",
                       "List all platforms that are available.",
                       NULL,
                       0)
    {
    }

    virtual
    ~CommandObjectPlatformList ()
    {
    }

    virtual bool
    Execute (Args& args, CommandReturnObject &result)
    {
        Stream &ostrm = result.GetOutputStream();
        ostrm.Printf("Available platforms:\n");
        
        PlatformSP host_platform_sp (Platform::GetDefaultPlatform());
        ostrm.Printf ("%s: %s\n", 
                      host_platform_sp->GetShortPluginName(), 
                      host_platform_sp->GetDescription());

        uint32_t idx;
        for (idx = 0; 1; ++idx)
        {
            const char *plugin_name = PluginManager::GetPlatformPluginNameAtIndex (idx);
            if (plugin_name == NULL)
                break;
            const char *plugin_desc = PluginManager::GetPlatformPluginDescriptionAtIndex (idx);
            if (plugin_desc == NULL)
                break;
            ostrm.Printf("%s: %s\n", plugin_name, plugin_desc);
        }
        
        if (idx == 0)
        {
            result.AppendError ("no platforms are available");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

//----------------------------------------------------------------------
// "platform status"
//----------------------------------------------------------------------
class CommandObjectPlatformStatus : public CommandObject
{
public:
    CommandObjectPlatformStatus (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "platform status",
                       "Display status for the currently selected platform.",
                       NULL,
                       0)
    {
    }

    virtual
    ~CommandObjectPlatformStatus ()
    {
    }

    virtual bool
    Execute (Args& args, CommandReturnObject &result)
    {
        Stream &ostrm = result.GetOutputStream();      
        
        PlatformSP selected_platform_sp (m_interpreter.GetDebugger().GetPlatformList().GetSelectedPlatform());
        if (selected_platform_sp)
        {
            selected_platform_sp->GetStatus (ostrm);
            result.SetStatus (eReturnStatusSuccessFinishResult);            
        }
        else
        {
            result.AppendError ("no platform us currently selected");
            result.SetStatus (eReturnStatusFailed);            
        }
        return result.Succeeded();
    }
};


//----------------------------------------------------------------------
// "platform select <platform-name>"
//----------------------------------------------------------------------
class CommandObjectPlatformSelect : public CommandObject
{
public:
    CommandObjectPlatformSelect (CommandInterpreter &interpreter) :
        CommandObject (interpreter, 
                       "platform select",
                       "Select a platform by name to be the currently selected platform.",
                       "platform select <platform-name>",
                       0)
    {
    }

    virtual
    ~CommandObjectPlatformSelect ()
    {
    }

    virtual bool
    Execute (Args& args, CommandReturnObject &result)
    {
        result.AppendError ("command not implemented");
        result.SetStatus (eReturnStatusFailed);
        return result.Succeeded();
    }
};



//----------------------------------------------------------------------
// CommandObjectPlatform constructor
//----------------------------------------------------------------------
CommandObjectPlatform::CommandObjectPlatform(CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "platform",
                            "A set of commands to manage and create platforms.",
                            "platform [create|list|status|select] ...")
{
    LoadSubCommand ("create", CommandObjectSP (new CommandObjectPlatformCreate  (interpreter)));
    LoadSubCommand ("list"  , CommandObjectSP (new CommandObjectPlatformList    (interpreter)));
    LoadSubCommand ("select", CommandObjectSP (new CommandObjectPlatformSelect  (interpreter)));
    LoadSubCommand ("status", CommandObjectSP (new CommandObjectPlatformStatus  (interpreter)));
}


//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
CommandObjectPlatform::~CommandObjectPlatform()
{
}
