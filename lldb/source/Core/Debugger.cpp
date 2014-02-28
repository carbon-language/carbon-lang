//===-- Debugger.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/Core/Debugger.h"

#include <map>

#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"

#include "lldb/lldb-private.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/State.h"
#include "lldb/Core/StreamAsynchronousIO.h"
#include "lldb/Core/StreamCallback.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/Host/DynamicLibrary.h"
#include "lldb/Host/Terminal.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/OptionValueSInt64.h"
#include "lldb/Interpreter/OptionValueString.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/AnsiTerminal.h"

using namespace lldb;
using namespace lldb_private;


static uint32_t g_shared_debugger_refcount = 0;
static lldb::user_id_t g_unique_id = 1;

#pragma mark Static Functions

static Mutex &
GetDebuggerListMutex ()
{
    static Mutex g_mutex(Mutex::eMutexTypeRecursive);
    return g_mutex;
}

typedef std::vector<DebuggerSP> DebuggerList;

static DebuggerList &
GetDebuggerList()
{
    // hide the static debugger list inside a singleton accessor to avoid
    // global init contructors
    static DebuggerList g_list;
    return g_list;
}

OptionEnumValueElement
g_show_disassembly_enum_values[] =
{
    { Debugger::eStopDisassemblyTypeNever,    "never",     "Never show disassembly when displaying a stop context."},
    { Debugger::eStopDisassemblyTypeNoSource, "no-source", "Show disassembly when there is no source information, or the source file is missing when displaying a stop context."},
    { Debugger::eStopDisassemblyTypeAlways,   "always",    "Always show disassembly when displaying a stop context."},
    { 0, NULL, NULL }
};

OptionEnumValueElement
g_language_enumerators[] =
{
    { eScriptLanguageNone,      "none",     "Disable scripting languages."},
    { eScriptLanguagePython,    "python",   "Select python as the default scripting language."},
    { eScriptLanguageDefault,   "default",  "Select the lldb default as the default scripting language."},
    { 0, NULL, NULL }
};

#define MODULE_WITH_FUNC "{ ${module.file.basename}{`${function.name-with-args}${function.pc-offset}}}"
#define FILE_AND_LINE "{ at ${line.file.basename}:${line.number}}"

#define DEFAULT_THREAD_FORMAT "thread #${thread.index}: tid = ${thread.id%tid}"\
    "{, ${frame.pc}}"\
    MODULE_WITH_FUNC\
    FILE_AND_LINE\
    "{, name = '${thread.name}'}"\
    "{, queue = '${thread.queue}'}"\
    "{, stop reason = ${thread.stop-reason}}"\
    "{\\nReturn value: ${thread.return-value}}"\
    "\\n"

#define DEFAULT_FRAME_FORMAT "frame #${frame.index}: ${frame.pc}"\
    MODULE_WITH_FUNC\
    FILE_AND_LINE\
    "\\n"



static PropertyDefinition
g_properties[] =
{
{   "auto-confirm",             OptionValue::eTypeBoolean, true, false, NULL, NULL, "If true all confirmation prompts will receive their default reply." },
{   "frame-format",             OptionValue::eTypeString , true, 0    , DEFAULT_FRAME_FORMAT, NULL, "The default frame format string to use when displaying stack frame information for threads." },
{   "notify-void",              OptionValue::eTypeBoolean, true, false, NULL, NULL, "Notify the user explicitly if an expression returns void (default: false)." },
{   "prompt",                   OptionValue::eTypeString , true, OptionValueString::eOptionEncodeCharacterEscapeSequences, "(lldb) ", NULL, "The debugger command line prompt displayed for the user." },
{   "script-lang",              OptionValue::eTypeEnum   , true, eScriptLanguagePython, NULL, g_language_enumerators, "The script language to be used for evaluating user-written scripts." },
{   "stop-disassembly-count",   OptionValue::eTypeSInt64 , true, 4    , NULL, NULL, "The number of disassembly lines to show when displaying a stopped context." },
{   "stop-disassembly-display", OptionValue::eTypeEnum   , true, Debugger::eStopDisassemblyTypeNoSource, NULL, g_show_disassembly_enum_values, "Control when to display disassembly when displaying a stopped context." },
{   "stop-line-count-after",    OptionValue::eTypeSInt64 , true, 3    , NULL, NULL, "The number of sources lines to display that come after the current source line when displaying a stopped context." },
{   "stop-line-count-before",   OptionValue::eTypeSInt64 , true, 3    , NULL, NULL, "The number of sources lines to display that come before the current source line when displaying a stopped context." },
{   "term-width",               OptionValue::eTypeSInt64 , true, 80   , NULL, NULL, "The maximum number of columns to use for displaying text." },
{   "thread-format",            OptionValue::eTypeString , true, 0    , DEFAULT_THREAD_FORMAT, NULL, "The default thread format string to use when displaying thread information." },
{   "use-external-editor",      OptionValue::eTypeBoolean, true, false, NULL, NULL, "Whether to use an external editor or not." },
{   "use-color",                OptionValue::eTypeBoolean, true, true , NULL, NULL, "Whether to use Ansi color codes or not." },
{   "auto-one-line-summaries",     OptionValue::eTypeBoolean, true, true, NULL, NULL, "If true, LLDB will automatically display small structs in one-liner format (default: true)." },

    {   NULL,                       OptionValue::eTypeInvalid, true, 0    , NULL, NULL, NULL }
};

enum
{
    ePropertyAutoConfirm = 0,
    ePropertyFrameFormat,
    ePropertyNotiftVoid,
    ePropertyPrompt,
    ePropertyScriptLanguage,
    ePropertyStopDisassemblyCount,
    ePropertyStopDisassemblyDisplay,
    ePropertyStopLineCountAfter,
    ePropertyStopLineCountBefore,
    ePropertyTerminalWidth,
    ePropertyThreadFormat,
    ePropertyUseExternalEditor,
    ePropertyUseColor,
    ePropertyAutoOneLineSummaries
};

Debugger::LoadPluginCallbackType Debugger::g_load_plugin_callback = NULL;

Error
Debugger::SetPropertyValue (const ExecutionContext *exe_ctx,
                            VarSetOperationType op,
                            const char *property_path,
                            const char *value)
{
    bool is_load_script = strcmp(property_path,"target.load-script-from-symbol-file") == 0;
    TargetSP target_sp;
    LoadScriptFromSymFile load_script_old_value;
    if (is_load_script && exe_ctx->GetTargetSP())
    {
        target_sp = exe_ctx->GetTargetSP();
        load_script_old_value = target_sp->TargetProperties::GetLoadScriptFromSymbolFile();
    }
    Error error (Properties::SetPropertyValue (exe_ctx, op, property_path, value));
    if (error.Success())
    {
        // FIXME it would be nice to have "on-change" callbacks for properties
        if (strcmp(property_path, g_properties[ePropertyPrompt].name) == 0)
        {
            const char *new_prompt = GetPrompt();
            std::string str = lldb_utility::ansi::FormatAnsiTerminalCodes (new_prompt, GetUseColor());
            if (str.length())
                new_prompt = str.c_str();
            GetCommandInterpreter().UpdatePrompt(new_prompt);
            EventSP prompt_change_event_sp (new Event(CommandInterpreter::eBroadcastBitResetPrompt, new EventDataBytes (new_prompt)));
            GetCommandInterpreter().BroadcastEvent (prompt_change_event_sp);
        }
        else if (strcmp(property_path, g_properties[ePropertyUseColor].name) == 0)
        {
			// use-color changed. Ping the prompt so it can reset the ansi terminal codes.
            SetPrompt (GetPrompt());
        }
        else if (is_load_script && target_sp && load_script_old_value == eLoadScriptFromSymFileWarn)
        {
            if (target_sp->TargetProperties::GetLoadScriptFromSymbolFile() == eLoadScriptFromSymFileTrue)
            {
                std::list<Error> errors;
                StreamString feedback_stream;
                if (!target_sp->LoadScriptingResources(errors,&feedback_stream))
                {
                    StreamFileSP stream_sp (GetErrorFile());
                    if (stream_sp)
                    {
                        for (auto error : errors)
                        {
                            stream_sp->Printf("%s\n",error.AsCString());
                        }
                        if (feedback_stream.GetSize())
                            stream_sp->Printf("%s",feedback_stream.GetData());
                    }
                }
            }
        }
    }
    return error;
}

bool
Debugger::GetAutoConfirm () const
{
    const uint32_t idx = ePropertyAutoConfirm;
    return m_collection_sp->GetPropertyAtIndexAsBoolean (NULL, idx, g_properties[idx].default_uint_value != 0);
}

const char *
Debugger::GetFrameFormat() const
{
    const uint32_t idx = ePropertyFrameFormat;
    return m_collection_sp->GetPropertyAtIndexAsString (NULL, idx, g_properties[idx].default_cstr_value);
}

bool
Debugger::GetNotifyVoid () const
{
    const uint32_t idx = ePropertyNotiftVoid;
    return m_collection_sp->GetPropertyAtIndexAsBoolean (NULL, idx, g_properties[idx].default_uint_value != 0);
}

const char *
Debugger::GetPrompt() const
{
    const uint32_t idx = ePropertyPrompt;
    return m_collection_sp->GetPropertyAtIndexAsString (NULL, idx, g_properties[idx].default_cstr_value);
}

void
Debugger::SetPrompt(const char *p)
{
    const uint32_t idx = ePropertyPrompt;
    m_collection_sp->SetPropertyAtIndexAsString (NULL, idx, p);
    const char *new_prompt = GetPrompt();
    std::string str = lldb_utility::ansi::FormatAnsiTerminalCodes (new_prompt, GetUseColor());
    if (str.length())
        new_prompt = str.c_str();
    GetCommandInterpreter().UpdatePrompt(new_prompt);
}

const char *
Debugger::GetThreadFormat() const
{
    const uint32_t idx = ePropertyThreadFormat;
    return m_collection_sp->GetPropertyAtIndexAsString (NULL, idx, g_properties[idx].default_cstr_value);
}

lldb::ScriptLanguage
Debugger::GetScriptLanguage() const
{
    const uint32_t idx = ePropertyScriptLanguage;
    return (lldb::ScriptLanguage)m_collection_sp->GetPropertyAtIndexAsEnumeration (NULL, idx, g_properties[idx].default_uint_value);
}

bool
Debugger::SetScriptLanguage (lldb::ScriptLanguage script_lang)
{
    const uint32_t idx = ePropertyScriptLanguage;
    return m_collection_sp->SetPropertyAtIndexAsEnumeration (NULL, idx, script_lang);
}

uint32_t
Debugger::GetTerminalWidth () const
{
    const uint32_t idx = ePropertyTerminalWidth;
    return m_collection_sp->GetPropertyAtIndexAsSInt64 (NULL, idx, g_properties[idx].default_uint_value);
}

bool
Debugger::SetTerminalWidth (uint32_t term_width)
{
    const uint32_t idx = ePropertyTerminalWidth;
    return m_collection_sp->SetPropertyAtIndexAsSInt64 (NULL, idx, term_width);
}

bool
Debugger::GetUseExternalEditor () const
{
    const uint32_t idx = ePropertyUseExternalEditor;
    return m_collection_sp->GetPropertyAtIndexAsBoolean (NULL, idx, g_properties[idx].default_uint_value != 0);
}

bool
Debugger::SetUseExternalEditor (bool b)
{
    const uint32_t idx = ePropertyUseExternalEditor;
    return m_collection_sp->SetPropertyAtIndexAsBoolean (NULL, idx, b);
}

bool
Debugger::GetUseColor () const
{
    const uint32_t idx = ePropertyUseColor;
    return m_collection_sp->GetPropertyAtIndexAsBoolean (NULL, idx, g_properties[idx].default_uint_value != 0);
}

bool
Debugger::SetUseColor (bool b)
{
    const uint32_t idx = ePropertyUseColor;
    bool ret = m_collection_sp->SetPropertyAtIndexAsBoolean (NULL, idx, b);
    SetPrompt (GetPrompt());
    return ret;
}

uint32_t
Debugger::GetStopSourceLineCount (bool before) const
{
    const uint32_t idx = before ? ePropertyStopLineCountBefore : ePropertyStopLineCountAfter;
    return m_collection_sp->GetPropertyAtIndexAsSInt64 (NULL, idx, g_properties[idx].default_uint_value);
}

Debugger::StopDisassemblyType
Debugger::GetStopDisassemblyDisplay () const
{
    const uint32_t idx = ePropertyStopDisassemblyDisplay;
    return (Debugger::StopDisassemblyType)m_collection_sp->GetPropertyAtIndexAsEnumeration (NULL, idx, g_properties[idx].default_uint_value);
}

uint32_t
Debugger::GetDisassemblyLineCount () const
{
    const uint32_t idx = ePropertyStopDisassemblyCount;
    return m_collection_sp->GetPropertyAtIndexAsSInt64 (NULL, idx, g_properties[idx].default_uint_value);
}

bool
Debugger::GetAutoOneLineSummaries () const
{
    const uint32_t idx = ePropertyAutoOneLineSummaries;
    return m_collection_sp->GetPropertyAtIndexAsBoolean (NULL, idx, true);

}

#pragma mark Debugger

//const DebuggerPropertiesSP &
//Debugger::GetSettings() const
//{
//    return m_properties_sp;
//}
//

int
Debugger::TestDebuggerRefCount ()
{
    return g_shared_debugger_refcount;
}

void
Debugger::Initialize (LoadPluginCallbackType load_plugin_callback)
{
    g_load_plugin_callback = load_plugin_callback;
    if (g_shared_debugger_refcount++ == 0)
        lldb_private::Initialize();
}

void
Debugger::Terminate ()
{
    if (g_shared_debugger_refcount > 0)
    {
        g_shared_debugger_refcount--;
        if (g_shared_debugger_refcount == 0)
        {
            lldb_private::WillTerminate();
            lldb_private::Terminate();

            // Clear our master list of debugger objects
            Mutex::Locker locker (GetDebuggerListMutex ());
            GetDebuggerList().clear();
        }
    }
}

void
Debugger::SettingsInitialize ()
{
    Target::SettingsInitialize ();
}

void
Debugger::SettingsTerminate ()
{
    Target::SettingsTerminate ();
}

bool
Debugger::LoadPlugin (const FileSpec& spec, Error& error)
{
    if (g_load_plugin_callback)
    {
        lldb::DynamicLibrarySP dynlib_sp = g_load_plugin_callback (shared_from_this(), spec, error);
        if (dynlib_sp)
        {
            m_loaded_plugins.push_back(dynlib_sp);
            return true;
        }
    }
    else
    {
        // The g_load_plugin_callback is registered in SBDebugger::Initialize()
        // and if the public API layer isn't available (code is linking against
        // all of the internal LLDB static libraries), then we can't load plugins
        error.SetErrorString("Public API layer is not available");
    }
    return false;
}

static FileSpec::EnumerateDirectoryResult
LoadPluginCallback
(
 void *baton,
 FileSpec::FileType file_type,
 const FileSpec &file_spec
 )
{
    Error error;
    
    static ConstString g_dylibext("dylib");
    static ConstString g_solibext("so");
    
    if (!baton)
        return FileSpec::eEnumerateDirectoryResultQuit;
    
    Debugger *debugger = (Debugger*)baton;
    
    // If we have a regular file, a symbolic link or unknown file type, try
    // and process the file. We must handle unknown as sometimes the directory
    // enumeration might be enumerating a file system that doesn't have correct
    // file type information.
    if (file_type == FileSpec::eFileTypeRegular         ||
        file_type == FileSpec::eFileTypeSymbolicLink    ||
        file_type == FileSpec::eFileTypeUnknown          )
    {
        FileSpec plugin_file_spec (file_spec);
        plugin_file_spec.ResolvePath ();
        
        if (plugin_file_spec.GetFileNameExtension() != g_dylibext &&
            plugin_file_spec.GetFileNameExtension() != g_solibext)
        {
            return FileSpec::eEnumerateDirectoryResultNext;
        }

        Error plugin_load_error;
        debugger->LoadPlugin (plugin_file_spec, plugin_load_error);
        
        return FileSpec::eEnumerateDirectoryResultNext;
    }
    
    else if (file_type == FileSpec::eFileTypeUnknown     ||
        file_type == FileSpec::eFileTypeDirectory   ||
        file_type == FileSpec::eFileTypeSymbolicLink )
    {
        // Try and recurse into anything that a directory or symbolic link.
        // We must also do this for unknown as sometimes the directory enumeration
        // might be enurating a file system that doesn't have correct file type
        // information.
        return FileSpec::eEnumerateDirectoryResultEnter;
    }
    
    return FileSpec::eEnumerateDirectoryResultNext;
}

void
Debugger::InstanceInitialize ()
{
    FileSpec dir_spec;
    const bool find_directories = true;
    const bool find_files = true;
    const bool find_other = true;
    char dir_path[PATH_MAX];
    if (Host::GetLLDBPath (ePathTypeLLDBSystemPlugins, dir_spec))
    {
        if (dir_spec.Exists() && dir_spec.GetPath(dir_path, sizeof(dir_path)))
        {
            FileSpec::EnumerateDirectory (dir_path,
                                          find_directories,
                                          find_files,
                                          find_other,
                                          LoadPluginCallback,
                                          this);
        }
    }
    
    if (Host::GetLLDBPath (ePathTypeLLDBUserPlugins, dir_spec))
    {
        if (dir_spec.Exists() && dir_spec.GetPath(dir_path, sizeof(dir_path)))
        {
            FileSpec::EnumerateDirectory (dir_path,
                                          find_directories,
                                          find_files,
                                          find_other,
                                          LoadPluginCallback,
                                          this);
        }
    }
    
    PluginManager::DebuggerInitialize (*this);
}

DebuggerSP
Debugger::CreateInstance (lldb::LogOutputCallback log_callback, void *baton)
{
    DebuggerSP debugger_sp (new Debugger(log_callback, baton));
    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        GetDebuggerList().push_back(debugger_sp);
    }
    debugger_sp->InstanceInitialize ();
    return debugger_sp;
}

void
Debugger::Destroy (DebuggerSP &debugger_sp)
{
    if (debugger_sp.get() == NULL)
        return;
        
    debugger_sp->Clear();

    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        DebuggerList &debugger_list = GetDebuggerList ();
        DebuggerList::iterator pos, end = debugger_list.end();
        for (pos = debugger_list.begin (); pos != end; ++pos)
        {
            if ((*pos).get() == debugger_sp.get())
            {
                debugger_list.erase (pos);
                return;
            }
        }
    }
}

DebuggerSP
Debugger::FindDebuggerWithInstanceName (const ConstString &instance_name)
{
    DebuggerSP debugger_sp;
    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        DebuggerList &debugger_list = GetDebuggerList();
        DebuggerList::iterator pos, end = debugger_list.end();

        for (pos = debugger_list.begin(); pos != end; ++pos)
        {
            if ((*pos).get()->m_instance_name == instance_name)
            {
                debugger_sp = *pos;
                break;
            }
        }
    }
    return debugger_sp;
}

TargetSP
Debugger::FindTargetWithProcessID (lldb::pid_t pid)
{
    TargetSP target_sp;
    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        DebuggerList &debugger_list = GetDebuggerList();
        DebuggerList::iterator pos, end = debugger_list.end();
        for (pos = debugger_list.begin(); pos != end; ++pos)
        {
            target_sp = (*pos)->GetTargetList().FindTargetWithProcessID (pid);
            if (target_sp)
                break;
        }
    }
    return target_sp;
}

TargetSP
Debugger::FindTargetWithProcess (Process *process)
{
    TargetSP target_sp;
    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        DebuggerList &debugger_list = GetDebuggerList();
        DebuggerList::iterator pos, end = debugger_list.end();
        for (pos = debugger_list.begin(); pos != end; ++pos)
        {
            target_sp = (*pos)->GetTargetList().FindTargetWithProcess (process);
            if (target_sp)
                break;
        }
    }
    return target_sp;
}

Debugger::Debugger (lldb::LogOutputCallback log_callback, void *baton) :
    UserID (g_unique_id++),
    Properties(OptionValuePropertiesSP(new OptionValueProperties())), 
    m_input_file_sp (new StreamFile (stdin, false)),
    m_output_file_sp (new StreamFile (stdout, false)),
    m_error_file_sp (new StreamFile (stderr, false)),
    m_terminal_state (),
    m_target_list (*this),
    m_platform_list (),
    m_listener ("lldb.Debugger"),
    m_source_manager_ap(),
    m_source_file_cache(),
    m_command_interpreter_ap (new CommandInterpreter (*this, eScriptLanguageDefault, false)),
    m_input_reader_stack (),
    m_instance_name (),
    m_loaded_plugins (),
    m_event_handler_thread (LLDB_INVALID_HOST_THREAD),
    m_io_handler_thread (LLDB_INVALID_HOST_THREAD)
{
    char instance_cstr[256];
    snprintf(instance_cstr, sizeof(instance_cstr), "debugger_%d", (int)GetID());
    m_instance_name.SetCString(instance_cstr);
    if (log_callback)
        m_log_callback_stream_sp.reset (new StreamCallback (log_callback, baton));
    m_command_interpreter_ap->Initialize ();
    // Always add our default platform to the platform list
    PlatformSP default_platform_sp (Platform::GetDefaultPlatform());
    assert (default_platform_sp.get());
    m_platform_list.Append (default_platform_sp, true);
    
    m_collection_sp->Initialize (g_properties);
    m_collection_sp->AppendProperty (ConstString("target"),
                                     ConstString("Settings specify to debugging targets."),
                                     true,
                                     Target::GetGlobalProperties()->GetValueProperties());
    if (m_command_interpreter_ap.get())
    {
        m_collection_sp->AppendProperty (ConstString("interpreter"),
                                         ConstString("Settings specify to the debugger's command interpreter."),
                                         true,
                                         m_command_interpreter_ap->GetValueProperties());
    }
    OptionValueSInt64 *term_width = m_collection_sp->GetPropertyAtIndexAsOptionValueSInt64 (NULL, ePropertyTerminalWidth);
    term_width->SetMinimumValue(10);
    term_width->SetMaximumValue(1024);

    // Turn off use-color if this is a dumb terminal.
    const char *term = getenv ("TERM");
    if (term && !strcmp (term, "dumb"))
        SetUseColor (false);
}

Debugger::~Debugger ()
{
    Clear();
}

void
Debugger::Clear()
{
    ClearIOHandlers();
    StopIOHandlerThread();
    StopEventHandlerThread();
    m_listener.Clear();
    int num_targets = m_target_list.GetNumTargets();
    for (int i = 0; i < num_targets; i++)
    {
        TargetSP target_sp (m_target_list.GetTargetAtIndex (i));
        if (target_sp)
        {
            ProcessSP process_sp (target_sp->GetProcessSP());
            if (process_sp)
                process_sp->Finalize();
            target_sp->Destroy();
        }
    }
    BroadcasterManager::Clear ();
    
    // Close the input file _before_ we close the input read communications class
    // as it does NOT own the input file, our m_input_file does.
    m_terminal_state.Clear();
    if (m_input_file_sp)
        m_input_file_sp->GetFile().Close ();
}

bool
Debugger::GetCloseInputOnEOF () const
{
//    return m_input_comm.GetCloseOnEOF();
    return false;
}

void
Debugger::SetCloseInputOnEOF (bool b)
{
//    m_input_comm.SetCloseOnEOF(b);
}

bool
Debugger::GetAsyncExecution ()
{
    return !m_command_interpreter_ap->GetSynchronous();
}

void
Debugger::SetAsyncExecution (bool async_execution)
{
    m_command_interpreter_ap->SetSynchronous (!async_execution);
}

    
void
Debugger::SetInputFileHandle (FILE *fh, bool tranfer_ownership)
{
    if (m_input_file_sp)
        m_input_file_sp->GetFile().SetStream (fh, tranfer_ownership);
    else
        m_input_file_sp.reset (new StreamFile (fh, tranfer_ownership));

    File &in_file = m_input_file_sp->GetFile();
    if (in_file.IsValid() == false)
        in_file.SetStream (stdin, true);

    // Save away the terminal state if that is relevant, so that we can restore it in RestoreInputState.
    SaveInputTerminalState ();
}

void
Debugger::SetOutputFileHandle (FILE *fh, bool tranfer_ownership)
{
    if (m_output_file_sp)
        m_output_file_sp->GetFile().SetStream (fh, tranfer_ownership);
    else
        m_output_file_sp.reset (new StreamFile (fh, tranfer_ownership));
    
    File &out_file = m_output_file_sp->GetFile();
    if (out_file.IsValid() == false)
        out_file.SetStream (stdout, false);
    
    // do not create the ScriptInterpreter just for setting the output file handle
    // as the constructor will know how to do the right thing on its own
    const bool can_create = false;
    ScriptInterpreter* script_interpreter = GetCommandInterpreter().GetScriptInterpreter(can_create);
    if (script_interpreter)
        script_interpreter->ResetOutputFileHandle (fh);
}

void
Debugger::SetErrorFileHandle (FILE *fh, bool tranfer_ownership)
{
    if (m_error_file_sp)
        m_error_file_sp->GetFile().SetStream (fh, tranfer_ownership);
    else
        m_error_file_sp.reset (new StreamFile (fh, tranfer_ownership));
    
    File &err_file = m_error_file_sp->GetFile();
    if (err_file.IsValid() == false)
        err_file.SetStream (stderr, false);
}

void
Debugger::SaveInputTerminalState ()
{
    if (m_input_file_sp)
    {
        File &in_file = m_input_file_sp->GetFile();
        if (in_file.GetDescriptor() != File::kInvalidDescriptor)
            m_terminal_state.Save(in_file.GetDescriptor(), true);
    }
}

void
Debugger::RestoreInputTerminalState ()
{
    m_terminal_state.Restore();
}

ExecutionContext
Debugger::GetSelectedExecutionContext ()
{
    ExecutionContext exe_ctx;
    TargetSP target_sp(GetSelectedTarget());
    exe_ctx.SetTargetSP (target_sp);
    
    if (target_sp)
    {
        ProcessSP process_sp (target_sp->GetProcessSP());
        exe_ctx.SetProcessSP (process_sp);
        if (process_sp && process_sp->IsRunning() == false)
        {
            ThreadSP thread_sp (process_sp->GetThreadList().GetSelectedThread());
            if (thread_sp)
            {
                exe_ctx.SetThreadSP (thread_sp);
                exe_ctx.SetFrameSP (thread_sp->GetSelectedFrame());
                if (exe_ctx.GetFramePtr() == NULL)
                    exe_ctx.SetFrameSP (thread_sp->GetStackFrameAtIndex (0));
            }
        }
    }
    return exe_ctx;
}

void
Debugger::DispatchInputInterrupt ()
{
    Mutex::Locker locker (m_input_reader_stack.GetMutex());
    IOHandlerSP reader_sp (m_input_reader_stack.Top());
    if (reader_sp)
        reader_sp->Interrupt();
}

void
Debugger::DispatchInputEndOfFile ()
{
    Mutex::Locker locker (m_input_reader_stack.GetMutex());
    IOHandlerSP reader_sp (m_input_reader_stack.Top());
    if (reader_sp)
        reader_sp->GotEOF();
}

void
Debugger::ClearIOHandlers ()
{
    // The bottom input reader should be the main debugger input reader.  We do not want to close that one here.
    Mutex::Locker locker (m_input_reader_stack.GetMutex());
    while (m_input_reader_stack.GetSize() > 1)
    {
        IOHandlerSP reader_sp (m_input_reader_stack.Top());
        if (reader_sp)
        {
            m_input_reader_stack.Pop();
            reader_sp->SetIsDone(true);
            reader_sp->Cancel();
        }
    }
}

void
Debugger::ExecuteIOHanders()
{
    
    while (1)
    {
        IOHandlerSP reader_sp(m_input_reader_stack.Top());
        if (!reader_sp)
            break;

        reader_sp->Activate();
        reader_sp->Run();
        reader_sp->Deactivate();

        // Remove all input readers that are done from the top of the stack
        while (1)
        {
            IOHandlerSP top_reader_sp = m_input_reader_stack.Top();
            if (top_reader_sp && top_reader_sp->GetIsDone())
                m_input_reader_stack.Pop();
            else
                break;
        }
    }
    ClearIOHandlers();
}

bool
Debugger::IsTopIOHandler (const lldb::IOHandlerSP& reader_sp)
{
    return m_input_reader_stack.IsTop (reader_sp);
}


ConstString
Debugger::GetTopIOHandlerControlSequence(char ch)
{
    return m_input_reader_stack.GetTopIOHandlerControlSequence (ch);
}

void
Debugger::RunIOHandler (const IOHandlerSP& reader_sp)
{
    Mutex::Locker locker (m_input_reader_stack.GetMutex());
    PushIOHandler (reader_sp);
    reader_sp->Activate();
    reader_sp->Run();
    PopIOHandler (reader_sp);
}

void
Debugger::AdoptTopIOHandlerFilesIfInvalid (StreamFileSP &in, StreamFileSP &out, StreamFileSP &err)
{
    // Before an IOHandler runs, it must have in/out/err streams.
    // This function is called when one ore more of the streams
    // are NULL. We use the top input reader's in/out/err streams,
    // or fall back to the debugger file handles, or we fall back
    // onto stdin/stdout/stderr as a last resort.
    
    Mutex::Locker locker (m_input_reader_stack.GetMutex());
    IOHandlerSP top_reader_sp (m_input_reader_stack.Top());
    // If no STDIN has been set, then set it appropriately
    if (!in)
    {
        if (top_reader_sp)
            in = top_reader_sp->GetInputStreamFile();
        else
            in = GetInputFile();
        
        // If there is nothing, use stdin
        if (!in)
            in = StreamFileSP(new StreamFile(stdin, false));
    }
    // If no STDOUT has been set, then set it appropriately
    if (!out)
    {
        if (top_reader_sp)
            out = top_reader_sp->GetOutputStreamFile();
        else
            out = GetOutputFile();
        
        // If there is nothing, use stdout
        if (!out)
            out = StreamFileSP(new StreamFile(stdout, false));
    }
    // If no STDERR has been set, then set it appropriately
    if (!err)
    {
        if (top_reader_sp)
            err = top_reader_sp->GetErrorStreamFile();
        else
            err = GetErrorFile();
        
        // If there is nothing, use stderr
        if (!err)
            err = StreamFileSP(new StreamFile(stdout, false));
        
    }
}

void
Debugger::PushIOHandler (const IOHandlerSP& reader_sp)
{
    if (!reader_sp)
        return;
 
    // Got the current top input reader...
    IOHandlerSP top_reader_sp (m_input_reader_stack.Top());
    
    // Don't push the same IO handler twice...
    if (reader_sp.get() != top_reader_sp.get())
    {
        // Push our new input reader
        m_input_reader_stack.Push (reader_sp);

        // Interrupt the top input reader to it will exit its Run() function
        // and let this new input reader take over
        if (top_reader_sp)
            top_reader_sp->Deactivate();
    }
}

bool
Debugger::PopIOHandler (const IOHandlerSP& pop_reader_sp)
{
    bool result = false;
    
    Mutex::Locker locker (m_input_reader_stack.GetMutex());

    // The reader on the stop of the stack is done, so let the next
    // read on the stack referesh its prompt and if there is one...
    if (!m_input_reader_stack.IsEmpty())
    {
        IOHandlerSP reader_sp(m_input_reader_stack.Top());
        
        if (!pop_reader_sp || pop_reader_sp.get() == reader_sp.get())
        {
            reader_sp->Deactivate();
            reader_sp->Cancel();
            m_input_reader_stack.Pop ();
            
            reader_sp = m_input_reader_stack.Top();
            if (reader_sp)
                reader_sp->Activate();

            result = true;
        }
    }
    return result;
}

bool
Debugger::HideTopIOHandler()
{
    Mutex::Locker locker;
    
    if (locker.TryLock(m_input_reader_stack.GetMutex()))
    {
        IOHandlerSP reader_sp(m_input_reader_stack.Top());
        if (reader_sp)
            reader_sp->Hide();
        return true;
    }
    return false;
}

void
Debugger::RefreshTopIOHandler()
{
    IOHandlerSP reader_sp(m_input_reader_stack.Top());
    if (reader_sp)
        reader_sp->Refresh();
}


StreamSP
Debugger::GetAsyncOutputStream ()
{
    return StreamSP (new StreamAsynchronousIO (GetCommandInterpreter(),
                                               CommandInterpreter::eBroadcastBitAsynchronousOutputData));
}

StreamSP
Debugger::GetAsyncErrorStream ()
{
    return StreamSP (new StreamAsynchronousIO (GetCommandInterpreter(),
                                               CommandInterpreter::eBroadcastBitAsynchronousErrorData));
}    

size_t
Debugger::GetNumDebuggers()
{
    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        return GetDebuggerList().size();
    }
    return 0;
}

lldb::DebuggerSP
Debugger::GetDebuggerAtIndex (size_t index)
{
    DebuggerSP debugger_sp;
    
    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        DebuggerList &debugger_list = GetDebuggerList();
        
        if (index < debugger_list.size())
            debugger_sp = debugger_list[index];
    }

    return debugger_sp;
}

DebuggerSP
Debugger::FindDebuggerWithID (lldb::user_id_t id)
{
    DebuggerSP debugger_sp;

    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        DebuggerList &debugger_list = GetDebuggerList();
        DebuggerList::iterator pos, end = debugger_list.end();
        for (pos = debugger_list.begin(); pos != end; ++pos)
        {
            if ((*pos).get()->GetID() == id)
            {
                debugger_sp = *pos;
                break;
            }
        }
    }
    return debugger_sp;
}

static void
TestPromptFormats (StackFrame *frame)
{
    if (frame == NULL)
        return;

    StreamString s;
    const char *prompt_format =         
    "{addr = '${addr}'\n}"
    "{process.id = '${process.id}'\n}"
    "{process.name = '${process.name}'\n}"
    "{process.file.basename = '${process.file.basename}'\n}"
    "{process.file.fullpath = '${process.file.fullpath}'\n}"
    "{thread.id = '${thread.id}'\n}"
    "{thread.index = '${thread.index}'\n}"
    "{thread.name = '${thread.name}'\n}"
    "{thread.queue = '${thread.queue}'\n}"
    "{thread.stop-reason = '${thread.stop-reason}'\n}"
    "{target.arch = '${target.arch}'\n}"
    "{module.file.basename = '${module.file.basename}'\n}"
    "{module.file.fullpath = '${module.file.fullpath}'\n}"
    "{file.basename = '${file.basename}'\n}"
    "{file.fullpath = '${file.fullpath}'\n}"
    "{frame.index = '${frame.index}'\n}"
    "{frame.pc = '${frame.pc}'\n}"
    "{frame.sp = '${frame.sp}'\n}"
    "{frame.fp = '${frame.fp}'\n}"
    "{frame.flags = '${frame.flags}'\n}"
    "{frame.reg.rdi = '${frame.reg.rdi}'\n}"
    "{frame.reg.rip = '${frame.reg.rip}'\n}"
    "{frame.reg.rsp = '${frame.reg.rsp}'\n}"
    "{frame.reg.rbp = '${frame.reg.rbp}'\n}"
    "{frame.reg.rflags = '${frame.reg.rflags}'\n}"
    "{frame.reg.xmm0 = '${frame.reg.xmm0}'\n}"
    "{frame.reg.carp = '${frame.reg.carp}'\n}"
    "{function.id = '${function.id}'\n}"
    "{function.name = '${function.name}'\n}"
    "{function.name-with-args = '${function.name-with-args}'\n}"
    "{function.addr-offset = '${function.addr-offset}'\n}"
    "{function.line-offset = '${function.line-offset}'\n}"
    "{function.pc-offset = '${function.pc-offset}'\n}"
    "{line.file.basename = '${line.file.basename}'\n}"
    "{line.file.fullpath = '${line.file.fullpath}'\n}"
    "{line.number = '${line.number}'\n}"
    "{line.start-addr = '${line.start-addr}'\n}"
    "{line.end-addr = '${line.end-addr}'\n}"
;

    SymbolContext sc (frame->GetSymbolContext(eSymbolContextEverything));
    ExecutionContext exe_ctx;
    frame->CalculateExecutionContext(exe_ctx);
    if (Debugger::FormatPrompt (prompt_format, &sc, &exe_ctx, &sc.line_entry.range.GetBaseAddress(), s))
    {
        printf("%s\n", s.GetData());
    }
    else
    {
        printf ("what we got: %s\n", s.GetData());
    }
}

static bool
ScanFormatDescriptor (const char* var_name_begin,
                      const char* var_name_end,
                      const char** var_name_final,
                      const char** percent_position,
                      Format* custom_format,
                      ValueObject::ValueObjectRepresentationStyle* val_obj_display)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
    *percent_position = ::strchr(var_name_begin,'%');
    if (!*percent_position || *percent_position > var_name_end)
    {
        if (log)
            log->Printf("[ScanFormatDescriptor] no format descriptor in string, skipping");
        *var_name_final = var_name_end;
    }
    else
    {
        *var_name_final = *percent_position;
        std::string format_name(*var_name_final+1, var_name_end-*var_name_final-1);
        if (log)
            log->Printf("[ScanFormatDescriptor] parsing %s as a format descriptor", format_name.c_str());
        if ( !FormatManager::GetFormatFromCString(format_name.c_str(),
                                                  true,
                                                  *custom_format) )
        {
            if (log)
                log->Printf("[ScanFormatDescriptor] %s is an unknown format", format_name.c_str());
            
            switch (format_name.front())
            {
                case '@':             // if this is an @ sign, print ObjC description
                    *val_obj_display = ValueObject::eValueObjectRepresentationStyleLanguageSpecific;
                    break;
                case 'V': // if this is a V, print the value using the default format
                    *val_obj_display = ValueObject::eValueObjectRepresentationStyleValue;
                    break;
                case 'L': // if this is an L, print the location of the value
                    *val_obj_display = ValueObject::eValueObjectRepresentationStyleLocation;
                    break;
                case 'S': // if this is an S, print the summary after all
                    *val_obj_display = ValueObject::eValueObjectRepresentationStyleSummary;
                    break;
                case '#': // if this is a '#', print the number of children
                    *val_obj_display = ValueObject::eValueObjectRepresentationStyleChildrenCount;
                    break;
                case 'T': // if this is a 'T', print the type
                    *val_obj_display = ValueObject::eValueObjectRepresentationStyleType;
                    break;
                case 'N': // if this is a 'N', print the name
                    *val_obj_display = ValueObject::eValueObjectRepresentationStyleName;
                    break;
                case '>': // if this is a '>', print the name
                    *val_obj_display = ValueObject::eValueObjectRepresentationStyleExpressionPath;
                    break;
                default:
                    if (log)
                        log->Printf("ScanFormatDescriptor] %s is an error, leaving the previous value alone", format_name.c_str());
                    break;
            }
        }
        // a good custom format tells us to print the value using it
        else
        {
            if (log)
                log->Printf("[ScanFormatDescriptor] will display value for this VO");
            *val_obj_display = ValueObject::eValueObjectRepresentationStyleValue;
        }
    }
    if (log)
        log->Printf("[ScanFormatDescriptor] final format description outcome: custom_format = %d, val_obj_display = %d",
                    *custom_format,
                    *val_obj_display);
    return true;
}

static bool
ScanBracketedRange (const char* var_name_begin,
                    const char* var_name_end,
                    const char* var_name_final,
                    const char** open_bracket_position,
                    const char** separator_position,
                    const char** close_bracket_position,
                    const char** var_name_final_if_array_range,
                    int64_t* index_lower,
                    int64_t* index_higher)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
    *open_bracket_position = ::strchr(var_name_begin,'[');
    if (*open_bracket_position && *open_bracket_position < var_name_final)
    {
        *separator_position = ::strchr(*open_bracket_position,'-'); // might be NULL if this is a simple var[N] bitfield
        *close_bracket_position = ::strchr(*open_bracket_position,']');
        // as usual, we assume that [] will come before %
        //printf("trying to expand a []\n");
        *var_name_final_if_array_range = *open_bracket_position;
        if (*close_bracket_position - *open_bracket_position == 1)
        {
            if (log)
                log->Printf("[ScanBracketedRange] '[]' detected.. going from 0 to end of data");
            *index_lower = 0;
        }
        else if (*separator_position == NULL || *separator_position > var_name_end)
        {
            char *end = NULL;
            *index_lower = ::strtoul (*open_bracket_position+1, &end, 0);
            *index_higher = *index_lower;
            if (log)
                log->Printf("[ScanBracketedRange] [%" PRId64 "] detected, high index is same", *index_lower);
        }
        else if (*close_bracket_position && *close_bracket_position < var_name_end)
        {
            char *end = NULL;
            *index_lower = ::strtoul (*open_bracket_position+1, &end, 0);
            *index_higher = ::strtoul (*separator_position+1, &end, 0);
            if (log)
                log->Printf("[ScanBracketedRange] [%" PRId64 "-%" PRId64 "] detected", *index_lower, *index_higher);
        }
        else
        {
            if (log)
                log->Printf("[ScanBracketedRange] expression is erroneous, cannot extract indices out of it");
            return false;
        }
        if (*index_lower > *index_higher && *index_higher > 0)
        {
            if (log)
                log->Printf("[ScanBracketedRange] swapping indices");
            int64_t temp = *index_lower;
            *index_lower = *index_higher;
            *index_higher = temp;
        }
    }
    else if (log)
            log->Printf("[ScanBracketedRange] no bracketed range, skipping entirely");
    return true;
}

template <typename T>
static bool RunScriptFormatKeyword(Stream &s, ScriptInterpreter *script_interpreter, T t, const std::string& script_name)
{
    if (script_interpreter)
    {
        Error script_error;
        std::string script_output;

        if (script_interpreter->RunScriptFormatKeyword(script_name.c_str(), t, script_output, script_error) && script_error.Success())
        {
            s.Printf("%s", script_output.c_str());
            return true;
        }
        else
        {
            s.Printf("<error: %s>",script_error.AsCString());
        }
    }
    return false;
}

static ValueObjectSP
ExpandIndexedExpression (ValueObject* valobj,
                         size_t index,
                         StackFrame* frame,
                         bool deref_pointer)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
    const char* ptr_deref_format = "[%d]";
    std::string ptr_deref_buffer(10,0);
    ::sprintf(&ptr_deref_buffer[0], ptr_deref_format, index);
    if (log)
        log->Printf("[ExpandIndexedExpression] name to deref: %s",ptr_deref_buffer.c_str());
    const char* first_unparsed;
    ValueObject::GetValueForExpressionPathOptions options;
    ValueObject::ExpressionPathEndResultType final_value_type;
    ValueObject::ExpressionPathScanEndReason reason_to_stop;
    ValueObject::ExpressionPathAftermath what_next = (deref_pointer ? ValueObject::eExpressionPathAftermathDereference : ValueObject::eExpressionPathAftermathNothing);
    ValueObjectSP item = valobj->GetValueForExpressionPath (ptr_deref_buffer.c_str(),
                                                          &first_unparsed,
                                                          &reason_to_stop,
                                                          &final_value_type,
                                                          options,
                                                          &what_next);
    if (!item)
    {
        if (log)
            log->Printf("[ExpandIndexedExpression] ERROR: unparsed portion = %s, why stopping = %d,"
               " final_value_type %d",
               first_unparsed, reason_to_stop, final_value_type);
    }
    else
    {
        if (log)
            log->Printf("[ExpandIndexedExpression] ALL RIGHT: unparsed portion = %s, why stopping = %d,"
               " final_value_type %d",
               first_unparsed, reason_to_stop, final_value_type);
    }
    return item;
}

static inline bool
IsToken(const char *var_name_begin, const char *var)
{
    return (::strncmp (var_name_begin, var, strlen(var)) == 0);
}

static bool
IsTokenWithFormat(const char *var_name_begin, const char *var, std::string &format, const char *default_format,
    const ExecutionContext *exe_ctx_ptr, const SymbolContext *sc_ptr)
{
    int var_len = strlen(var);
    if (::strncmp (var_name_begin, var, var_len) == 0)
    {
        var_name_begin += var_len;
        if (*var_name_begin == '}')
        {
            format = default_format;
            return true;
        }
        else if (*var_name_begin == '%')
        {
            // Allow format specifiers: x|X|u with optional width specifiers.
            //   ${thread.id%x}    ; hex
            //   ${thread.id%X}    ; uppercase hex
            //   ${thread.id%u}    ; unsigned decimal
            //   ${thread.id%8.8X} ; width.precision + specifier
            //   ${thread.id%tid}  ; unsigned on FreeBSD/Linux, otherwise default_format (0x%4.4x for thread.id)
            int dot_count = 0;
            const char *specifier = NULL;
            int width_precision_length = 0;
            const char *width_precision = ++var_name_begin;
            while (isdigit(*var_name_begin) || *var_name_begin == '.')
            {
                dot_count += (*var_name_begin == '.');
                if (dot_count > 1)
                    break;
                var_name_begin++;
                width_precision_length++;
            }

            if (IsToken (var_name_begin, "tid}"))
            {
                Target *target = Target::GetTargetFromContexts (exe_ctx_ptr, sc_ptr);
                if (target)
                {
                    ArchSpec arch (target->GetArchitecture ());
                    llvm::Triple::OSType ostype = arch.IsValid() ? arch.GetTriple().getOS() : llvm::Triple::UnknownOS;
                    if ((ostype == llvm::Triple::FreeBSD) || (ostype == llvm::Triple::Linux))
                        specifier = PRIu64;
                }
                if (!specifier)
                {
                    format = default_format;
                    return true;
                }
            }
            else if (IsToken (var_name_begin, "x}"))
                specifier = PRIx64;
            else if (IsToken (var_name_begin, "X}"))
                specifier = PRIX64;
            else if (IsToken (var_name_begin, "u}"))
                specifier = PRIu64;

            if (specifier)
            {
                format = "%";
                if (width_precision_length)
                    format += std::string(width_precision, width_precision_length);
                format += specifier;
                return true;
            }
        }
    }
    return false;
}

static bool
FormatPromptRecurse
(
    const char *format,
    const SymbolContext *sc,
    const ExecutionContext *exe_ctx,
    const Address *addr,
    Stream &s,
    const char **end,
    ValueObject* valobj
)
{
    ValueObject* realvalobj = NULL; // makes it super-easy to parse pointers
    bool success = true;
    const char *p;
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));

    for (p = format; *p != '\0'; ++p)
    {
        if (realvalobj)
        {
            valobj = realvalobj;
            realvalobj = NULL;
        }
        size_t non_special_chars = ::strcspn (p, "${}\\");
        if (non_special_chars > 0)
        {
            if (success)
                s.Write (p, non_special_chars);
            p += non_special_chars;            
        }

        if (*p == '\0')
        {
            break;
        }
        else if (*p == '{')
        {
            // Start a new scope that must have everything it needs if it is to
            // to make it into the final output stream "s". If you want to make
            // a format that only prints out the function or symbol name if there
            // is one in the symbol context you can use:
            //      "{function =${function.name}}"
            // The first '{' starts a new scope that end with the matching '}' at
            // the end of the string. The contents "function =${function.name}"
            // will then be evaluated and only be output if there is a function
            // or symbol with a valid name. 
            StreamString sub_strm;

            ++p;  // Skip the '{'

            if (FormatPromptRecurse (p, sc, exe_ctx, addr, sub_strm, &p, valobj))
            {
                // The stream had all it needed
                s.Write(sub_strm.GetData(), sub_strm.GetSize());
            }
            if (*p != '}')
            {
                success = false;
                break;
            }
        }
        else if (*p == '}')
        {
            // End of a enclosing scope
            break;
        }
        else if (*p == '$')
        {
            // We have a prompt variable to print
            ++p;
            if (*p == '{')
            {
                ++p;
                const char *var_name_begin = p;
                const char *var_name_end = ::strchr (p, '}');

                if (var_name_end && var_name_begin < var_name_end)
                {
                    // if we have already failed to parse, skip this variable
                    if (success)
                    {
                        const char *cstr = NULL;
                        std::string token_format;
                        Address format_addr;
                        bool calculate_format_addr_function_offset = false;
                        // Set reg_kind and reg_num to invalid values
                        RegisterKind reg_kind = kNumRegisterKinds; 
                        uint32_t reg_num = LLDB_INVALID_REGNUM;
                        FileSpec format_file_spec;
                        const RegisterInfo *reg_info = NULL;
                        RegisterContext *reg_ctx = NULL;
                        bool do_deref_pointer = false;
                        ValueObject::ExpressionPathScanEndReason reason_to_stop = ValueObject::eExpressionPathScanEndReasonEndOfString;
                        ValueObject::ExpressionPathEndResultType final_value_type = ValueObject::eExpressionPathEndResultTypePlain;
                        
                        // Each variable must set success to true below...
                        bool var_success = false;
                        switch (var_name_begin[0])
                        {
                        case '*':
                        case 'v':
                        case 's':
                            {
                                if (!valobj)
                                    break;
                                
                                if (log)
                                    log->Printf("[Debugger::FormatPrompt] initial string: %s",var_name_begin);
                                
                                // check for *var and *svar
                                if (*var_name_begin == '*')
                                {
                                    do_deref_pointer = true;
                                    var_name_begin++;
                                    if (log)
                                        log->Printf("[Debugger::FormatPrompt] found a deref, new string is: %s",var_name_begin);
                                }
                                
                                if (*var_name_begin == 's')
                                {
                                    if (!valobj->IsSynthetic())
                                        valobj = valobj->GetSyntheticValue().get();
                                    if (!valobj)
                                        break;
                                    var_name_begin++;
                                    if (log)
                                        log->Printf("[Debugger::FormatPrompt] found a synthetic, new string is: %s",var_name_begin);
                                }
                                
                                // should be a 'v' by now
                                if (*var_name_begin != 'v')
                                    break;
                                
                                if (log)
                                    log->Printf("[Debugger::FormatPrompt] string I am working with: %s",var_name_begin);
                                                                
                                ValueObject::ExpressionPathAftermath what_next = (do_deref_pointer ?
                                                                                  ValueObject::eExpressionPathAftermathDereference : ValueObject::eExpressionPathAftermathNothing);
                                ValueObject::GetValueForExpressionPathOptions options;
                                options.DontCheckDotVsArrowSyntax().DoAllowBitfieldSyntax().DoAllowFragileIVar().DoAllowSyntheticChildren();
                                ValueObject::ValueObjectRepresentationStyle val_obj_display = ValueObject::eValueObjectRepresentationStyleSummary;
                                ValueObject* target = NULL;
                                Format custom_format = eFormatInvalid;
                                const char* var_name_final = NULL;
                                const char* var_name_final_if_array_range = NULL;
                                const char* close_bracket_position = NULL;
                                int64_t index_lower = -1;
                                int64_t index_higher = -1;
                                bool is_array_range = false;
                                const char* first_unparsed;
                                bool was_plain_var = false;
                                bool was_var_format = false;
                                bool was_var_indexed = false;

                                if (!valobj) break;
                                // simplest case ${var}, just print valobj's value
                                if (IsToken (var_name_begin, "var}"))
                                {
                                    was_plain_var = true;
                                    target = valobj;
                                    val_obj_display = ValueObject::eValueObjectRepresentationStyleValue;
                                }
                                else if (IsToken (var_name_begin,"var%"))
                                {
                                    was_var_format = true;
                                    // this is a variable with some custom format applied to it
                                    const char* percent_position;
                                    target = valobj;
                                    val_obj_display = ValueObject::eValueObjectRepresentationStyleValue;
                                    ScanFormatDescriptor (var_name_begin,
                                                          var_name_end,
                                                          &var_name_final,
                                                          &percent_position,
                                                          &custom_format,
                                                          &val_obj_display);
                                }
                                    // this is ${var.something} or multiple .something nested
                                else if (IsToken (var_name_begin, "var"))
                                {
                                    if (IsToken (var_name_begin, "var["))
                                        was_var_indexed = true;
                                    const char* percent_position;
                                    ScanFormatDescriptor (var_name_begin,
                                                          var_name_end,
                                                          &var_name_final,
                                                          &percent_position,
                                                          &custom_format,
                                                          &val_obj_display);
                                    
                                    const char* open_bracket_position;
                                    const char* separator_position;
                                    ScanBracketedRange (var_name_begin,
                                                        var_name_end,
                                                        var_name_final,
                                                        &open_bracket_position,
                                                        &separator_position,
                                                        &close_bracket_position,
                                                        &var_name_final_if_array_range,
                                                        &index_lower,
                                                        &index_higher);
                                                                    
                                    Error error;
                                    
                                    std::string expr_path(var_name_final-var_name_begin-1,0);
                                    memcpy(&expr_path[0], var_name_begin+3,var_name_final-var_name_begin-3);

                                    if (log)
                                        log->Printf("[Debugger::FormatPrompt] symbol to expand: %s",expr_path.c_str());
                                    
                                    target = valobj->GetValueForExpressionPath(expr_path.c_str(),
                                                                             &first_unparsed,
                                                                             &reason_to_stop,
                                                                             &final_value_type,
                                                                             options,
                                                                             &what_next).get();
                                    
                                    if (!target)
                                    {
                                        if (log)
                                            log->Printf("[Debugger::FormatPrompt] ERROR: unparsed portion = %s, why stopping = %d,"
                                               " final_value_type %d",
                                               first_unparsed, reason_to_stop, final_value_type);
                                        break;
                                    }
                                    else
                                    {
                                        if (log)
                                            log->Printf("[Debugger::FormatPrompt] ALL RIGHT: unparsed portion = %s, why stopping = %d,"
                                               " final_value_type %d",
                                               first_unparsed, reason_to_stop, final_value_type);
                                    }
                                }
                                else
                                    break;
                                
                                is_array_range = (final_value_type == ValueObject::eExpressionPathEndResultTypeBoundedRange ||
                                                  final_value_type == ValueObject::eExpressionPathEndResultTypeUnboundedRange);
                                
                                do_deref_pointer = (what_next == ValueObject::eExpressionPathAftermathDereference);

                                if (do_deref_pointer && !is_array_range)
                                {
                                    // I have not deref-ed yet, let's do it
                                    // this happens when we are not going through GetValueForVariableExpressionPath
                                    // to get to the target ValueObject
                                    Error error;
                                    target = target->Dereference(error).get();
                                    if (error.Fail())
                                    {
                                        if (log)
                                            log->Printf("[Debugger::FormatPrompt] ERROR: %s\n", error.AsCString("unknown")); \
                                        break;
                                    }
                                    do_deref_pointer = false;
                                }
                                
                                // we do not want to use the summary for a bitfield of type T:n
                                // if we were originally dealing with just a T - that would get
                                // us into an endless recursion
                                if (target->IsBitfield() && was_var_indexed)
                                {
                                    // TODO: check for a (T:n)-specific summary - we should still obey that
                                    StreamString bitfield_name;
                                    bitfield_name.Printf("%s:%d", target->GetTypeName().AsCString(), target->GetBitfieldBitSize());
                                    lldb::TypeNameSpecifierImplSP type_sp(new TypeNameSpecifierImpl(bitfield_name.GetData(),false));
                                    if (!DataVisualization::GetSummaryForType(type_sp))
                                        val_obj_display = ValueObject::eValueObjectRepresentationStyleValue;
                                }
                                
                                // TODO use flags for these
                                const uint32_t type_info_flags = target->GetClangType().GetTypeInfo(NULL);
                                bool is_array = (type_info_flags & ClangASTType::eTypeIsArray) != 0;
                                bool is_pointer = (type_info_flags & ClangASTType::eTypeIsPointer) != 0;
                                bool is_aggregate = target->GetClangType().IsAggregateType();
                                
                                if ((is_array || is_pointer) && (!is_array_range) && val_obj_display == ValueObject::eValueObjectRepresentationStyleValue) // this should be wrong, but there are some exceptions
                                {
                                    StreamString str_temp;
                                    if (log)
                                        log->Printf("[Debugger::FormatPrompt] I am into array || pointer && !range");
                                    
                                    if (target->HasSpecialPrintableRepresentation(val_obj_display, custom_format))
                                    {
                                        // try to use the special cases
                                        var_success = target->DumpPrintableRepresentation(str_temp,
                                                                                          val_obj_display,
                                                                                          custom_format);
                                        if (log)
                                            log->Printf("[Debugger::FormatPrompt] special cases did%s match", var_success ? "" : "n't");
                                        
                                        // should not happen
                                        if (var_success)
                                            s << str_temp.GetData();
                                        var_success = true;
                                        break;
                                    }
                                    else
                                    {
                                        if (was_plain_var) // if ${var}
                                        {
                                            s << target->GetTypeName() << " @ " << target->GetLocationAsCString();
                                        }
                                        else if (is_pointer) // if pointer, value is the address stored
                                        {
                                            target->DumpPrintableRepresentation (s,
                                                                                 val_obj_display,
                                                                                 custom_format,
                                                                                 ValueObject::ePrintableRepresentationSpecialCasesDisable);
                                        }
                                        var_success = true;
                                        break;
                                    }
                                }
                                
                                // if directly trying to print ${var}, and this is an aggregate, display a nice
                                // type @ location message
                                if (is_aggregate && was_plain_var)
                                {
                                    s << target->GetTypeName() << " @ " << target->GetLocationAsCString();
                                    var_success = true;
                                    break;
                                }
                                
                                // if directly trying to print ${var%V}, and this is an aggregate, do not let the user do it
                                if (is_aggregate && ((was_var_format && val_obj_display == ValueObject::eValueObjectRepresentationStyleValue)))
                                {
                                    s << "<invalid use of aggregate type>";
                                    var_success = true;
                                    break;
                                }
                                                                
                                if (!is_array_range)
                                {
                                    if (log)
                                        log->Printf("[Debugger::FormatPrompt] dumping ordinary printable output");
                                    var_success = target->DumpPrintableRepresentation(s,val_obj_display, custom_format);
                                }
                                else
                                {   
                                    if (log)
                                        log->Printf("[Debugger::FormatPrompt] checking if I can handle as array");
                                    if (!is_array && !is_pointer)
                                        break;
                                    if (log)
                                        log->Printf("[Debugger::FormatPrompt] handle as array");
                                    const char* special_directions = NULL;
                                    StreamString special_directions_writer;
                                    if (close_bracket_position && (var_name_end-close_bracket_position > 1))
                                    {
                                        ConstString additional_data;
                                        additional_data.SetCStringWithLength(close_bracket_position+1, var_name_end-close_bracket_position-1);
                                        special_directions_writer.Printf("${%svar%s}",
                                                                         do_deref_pointer ? "*" : "",
                                                                         additional_data.GetCString());
                                        special_directions = special_directions_writer.GetData();
                                    }
                                    
                                    // let us display items index_lower thru index_higher of this array
                                    s.PutChar('[');
                                    var_success = true;

                                    if (index_higher < 0)
                                        index_higher = valobj->GetNumChildren() - 1;
                                    
                                    uint32_t max_num_children = target->GetTargetSP()->GetMaximumNumberOfChildrenToDisplay();
                                    
                                    for (;index_lower<=index_higher;index_lower++)
                                    {
                                        ValueObject* item = ExpandIndexedExpression (target,
                                                                                     index_lower,
                                                                                     exe_ctx->GetFramePtr(),
                                                                                     false).get();
                                        
                                        if (!item)
                                        {
                                            if (log)
                                                log->Printf("[Debugger::FormatPrompt] ERROR in getting child item at index %" PRId64, index_lower);
                                        }
                                        else
                                        {
                                            if (log)
                                                log->Printf("[Debugger::FormatPrompt] special_directions for child item: %s",special_directions);
                                        }

                                        if (!special_directions)
                                            var_success &= item->DumpPrintableRepresentation(s,val_obj_display, custom_format);
                                        else
                                            var_success &= FormatPromptRecurse(special_directions, sc, exe_ctx, addr, s, NULL, item);
                                        
                                        if (--max_num_children == 0)
                                        {
                                            s.PutCString(", ...");
                                            break;
                                        }
                                        
                                        if (index_lower < index_higher)
                                            s.PutChar(',');
                                    }
                                    s.PutChar(']');
                                }
                            }
                            break;
                        case 'a':
                            if (IsToken (var_name_begin, "addr}"))
                            {
                                if (addr && addr->IsValid())
                                {
                                    var_success = true;
                                    format_addr = *addr;
                                }
                            }
                            break;

                        case 'p':
                            if (IsToken (var_name_begin, "process."))
                            {
                                if (exe_ctx)
                                {
                                    Process *process = exe_ctx->GetProcessPtr();
                                    if (process)
                                    {
                                        var_name_begin += ::strlen ("process.");
                                        if (IsTokenWithFormat (var_name_begin, "id", token_format, "%" PRIu64, exe_ctx, sc))
                                        {
                                            s.Printf(token_format.c_str(), process->GetID());
                                            var_success = true;
                                        }
                                        else if ((IsToken (var_name_begin, "name}")) ||
                                                (IsToken (var_name_begin, "file.basename}")) ||
                                                (IsToken (var_name_begin, "file.fullpath}")))
                                        {
                                            Module *exe_module = process->GetTarget().GetExecutableModulePointer();
                                            if (exe_module)
                                            {
                                                if (var_name_begin[0] == 'n' || var_name_begin[5] == 'f')
                                                {
                                                    format_file_spec.GetFilename() = exe_module->GetFileSpec().GetFilename();
                                                    var_success = (bool)format_file_spec;
                                                }
                                                else
                                                {
                                                    format_file_spec = exe_module->GetFileSpec();
                                                    var_success = (bool)format_file_spec;
                                                }
                                            }
                                        }
                                        else if (IsToken (var_name_begin, "script:"))
                                        {
                                            var_name_begin += ::strlen("script:");
                                            std::string script_name(var_name_begin,var_name_end);
                                            ScriptInterpreter* script_interpreter = process->GetTarget().GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
                                            if (RunScriptFormatKeyword (s, script_interpreter, process, script_name))
                                                var_success = true;
                                        }
                                    }
                                }
                            }
                            break;
                        
                        case 't':
                           if (IsToken (var_name_begin, "thread."))
                            {
                                if (exe_ctx)
                                {
                                    Thread *thread = exe_ctx->GetThreadPtr();
                                    if (thread)
                                    {
                                        var_name_begin += ::strlen ("thread.");
                                        if (IsTokenWithFormat (var_name_begin, "id", token_format, "0x%4.4" PRIx64, exe_ctx, sc))
                                        {
                                            s.Printf(token_format.c_str(), thread->GetID());
                                            var_success = true;
                                        }
                                        else if (IsTokenWithFormat (var_name_begin, "protocol_id", token_format, "0x%4.4" PRIx64, exe_ctx, sc))
                                        {
                                            s.Printf(token_format.c_str(), thread->GetProtocolID());
                                            var_success = true;
                                        }
                                        else if (IsTokenWithFormat (var_name_begin, "index", token_format, "%" PRIu64, exe_ctx, sc))
                                        {
                                            s.Printf(token_format.c_str(), (uint64_t)thread->GetIndexID());
                                            var_success = true;
                                        }
                                        else if (IsToken (var_name_begin, "name}"))
                                        {
                                            cstr = thread->GetName();
                                            var_success = cstr && cstr[0];
                                            if (var_success)
                                                s.PutCString(cstr);
                                        }
                                        else if (IsToken (var_name_begin, "queue}"))
                                        {
                                            cstr = thread->GetQueueName();
                                            var_success = cstr && cstr[0];
                                            if (var_success)
                                                s.PutCString(cstr);
                                        }
                                        else if (IsToken (var_name_begin, "stop-reason}"))
                                        {
                                            StopInfoSP stop_info_sp = thread->GetStopInfo ();
                                            if (stop_info_sp && stop_info_sp->IsValid())
                                            {
                                                cstr = stop_info_sp->GetDescription();
                                                if (cstr && cstr[0])
                                                {
                                                    s.PutCString(cstr);
                                                    var_success = true;
                                                }
                                            }
                                        }
                                        else if (IsToken (var_name_begin, "return-value}"))
                                        {
                                            StopInfoSP stop_info_sp = thread->GetStopInfo ();
                                            if (stop_info_sp && stop_info_sp->IsValid())
                                            {
                                                ValueObjectSP return_valobj_sp = StopInfo::GetReturnValueObject (stop_info_sp);
                                                if (return_valobj_sp)
                                                {
                                                    return_valobj_sp->Dump(s);
                                                    var_success = true;
                                                }
                                            }
                                        }
                                        else if (IsToken (var_name_begin, "script:"))
                                        {
                                            var_name_begin += ::strlen("script:");
                                            std::string script_name(var_name_begin,var_name_end);
                                            ScriptInterpreter* script_interpreter = thread->GetProcess()->GetTarget().GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
                                            if (RunScriptFormatKeyword (s, script_interpreter, thread, script_name))
                                                var_success = true;
                                        }
                                    }
                                }
                            }
                            else if (IsToken (var_name_begin, "target."))
                            {
                                // TODO: hookup properties
//                                if (!target_properties_sp)
//                                {
//                                    Target *target = Target::GetTargetFromContexts (exe_ctx, sc);
//                                    if (target)
//                                        target_properties_sp = target->GetProperties();
//                                }
//
//                                if (target_properties_sp)
//                                {
//                                    var_name_begin += ::strlen ("target.");
//                                    const char *end_property = strchr(var_name_begin, '}');
//                                    if (end_property)
//                                    {
//                                        ConstString property_name(var_name_begin, end_property - var_name_begin);
//                                        std::string property_value (target_properties_sp->GetPropertyValue(property_name));
//                                        if (!property_value.empty())
//                                        {
//                                            s.PutCString (property_value.c_str());
//                                            var_success = true;
//                                        }
//                                    }
//                                }                                        
                                Target *target = Target::GetTargetFromContexts (exe_ctx, sc);
                                if (target)
                                {
                                    var_name_begin += ::strlen ("target.");
                                    if (IsToken (var_name_begin, "arch}"))
                                    {
                                        ArchSpec arch (target->GetArchitecture ());
                                        if (arch.IsValid())
                                        {
                                            s.PutCString (arch.GetArchitectureName());
                                            var_success = true;
                                        }
                                    }
                                    else if (IsToken (var_name_begin, "script:"))
                                    {
                                        var_name_begin += ::strlen("script:");
                                        std::string script_name(var_name_begin,var_name_end);
                                        ScriptInterpreter* script_interpreter = target->GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
                                        if (RunScriptFormatKeyword (s, script_interpreter, target, script_name))
                                            var_success = true;
                                    }
                                }
                            }
                            break;
                            
                            
                        case 'm':
                           if (IsToken (var_name_begin, "module."))
                            {
                                if (sc && sc->module_sp.get())
                                {
                                    Module *module = sc->module_sp.get();
                                    var_name_begin += ::strlen ("module.");
                                    
                                    if (IsToken (var_name_begin, "file."))
                                    {
                                        if (module->GetFileSpec())
                                        {
                                            var_name_begin += ::strlen ("file.");
                                            
                                            if (IsToken (var_name_begin, "basename}"))
                                            {
                                                format_file_spec.GetFilename() = module->GetFileSpec().GetFilename();
                                                var_success = (bool)format_file_spec;
                                            }
                                            else if (IsToken (var_name_begin, "fullpath}"))
                                            {
                                                format_file_spec = module->GetFileSpec();
                                                var_success = (bool)format_file_spec;
                                            }
                                        }
                                    }
                                }
                            }
                            break;
                            
                        
                        case 'f':
                           if (IsToken (var_name_begin, "file."))
                            {
                                if (sc && sc->comp_unit != NULL)
                                {
                                    var_name_begin += ::strlen ("file.");
                                    
                                    if (IsToken (var_name_begin, "basename}"))
                                    {
                                        format_file_spec.GetFilename() = sc->comp_unit->GetFilename();
                                        var_success = (bool)format_file_spec;
                                    }
                                    else if (IsToken (var_name_begin, "fullpath}"))
                                    {
                                        format_file_spec = *sc->comp_unit;
                                        var_success = (bool)format_file_spec;
                                    }
                                }
                            }
                           else if (IsToken (var_name_begin, "frame."))
                            {
                                if (exe_ctx)
                                {
                                    StackFrame *frame = exe_ctx->GetFramePtr();
                                    if (frame)
                                    {
                                        var_name_begin += ::strlen ("frame.");
                                        if (IsToken (var_name_begin, "index}"))
                                        {
                                            s.Printf("%u", frame->GetFrameIndex());
                                            var_success = true;
                                        }
                                        else if (IsToken (var_name_begin, "pc}"))
                                        {
                                            reg_kind = eRegisterKindGeneric;
                                            reg_num = LLDB_REGNUM_GENERIC_PC;
                                            var_success = true;
                                        }
                                        else if (IsToken (var_name_begin, "sp}"))
                                        {
                                            reg_kind = eRegisterKindGeneric;
                                            reg_num = LLDB_REGNUM_GENERIC_SP;
                                            var_success = true;
                                        }
                                        else if (IsToken (var_name_begin, "fp}"))
                                        {
                                            reg_kind = eRegisterKindGeneric;
                                            reg_num = LLDB_REGNUM_GENERIC_FP;
                                            var_success = true;
                                        }
                                        else if (IsToken (var_name_begin, "flags}"))
                                        {
                                            reg_kind = eRegisterKindGeneric;
                                            reg_num = LLDB_REGNUM_GENERIC_FLAGS;
                                            var_success = true;
                                        }
                                        else if (IsToken (var_name_begin, "reg."))
                                        {
                                            reg_ctx = frame->GetRegisterContext().get();
                                            if (reg_ctx)
                                            {
                                                var_name_begin += ::strlen ("reg.");
                                                if (var_name_begin < var_name_end)
                                                {
                                                    std::string reg_name (var_name_begin, var_name_end);
                                                    reg_info = reg_ctx->GetRegisterInfoByName (reg_name.c_str());
                                                    if (reg_info)
                                                        var_success = true;
                                                }
                                            }
                                        }
                                        else if (IsToken (var_name_begin, "script:"))
                                        {
                                            var_name_begin += ::strlen("script:");
                                            std::string script_name(var_name_begin,var_name_end);
                                            ScriptInterpreter* script_interpreter = frame->GetThread()->GetProcess()->GetTarget().GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
                                            if (RunScriptFormatKeyword (s, script_interpreter, frame, script_name))
                                                var_success = true;
                                        }
                                    }
                                }
                            }
                            else if (IsToken (var_name_begin, "function."))
                            {
                                if (sc && (sc->function != NULL || sc->symbol != NULL))
                                {
                                    var_name_begin += ::strlen ("function.");
                                    if (IsToken (var_name_begin, "id}"))
                                    {
                                        if (sc->function)
                                            s.Printf("function{0x%8.8" PRIx64 "}", sc->function->GetID());
                                        else
                                            s.Printf("symbol[%u]", sc->symbol->GetID());

                                        var_success = true;
                                    }
                                    else if (IsToken (var_name_begin, "name}"))
                                    {
                                        if (sc->function)
                                            cstr = sc->function->GetName().AsCString (NULL);
                                        else if (sc->symbol)
                                            cstr = sc->symbol->GetName().AsCString (NULL);
                                        if (cstr)
                                        {
                                            s.PutCString(cstr);
                                            
                                            if (sc->block)
                                            {
                                                Block *inline_block = sc->block->GetContainingInlinedBlock ();
                                                if (inline_block)
                                                {
                                                    const InlineFunctionInfo *inline_info = sc->block->GetInlinedFunctionInfo();
                                                    if (inline_info)
                                                    {
                                                        s.PutCString(" [inlined] ");
                                                        inline_info->GetName().Dump(&s);
                                                    }
                                                }
                                            }
                                            var_success = true;
                                        }
                                    }
                                    else if (IsToken (var_name_begin, "name-with-args}"))
                                    {
                                        // Print the function name with arguments in it

                                        if (sc->function)
                                        {
                                            var_success = true;
                                            ExecutionContextScope *exe_scope = exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL;
                                            cstr = sc->function->GetName().AsCString (NULL);
                                            if (cstr)
                                            {
                                                const InlineFunctionInfo *inline_info = NULL;
                                                VariableListSP variable_list_sp;
                                                bool get_function_vars = true;
                                                if (sc->block)
                                                {
                                                    Block *inline_block = sc->block->GetContainingInlinedBlock ();

                                                    if (inline_block)
                                                    {
                                                        get_function_vars = false;
                                                        inline_info = sc->block->GetInlinedFunctionInfo();
                                                        if (inline_info)
                                                            variable_list_sp = inline_block->GetBlockVariableList (true);
                                                    }
                                                }
                                                
                                                if (get_function_vars)
                                                {
                                                    variable_list_sp = sc->function->GetBlock(true).GetBlockVariableList (true);
                                                }
                                                
                                                if (inline_info)
                                                {
                                                    s.PutCString (cstr);
                                                    s.PutCString (" [inlined] ");
                                                    cstr = inline_info->GetName().GetCString();
                                                }
                                                
                                                VariableList args;
                                                if (variable_list_sp)
                                                    variable_list_sp->AppendVariablesWithScope(eValueTypeVariableArgument, args);
                                                if (args.GetSize() > 0)
                                                {
                                                    const char *open_paren = strchr (cstr, '(');
                                                    const char *close_paren = NULL;
                                                    if (open_paren)
                                                    {
                                                        if (IsToken (open_paren, "(anonymous namespace)"))
                                                        {
                                                            open_paren = strchr (open_paren + strlen("(anonymous namespace)"), '(');
                                                            if (open_paren)
                                                                close_paren = strchr (open_paren, ')');
                                                        }
                                                        else
                                                            close_paren = strchr (open_paren, ')');
                                                    }
                                                    
                                                    if (open_paren)
                                                        s.Write(cstr, open_paren - cstr + 1);
                                                    else
                                                    {
                                                        s.PutCString (cstr);
                                                        s.PutChar ('(');
                                                    }
                                                    const size_t num_args = args.GetSize();
                                                    for (size_t arg_idx = 0; arg_idx < num_args; ++arg_idx)
                                                    {
                                                        VariableSP var_sp (args.GetVariableAtIndex (arg_idx));
                                                        ValueObjectSP var_value_sp (ValueObjectVariable::Create (exe_scope, var_sp));
                                                        const char *var_name = var_value_sp->GetName().GetCString();
                                                        const char *var_value = var_value_sp->GetValueAsCString();
                                                        if (arg_idx > 0)
                                                            s.PutCString (", ");
                                                        if (var_value_sp->GetError().Success())
                                                        {
                                                            if (var_value)
                                                                s.Printf ("%s=%s", var_name, var_value);
                                                            else
                                                                s.Printf ("%s=%s at %s", var_name, var_value_sp->GetTypeName().GetCString(), var_value_sp->GetLocationAsCString());
                                                        }
                                                        else
                                                            s.Printf ("%s=<unavailable>", var_name);
                                                    }
                                                    
                                                    if (close_paren)
                                                        s.PutCString (close_paren);
                                                    else
                                                        s.PutChar(')');

                                                }
                                                else
                                                {
                                                    s.PutCString(cstr);
                                                }
                                            }
                                        }
                                        else if (sc->symbol)
                                        {
                                            cstr = sc->symbol->GetName().AsCString (NULL);
                                            if (cstr)
                                            {
                                                s.PutCString(cstr);
                                                var_success = true;
                                            }
                                        }
                                    }
                                    else if (IsToken (var_name_begin, "addr-offset}"))
                                    {
                                        var_success = addr != NULL;
                                        if (var_success)
                                        {
                                            format_addr = *addr;
                                            calculate_format_addr_function_offset = true;
                                        }
                                    }
                                    else if (IsToken (var_name_begin, "line-offset}"))
                                    {
                                        var_success = sc->line_entry.range.GetBaseAddress().IsValid();
                                        if (var_success)
                                        {
                                            format_addr = sc->line_entry.range.GetBaseAddress();
                                            calculate_format_addr_function_offset = true;
                                        }
                                    }
                                    else if (IsToken (var_name_begin, "pc-offset}"))
                                    {
                                        StackFrame *frame = exe_ctx->GetFramePtr();
                                        var_success = frame != NULL;
                                        if (var_success)
                                        {
                                            format_addr = frame->GetFrameCodeAddress();
                                            calculate_format_addr_function_offset = true;
                                        }
                                    }
                                }
                            }
                            break;

                        case 'l':
                            if (IsToken (var_name_begin, "line."))
                            {
                                if (sc && sc->line_entry.IsValid())
                                {
                                    var_name_begin += ::strlen ("line.");
                                    if (IsToken (var_name_begin, "file."))
                                    {
                                        var_name_begin += ::strlen ("file.");
                                        
                                        if (IsToken (var_name_begin, "basename}"))
                                        {
                                            format_file_spec.GetFilename() = sc->line_entry.file.GetFilename();
                                            var_success = (bool)format_file_spec;
                                        }
                                        else if (IsToken (var_name_begin, "fullpath}"))
                                        {
                                            format_file_spec = sc->line_entry.file;
                                            var_success = (bool)format_file_spec;
                                        }
                                    }
                                    else if (IsTokenWithFormat (var_name_begin, "number", token_format, "%" PRIu64, exe_ctx, sc))
                                    {
                                        var_success = true;
                                        s.Printf(token_format.c_str(), (uint64_t)sc->line_entry.line);
                                    }
                                    else if ((IsToken (var_name_begin, "start-addr}")) ||
                                             (IsToken (var_name_begin, "end-addr}")))
                                    {
                                        var_success = sc && sc->line_entry.range.GetBaseAddress().IsValid();
                                        if (var_success)
                                        {
                                            format_addr = sc->line_entry.range.GetBaseAddress();
                                            if (var_name_begin[0] == 'e')
                                                format_addr.Slide (sc->line_entry.range.GetByteSize());
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        
                        if (var_success)
                        {
                            // If format addr is valid, then we need to print an address
                            if (reg_num != LLDB_INVALID_REGNUM)
                            {
                                StackFrame *frame = exe_ctx->GetFramePtr();
                                // We have a register value to display...
                                if (reg_num == LLDB_REGNUM_GENERIC_PC && reg_kind == eRegisterKindGeneric)
                                {
                                    format_addr = frame->GetFrameCodeAddress();
                                }
                                else
                                {
                                    if (reg_ctx == NULL)
                                        reg_ctx = frame->GetRegisterContext().get();

                                    if (reg_ctx)
                                    {
                                        if (reg_kind != kNumRegisterKinds)
                                            reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber(reg_kind, reg_num);
                                        reg_info = reg_ctx->GetRegisterInfoAtIndex (reg_num);
                                        var_success = reg_info != NULL;
                                    }
                                }
                            }
                            
                            if (reg_info != NULL)
                            {
                                RegisterValue reg_value;
                                var_success = reg_ctx->ReadRegister (reg_info, reg_value);
                                if (var_success)
                                {
                                    reg_value.Dump(&s, reg_info, false, false, eFormatDefault);
                                }
                            }                            
                            
                            if (format_file_spec)
                            {
                                s << format_file_spec;
                            }

                            // If format addr is valid, then we need to print an address
                            if (format_addr.IsValid())
                            {
                                var_success = false;

                                if (calculate_format_addr_function_offset)
                                {
                                    Address func_addr;
                                    
                                    if (sc)
                                    {
                                        if (sc->function)
                                        {
                                            func_addr = sc->function->GetAddressRange().GetBaseAddress();
                                            if (sc->block)
                                            {
                                                // Check to make sure we aren't in an inline
                                                // function. If we are, use the inline block
                                                // range that contains "format_addr" since
                                                // blocks can be discontiguous.
                                                Block *inline_block = sc->block->GetContainingInlinedBlock ();
                                                AddressRange inline_range;
                                                if (inline_block && inline_block->GetRangeContainingAddress (format_addr, inline_range))
                                                    func_addr = inline_range.GetBaseAddress();
                                            }
                                        }
                                        else if (sc->symbol && sc->symbol->ValueIsAddress())
                                            func_addr = sc->symbol->GetAddress();
                                    }
                                    
                                    if (func_addr.IsValid())
                                    {
                                        if (func_addr.GetSection() == format_addr.GetSection())
                                        {
                                            addr_t func_file_addr = func_addr.GetFileAddress();
                                            addr_t addr_file_addr = format_addr.GetFileAddress();
                                            if (addr_file_addr > func_file_addr)
                                                s.Printf(" + %" PRIu64, addr_file_addr - func_file_addr);
                                            else if (addr_file_addr < func_file_addr)
                                                s.Printf(" - %" PRIu64, func_file_addr - addr_file_addr);
                                            var_success = true;
                                        }
                                        else
                                        {
                                            Target *target = Target::GetTargetFromContexts (exe_ctx, sc);
                                            if (target)
                                            {
                                                addr_t func_load_addr = func_addr.GetLoadAddress (target);
                                                addr_t addr_load_addr = format_addr.GetLoadAddress (target);
                                                if (addr_load_addr > func_load_addr)
                                                    s.Printf(" + %" PRIu64, addr_load_addr - func_load_addr);
                                                else if (addr_load_addr < func_load_addr)
                                                    s.Printf(" - %" PRIu64, func_load_addr - addr_load_addr);
                                                var_success = true;
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    Target *target = Target::GetTargetFromContexts (exe_ctx, sc);
                                    addr_t vaddr = LLDB_INVALID_ADDRESS;
                                    if (exe_ctx && !target->GetSectionLoadList().IsEmpty())
                                        vaddr = format_addr.GetLoadAddress (target);
                                    if (vaddr == LLDB_INVALID_ADDRESS)
                                        vaddr = format_addr.GetFileAddress ();

                                    if (vaddr != LLDB_INVALID_ADDRESS)
                                    {
                                        int addr_width = target->GetArchitecture().GetAddressByteSize() * 2;
                                        if (addr_width == 0)
                                            addr_width = 16;
                                        s.Printf("0x%*.*" PRIx64, addr_width, addr_width, vaddr);
                                        var_success = true;
                                    }
                                }
                            }
                        }

                        if (var_success == false)
                            success = false;
                    }
                    p = var_name_end;
                }
                else
                    break;
            }
            else
            {
                // We got a dollar sign with no '{' after it, it must just be a dollar sign
                s.PutChar(*p);
            }
        }
        else if (*p == '\\')
        {
            ++p; // skip the slash
            switch (*p)
            {
            case 'a': s.PutChar ('\a'); break;
            case 'b': s.PutChar ('\b'); break;
            case 'f': s.PutChar ('\f'); break;
            case 'n': s.PutChar ('\n'); break;
            case 'r': s.PutChar ('\r'); break;
            case 't': s.PutChar ('\t'); break;
            case 'v': s.PutChar ('\v'); break;
            case '\'': s.PutChar ('\''); break; 
            case '\\': s.PutChar ('\\'); break; 
            case '0':
                // 1 to 3 octal chars
                {
                    // Make a string that can hold onto the initial zero char,
                    // up to 3 octal digits, and a terminating NULL.
                    char oct_str[5] = { 0, 0, 0, 0, 0 };

                    int i;
                    for (i=0; (p[i] >= '0' && p[i] <= '7') && i<4; ++i)
                        oct_str[i] = p[i];

                    // We don't want to consume the last octal character since
                    // the main for loop will do this for us, so we advance p by
                    // one less than i (even if i is zero)
                    p += i - 1;
                    unsigned long octal_value = ::strtoul (oct_str, NULL, 8);
                    if (octal_value <= UINT8_MAX)
                    {
                        s.PutChar((char)octal_value);
                    }
                }
                break;

            case 'x':
                // hex number in the format 
                if (isxdigit(p[1]))
                {
                    ++p;    // Skip the 'x'

                    // Make a string that can hold onto two hex chars plus a
                    // NULL terminator
                    char hex_str[3] = { 0,0,0 };
                    hex_str[0] = *p;
                    if (isxdigit(p[1]))
                    {
                        ++p; // Skip the first of the two hex chars
                        hex_str[1] = *p;
                    }

                    unsigned long hex_value = strtoul (hex_str, NULL, 16);                    
                    if (hex_value <= UINT8_MAX)
                        s.PutChar ((char)hex_value);
                }
                else
                {
                    s.PutChar('x');
                }
                break;
                
            default:
                // Just desensitize any other character by just printing what
                // came after the '\'
                s << *p;
                break;
            
            }

        }
    }
    if (end) 
        *end = p;
    return success;
}

bool
Debugger::FormatPrompt
(
    const char *format,
    const SymbolContext *sc,
    const ExecutionContext *exe_ctx,
    const Address *addr,
    Stream &s,
    ValueObject* valobj
)
{
    bool use_color = exe_ctx ? exe_ctx->GetTargetRef().GetDebugger().GetUseColor() : true;
    std::string format_str = lldb_utility::ansi::FormatAnsiTerminalCodes (format, use_color);
    if (format_str.length())
        format = format_str.c_str();
    return FormatPromptRecurse (format, sc, exe_ctx, addr, s, NULL, valobj);
}

void
Debugger::SetLoggingCallback (lldb::LogOutputCallback log_callback, void *baton)
{
    // For simplicity's sake, I am not going to deal with how to close down any
    // open logging streams, I just redirect everything from here on out to the
    // callback.
    m_log_callback_stream_sp.reset (new StreamCallback (log_callback, baton));
}

bool
Debugger::EnableLog (const char *channel, const char **categories, const char *log_file, uint32_t log_options, Stream &error_stream)
{
    Log::Callbacks log_callbacks;

    StreamSP log_stream_sp;
    if (m_log_callback_stream_sp)
    {
        log_stream_sp = m_log_callback_stream_sp;
        // For now when using the callback mode you always get thread & timestamp.
        log_options |= LLDB_LOG_OPTION_PREPEND_TIMESTAMP | LLDB_LOG_OPTION_PREPEND_THREAD_NAME;
    }
    else if (log_file == NULL || *log_file == '\0')
    {
        log_stream_sp = GetOutputFile();
    }
    else
    {
        LogStreamMap::iterator pos = m_log_streams.find(log_file);
        if (pos != m_log_streams.end())
            log_stream_sp = pos->second.lock();
        if (!log_stream_sp)
        {
            log_stream_sp.reset (new StreamFile (log_file));
            m_log_streams[log_file] = log_stream_sp;
        }
    }
    assert (log_stream_sp.get());
    
    if (log_options == 0)
        log_options = LLDB_LOG_OPTION_PREPEND_THREAD_NAME | LLDB_LOG_OPTION_THREADSAFE;
        
    if (Log::GetLogChannelCallbacks (ConstString(channel), log_callbacks))
    {
        log_callbacks.enable (log_stream_sp, log_options, categories, &error_stream);
        return true;
    }
    else
    {
        LogChannelSP log_channel_sp (LogChannel::FindPlugin (channel));
        if (log_channel_sp)
        {
            if (log_channel_sp->Enable (log_stream_sp, log_options, &error_stream, categories))
            {
                return true;
            }
            else
            {
                error_stream.Printf ("Invalid log channel '%s'.\n", channel);
                return false;
            }
        }
        else
        {
            error_stream.Printf ("Invalid log channel '%s'.\n", channel);
            return false;
        }
    }
    return false;
}

SourceManager &
Debugger::GetSourceManager ()
{
    if (m_source_manager_ap.get() == NULL)
        m_source_manager_ap.reset (new SourceManager (shared_from_this()));
    return *m_source_manager_ap;
}



// This function handles events that were broadcast by the process.
void
Debugger::HandleBreakpointEvent (const EventSP &event_sp)
{
    using namespace lldb;
    const uint32_t event_type = Breakpoint::BreakpointEventData::GetBreakpointEventTypeFromEvent (event_sp);
    
//    if (event_type & eBreakpointEventTypeAdded
//        || event_type & eBreakpointEventTypeRemoved
//        || event_type & eBreakpointEventTypeEnabled
//        || event_type & eBreakpointEventTypeDisabled
//        || event_type & eBreakpointEventTypeCommandChanged
//        || event_type & eBreakpointEventTypeConditionChanged
//        || event_type & eBreakpointEventTypeIgnoreChanged
//        || event_type & eBreakpointEventTypeLocationsResolved)
//    {
//        // Don't do anything about these events, since the breakpoint commands already echo these actions.
//    }
//    
    if (event_type & eBreakpointEventTypeLocationsAdded)
    {
        uint32_t num_new_locations = Breakpoint::BreakpointEventData::GetNumBreakpointLocationsFromEvent(event_sp);
        if (num_new_locations > 0)
        {
            BreakpointSP breakpoint = Breakpoint::BreakpointEventData::GetBreakpointFromEvent(event_sp);
            StreamFileSP output_sp (GetOutputFile());
            if (output_sp)
            {
                output_sp->Printf("%d location%s added to breakpoint %d\n",
                                  num_new_locations,
                                  num_new_locations == 1 ? "" : "s",
                                  breakpoint->GetID());
                RefreshTopIOHandler();
            }
        }
    }
//    else if (event_type & eBreakpointEventTypeLocationsRemoved)
//    {
//        // These locations just get disabled, not sure it is worth spamming folks about this on the command line.
//    }
//    else if (event_type & eBreakpointEventTypeLocationsResolved)
//    {
//        // This might be an interesting thing to note, but I'm going to leave it quiet for now, it just looked noisy.
//    }
}

size_t
Debugger::GetProcessSTDOUT (Process *process, Stream *stream)
{
    size_t total_bytes = 0;
    if (stream == NULL)
        stream = GetOutputFile().get();

    if (stream)
    {
        //  The process has stuff waiting for stdout; get it and write it out to the appropriate place.
        if (process == NULL)
        {
            TargetSP target_sp = GetTargetList().GetSelectedTarget();
            if (target_sp)
                process = target_sp->GetProcessSP().get();
        }
        if (process)
        {
            Error error;
            size_t len;
            char stdio_buffer[1024];
            while ((len = process->GetSTDOUT (stdio_buffer, sizeof (stdio_buffer), error)) > 0)
            {
                stream->Write(stdio_buffer, len);
                total_bytes += len;
            }
        }
        stream->Flush();
    }
    return total_bytes;
}

size_t
Debugger::GetProcessSTDERR (Process *process, Stream *stream)
{
    size_t total_bytes = 0;
    if (stream == NULL)
        stream = GetOutputFile().get();
    
    if (stream)
    {
        //  The process has stuff waiting for stderr; get it and write it out to the appropriate place.
        if (process == NULL)
        {
            TargetSP target_sp = GetTargetList().GetSelectedTarget();
            if (target_sp)
                process = target_sp->GetProcessSP().get();
        }
        if (process)
        {
            Error error;
            size_t len;
            char stdio_buffer[1024];
            while ((len = process->GetSTDERR (stdio_buffer, sizeof (stdio_buffer), error)) > 0)
            {
                stream->Write(stdio_buffer, len);
                total_bytes += len;
            }
        }
        stream->Flush();
    }
    return total_bytes;
}

// This function handles events that were broadcast by the process.
void
Debugger::HandleProcessEvent (const EventSP &event_sp)
{
    using namespace lldb;
    const uint32_t event_type = event_sp->GetType();
    ProcessSP process_sp = Process::ProcessEventData::GetProcessFromEvent(event_sp.get());
    
    StreamString output_stream;
    StreamString error_stream;
    const bool gui_enabled = IsForwardingEvents();

    if (!gui_enabled)
    {
        bool pop_process_io_handler = false;
        assert (process_sp);
    
        if (event_type & Process::eBroadcastBitSTDOUT || event_type & Process::eBroadcastBitStateChanged)
        {
            GetProcessSTDOUT (process_sp.get(), &output_stream);
        }
        
        if (event_type & Process::eBroadcastBitSTDERR || event_type & Process::eBroadcastBitStateChanged)
        {
            GetProcessSTDERR (process_sp.get(), &error_stream);
        }
    
        if (event_type & Process::eBroadcastBitStateChanged)
        {

            // Drain all stout and stderr so we don't see any output come after
            // we print our prompts
            // Something changed in the process;  get the event and report the process's current status and location to
            // the user.
            StateType event_state = Process::ProcessEventData::GetStateFromEvent (event_sp.get());
            if (event_state == eStateInvalid)
                return;
            
            switch (event_state)
            {
                case eStateInvalid:
                case eStateUnloaded:
                case eStateConnected:
                case eStateAttaching:
                case eStateLaunching:
                case eStateStepping:
                case eStateDetached:
                    {
                        output_stream.Printf("Process %" PRIu64 " %s\n",
                                             process_sp->GetID(),
                                             StateAsCString (event_state));
                        
                        if (event_state == eStateDetached)
                            pop_process_io_handler = true;
                    }
                    break;
                    
                case eStateRunning:
                    // Don't be chatty when we run...
                    break;
                    
                case eStateExited:
                    process_sp->GetStatus(output_stream);
                    pop_process_io_handler = true;
                    break;
                    
                case eStateStopped:
                case eStateCrashed:
                case eStateSuspended:
                    // Make sure the program hasn't been auto-restarted:
                    if (Process::ProcessEventData::GetRestartedFromEvent (event_sp.get()))
                    {
                        size_t num_reasons = Process::ProcessEventData::GetNumRestartedReasons(event_sp.get());
                        if (num_reasons > 0)
                        {
                            // FIXME: Do we want to report this, or would that just be annoyingly chatty?
                            if (num_reasons == 1)
                            {
                                const char *reason = Process::ProcessEventData::GetRestartedReasonAtIndex (event_sp.get(), 0);
                                output_stream.Printf("Process %" PRIu64 " stopped and restarted: %s\n",
                                                     process_sp->GetID(),
                                                     reason ? reason : "<UNKNOWN REASON>");
                            }
                            else
                            {
                                output_stream.Printf("Process %" PRIu64 " stopped and restarted, reasons:\n",
                                                     process_sp->GetID());
                                

                                for (size_t i = 0; i < num_reasons; i++)
                                {
                                    const char *reason = Process::ProcessEventData::GetRestartedReasonAtIndex (event_sp.get(), i);
                                    output_stream.Printf("\t%s\n", reason ? reason : "<UNKNOWN REASON>");
                                }
                            }
                        }
                    }
                    else
                    {
                        // Lock the thread list so it doesn't change on us
                        ThreadList &thread_list = process_sp->GetThreadList();
                        Mutex::Locker locker (thread_list.GetMutex());
                        
                        ThreadSP curr_thread (thread_list.GetSelectedThread());
                        ThreadSP thread;
                        StopReason curr_thread_stop_reason = eStopReasonInvalid;
                        if (curr_thread)
                            curr_thread_stop_reason = curr_thread->GetStopReason();
                        if (!curr_thread ||
                            !curr_thread->IsValid() ||
                            curr_thread_stop_reason == eStopReasonInvalid ||
                            curr_thread_stop_reason == eStopReasonNone)
                        {
                            // Prefer a thread that has just completed its plan over another thread as current thread.
                            ThreadSP plan_thread;
                            ThreadSP other_thread;
                            const size_t num_threads = thread_list.GetSize();
                            size_t i;
                            for (i = 0; i < num_threads; ++i)
                            {
                                thread = thread_list.GetThreadAtIndex(i);
                                StopReason thread_stop_reason = thread->GetStopReason();
                                switch (thread_stop_reason)
                                {
                                    case eStopReasonInvalid:
                                    case eStopReasonNone:
                                        break;
                                        
                                    case eStopReasonTrace:
                                    case eStopReasonBreakpoint:
                                    case eStopReasonWatchpoint:
                                    case eStopReasonSignal:
                                    case eStopReasonException:
                                    case eStopReasonExec:
                                    case eStopReasonThreadExiting:
                                        if (!other_thread)
                                            other_thread = thread;
                                        break;
                                    case eStopReasonPlanComplete:
                                        if (!plan_thread)
                                            plan_thread = thread;
                                        break;
                                }
                            }
                            if (plan_thread)
                                thread_list.SetSelectedThreadByID (plan_thread->GetID());
                            else if (other_thread)
                                thread_list.SetSelectedThreadByID (other_thread->GetID());
                            else
                            {
                                if (curr_thread && curr_thread->IsValid())
                                    thread = curr_thread;
                                else
                                    thread = thread_list.GetThreadAtIndex(0);
                                
                                if (thread)
                                    thread_list.SetSelectedThreadByID (thread->GetID());
                            }
                        }

                        if (GetTargetList().GetSelectedTarget().get() == &process_sp->GetTarget())
                        {
                            const bool only_threads_with_stop_reason = true;
                            const uint32_t start_frame = 0;
                            const uint32_t num_frames = 1;
                            const uint32_t num_frames_with_source = 1;
                            process_sp->GetStatus(output_stream);
                            process_sp->GetThreadStatus (output_stream,
                                                         only_threads_with_stop_reason,
                                                         start_frame,
                                                         num_frames,
                                                         num_frames_with_source);
                        }
                        else
                        {
                            uint32_t target_idx = GetTargetList().GetIndexOfTarget(process_sp->GetTarget().shared_from_this());
                            if (target_idx != UINT32_MAX)
                                output_stream.Printf ("Target %d: (", target_idx);
                            else
                                output_stream.Printf ("Target <unknown index>: (");
                            process_sp->GetTarget().Dump (&output_stream, eDescriptionLevelBrief);
                            output_stream.Printf (") stopped.\n");
                        }
                        
                        // Pop the process IO handler
                        pop_process_io_handler = true;
                    }
                    break;
            }
        }
    
        if (output_stream.GetSize() || error_stream.GetSize())
        {
            StreamFileSP error_stream_sp (GetOutputFile());
            bool top_io_handler_hid = HideTopIOHandler();

            if (output_stream.GetSize())
            {
                StreamFileSP output_stream_sp (GetOutputFile());
                if (output_stream_sp)
                    output_stream_sp->Write (output_stream.GetData(), output_stream.GetSize());
            }

            if (error_stream.GetSize())
            {
                StreamFileSP error_stream_sp (GetErrorFile());
                if (error_stream_sp)
                    error_stream_sp->Write (error_stream.GetData(), error_stream.GetSize());
            }

            if (top_io_handler_hid)
                RefreshTopIOHandler();
        }

        if (pop_process_io_handler)
            process_sp->PopProcessIOHandler();
    }
}

void
Debugger::HandleThreadEvent (const EventSP &event_sp)
{
    // At present the only thread event we handle is the Frame Changed event,
    // and all we do for that is just reprint the thread status for that thread.
    using namespace lldb;
    const uint32_t event_type = event_sp->GetType();
    if (event_type == Thread::eBroadcastBitStackChanged   ||
        event_type == Thread::eBroadcastBitThreadSelected )
    {
        ThreadSP thread_sp (Thread::ThreadEventData::GetThreadFromEvent (event_sp.get()));
        if (thread_sp)
        {
            HideTopIOHandler();
            StreamFileSP stream_sp (GetOutputFile());
            thread_sp->GetStatus(*stream_sp, 0, 1, 1);
            RefreshTopIOHandler();
        }
    }
}

bool
Debugger::IsForwardingEvents ()
{
    return (bool)m_forward_listener_sp;
}

void
Debugger::EnableForwardEvents (const ListenerSP &listener_sp)
{
    m_forward_listener_sp = listener_sp;
}

void
Debugger::CancelForwardEvents (const ListenerSP &listener_sp)
{
    m_forward_listener_sp.reset();
}


void
Debugger::DefaultEventHandler()
{
    Listener& listener(GetListener());
    ConstString broadcaster_class_target(Target::GetStaticBroadcasterClass());
    ConstString broadcaster_class_process(Process::GetStaticBroadcasterClass());
    ConstString broadcaster_class_thread(Thread::GetStaticBroadcasterClass());
    BroadcastEventSpec target_event_spec (broadcaster_class_target,
                                          Target::eBroadcastBitBreakpointChanged);

    BroadcastEventSpec process_event_spec (broadcaster_class_process,
                                           Process::eBroadcastBitStateChanged   |
                                           Process::eBroadcastBitSTDOUT         |
                                           Process::eBroadcastBitSTDERR);

    BroadcastEventSpec thread_event_spec (broadcaster_class_thread,
                                          Thread::eBroadcastBitStackChanged     |
                                          Thread::eBroadcastBitThreadSelected   );
    
    listener.StartListeningForEventSpec (*this, target_event_spec);
    listener.StartListeningForEventSpec (*this, process_event_spec);
    listener.StartListeningForEventSpec (*this, thread_event_spec);
    listener.StartListeningForEvents (m_command_interpreter_ap.get(),
                                      CommandInterpreter::eBroadcastBitQuitCommandReceived      |
                                      CommandInterpreter::eBroadcastBitAsynchronousOutputData   |
                                      CommandInterpreter::eBroadcastBitAsynchronousErrorData    );
    
    bool done = false;
    while (!done)
    {
//        Mutex::Locker locker;
//        if (locker.TryLock(m_input_reader_stack.GetMutex()))
//        {
//            if (m_input_reader_stack.IsEmpty())
//                break;
//        }
//
        EventSP event_sp;
        if (listener.WaitForEvent(NULL, event_sp))
        {
            if (event_sp)
            {
                Broadcaster *broadcaster = event_sp->GetBroadcaster();
                if (broadcaster)
                {
                    uint32_t event_type = event_sp->GetType();
                    ConstString broadcaster_class (broadcaster->GetBroadcasterClass());
                    if (broadcaster_class == broadcaster_class_process)
                    {
                        HandleProcessEvent (event_sp);
                    }
                    else if (broadcaster_class == broadcaster_class_target)
                    {
                        if (Breakpoint::BreakpointEventData::GetEventDataFromEvent(event_sp.get()))
                        {
                            HandleBreakpointEvent (event_sp);
                        }
                    }
                    else if (broadcaster_class == broadcaster_class_thread)
                    {
                        HandleThreadEvent (event_sp);
                    }
                    else if (broadcaster == m_command_interpreter_ap.get())
                    {
                        if (event_type & CommandInterpreter::eBroadcastBitQuitCommandReceived)
                        {
                            done = true;
                        }
                        else if (event_type & CommandInterpreter::eBroadcastBitAsynchronousErrorData)
                        {
                            const char *data = reinterpret_cast<const char *>(EventDataBytes::GetBytesFromEvent (event_sp.get()));
                            if (data && data[0])
                            {
                                StreamFileSP error_sp (GetErrorFile());
                                if (error_sp)
                                {
                                    HideTopIOHandler();
                                    error_sp->PutCString(data);
                                    error_sp->Flush();
                                    RefreshTopIOHandler();
                                }
                            }
                        }
                        else if (event_type & CommandInterpreter::eBroadcastBitAsynchronousOutputData)
                        {
                            const char *data = reinterpret_cast<const char *>(EventDataBytes::GetBytesFromEvent (event_sp.get()));
                            if (data && data[0])
                            {
                                StreamFileSP output_sp (GetOutputFile());
                                if (output_sp)
                                {
                                    HideTopIOHandler();
                                    output_sp->PutCString(data);
                                    output_sp->Flush();
                                    RefreshTopIOHandler();
                                }
                            }
                        }
                    }
                }
                
                if (m_forward_listener_sp)
                    m_forward_listener_sp->AddEvent(event_sp);
            }
        }
    }
}

lldb::thread_result_t
Debugger::EventHandlerThread (lldb::thread_arg_t arg)
{
    ((Debugger *)arg)->DefaultEventHandler();
    return NULL;
}

bool
Debugger::StartEventHandlerThread()
{
    if (!IS_VALID_LLDB_HOST_THREAD(m_event_handler_thread))
        m_event_handler_thread = Host::ThreadCreate("lldb.debugger.event-handler", EventHandlerThread, this, NULL);
    return IS_VALID_LLDB_HOST_THREAD(m_event_handler_thread);
}

void
Debugger::StopEventHandlerThread()
{
    if (IS_VALID_LLDB_HOST_THREAD(m_event_handler_thread))
    {
        GetCommandInterpreter().BroadcastEvent(CommandInterpreter::eBroadcastBitQuitCommandReceived);
        Host::ThreadJoin(m_event_handler_thread, NULL, NULL);
        m_event_handler_thread = LLDB_INVALID_HOST_THREAD;
    }
}


lldb::thread_result_t
Debugger::IOHandlerThread (lldb::thread_arg_t arg)
{
    Debugger *debugger = (Debugger *)arg;
    debugger->ExecuteIOHanders();
    debugger->StopEventHandlerThread();
    return NULL;
}

bool
Debugger::StartIOHandlerThread()
{
    if (!IS_VALID_LLDB_HOST_THREAD(m_io_handler_thread))
        m_io_handler_thread = Host::ThreadCreate("lldb.debugger.io-handler", IOHandlerThread, this, NULL);
    return IS_VALID_LLDB_HOST_THREAD(m_io_handler_thread);
}

void
Debugger::StopIOHandlerThread()
{
    if (IS_VALID_LLDB_HOST_THREAD(m_io_handler_thread))
    {
        if (m_input_file_sp)
            m_input_file_sp->GetFile().Close();
        Host::ThreadJoin(m_io_handler_thread, NULL, NULL);
        m_io_handler_thread = LLDB_INVALID_HOST_THREAD;
    }
}


