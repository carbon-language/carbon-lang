//===-- Driver.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// In-house headers:
#include "MICmnConfig.h"
#if MICONFIG_COMPILE_MIDRIVER_WITH_LLDBDRIVER

#ifndef lldb_Driver_h_
#define lldb_Driver_h_

//#include "Platform.h" // IOR removed
#include "lldb/Utility/PseudoTerminal.h"

#include <set>
#include <bitset>
#include <string>
#include <vector>

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "MIDriverMgr.h"
#include "MIDriverBase.h"

#define ASYNC true
#define NO_ASYNC false

class IOChannel;

class Driver : public lldb::SBBroadcaster, public CMIDriverBase, public CMIDriverMgr::IDriver
{
    // MI required code:
    // Static:
  public:
    static Driver *CreateSelf(void);

    // Methods:
  public:
    bool MISetup(CMIUtilString &vwErrMsg);

    // Overridden:
  public:
    // From CMIDriverMgr::IDriver
    virtual bool DoInitialize(void);
    virtual bool DoShutdown(void);
    virtual bool DoMainLoop(void);
    virtual void DoResizeWindow(const uint32_t vWindowSizeWsCol);
    virtual lldb::SBError DoParseArgs(const int argc, const char *argv[], FILE *vpStdOut, bool &vwbExiting);
    virtual CMIUtilString GetError(void) const;
    virtual const CMIUtilString &GetName(void) const;
    virtual lldb::SBDebugger &GetTheDebugger(void);
    virtual bool GetDriverIsGDBMICompatibleDriver(void) const;
    virtual bool SetId(const CMIUtilString &vID);
    virtual const CMIUtilString &GetId(void) const;
    // From CMIDriverBase
    virtual bool DoFallThruToAnotherDriver(const CMIUtilString &vCmd, CMIUtilString &vwErrMsg);
    virtual bool SetDriverParent(const CMIDriverBase &vrOtherDriver);
    virtual const CMIUtilString &GetDriverName(void) const;
    virtual const CMIUtilString &GetDriverId(void) const;

    // Original code:
  public:
    Driver();

    virtual ~Driver();

    void MainLoop();

    lldb::SBError ParseArgs(int argc, const char *argv[], FILE *out_fh, bool &do_exit);

    const char *GetFilename() const;

    const char *GetCrashLogFilename() const;

    const char *GetArchName() const;

    lldb::ScriptLanguage GetScriptLanguage() const;

    void ExecuteInitialCommands(bool before_file);

    bool GetDebugMode() const;

    class OptionData
    {
      public:
        OptionData();
        ~OptionData();

        void Clear();

        void AddInitialCommand(const char *command, bool before_file, bool is_file, lldb::SBError &error);

        // static OptionDefinition m_cmd_option_table[];

        std::vector<std::string> m_args;
        lldb::ScriptLanguage m_script_lang;
        std::string m_core_file;
        std::string m_crash_log;
        std::vector<std::pair<bool, std::string>> m_initial_commands;
        std::vector<std::pair<bool, std::string>> m_after_file_commands;
        bool m_debug_mode;
        bool m_source_quietly;
        bool m_print_version;
        bool m_print_python_path;
        bool m_print_help;
        bool m_wait_for;
        std::string m_process_name;
        lldb::pid_t m_process_pid;
        bool m_use_external_editor; // FIXME: When we have set/show variables we can remove this from here.
        typedef std::set<char> OptionSet;
        OptionSet m_seen_options;
    };

    static lldb::SBError SetOptionValue(int option_idx, const char *option_arg, Driver::OptionData &data);

    lldb::SBDebugger &
    GetDebugger()
    {
        return m_debugger;
    }

    void ResizeWindow(unsigned short col);

  private:
    lldb::SBDebugger m_debugger;
    OptionData m_option_data;

    void ResetOptionValues();

    void ReadyForCommand();
};

extern Driver *g_driver;

#endif // lldb_Driver_h_

#endif // MICONFIG_COMPILE_MIDRIVER_WITH_LLDBDRIVER
