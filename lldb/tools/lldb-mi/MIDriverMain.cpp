//===-- MIDriverMain.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MIDriverMain.cpp
//
// Overview:    Defines the entry point for the console application.
//              The MI application (project name MI) runs in two modes:
//              An LLDB native driver mode where it acts no different from the LLDB driver.
//              The other mode is the MI when it finds on the command line
//              the --interpreter option. Command line argument --help on its own will give
//              help for the LLDB driver. If entered with --interpreter then MI help will
//              provided.
//              To implement new MI commands derive a new command class from the command base
//              class. To enable the new command for interpretation add the new command class
//              to the command factory. The files of relevance are:
//                  MICmdCommands.cpp
//                  MICmdBase.h / .cpp
//                  MICmdCmd.h / .cpp
// Versions:    1.0.0.1     First version from scratch 28/1/2014 to 28/3/2014. MI not complete.
//              1.0.0.2     First deliverable to client 7/3/2014. MI not complete.
//              1.0.0.3     Code refactor tidy. Release to community for evaluation 17/5/2014. MI not complete.
//              1.0.0.4     Post release to the community for evaluation 17/5/2014. MI not complete.
//              1.0.0.5     Second deliverable to client 16/6/2014.
//              1.0.0.6     Post release of second deliverable to client 16/6/2014.
//                          Released to the community 24/6/2014.
//              1.0.0.7     Post release to the community.
//                          Delivered to client 30/6/2014.
//              1.0.0.8     Delivered to client 29/7/2014.
//              1.0.0.9     Post release to client 29/7/2014.
//              See MIreadme.txt for list of MI commands implemented.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadme.txt.
//
// Copyright:   None.
//--

#if defined(_MSC_VER)
#define _INC_SIGNAL // Stop window's signal.h being included - CODETAG_IOR_SIGNALS
#endif              // _MSC_VER

// Third party headers:
#include <stdio.h>
#include <lldb/API/SBHostOS.h>

// In house headers:
#include "MICmnConfig.h"
#include "Platform.h" // Define signals - CODETAG_IOR_SIGNALS
#include "Driver.h"
#include "MIDriverMgr.h"
#include "MIDriver.h"
#include "MICmnResources.h"
#include "MICmnStreamStdin.h"
#include "MIUtilDebug.h"
#include "MICmnLog.h"

#if MICONFIG_COMPILE_MIDRIVER_VERSION

#if defined(_MSC_VER)
#pragma warning(once : 4530) // Warning C4530: C++ exception handler used, but unwind semantics are not enabled. Specify /EHsc
#endif                       // _MSC_VER

// ToDo: Reevaluate if this function needs to be implemented like the UNIX equivalent
// CODETAG_IOR_SIGNALS
//++ ------------------------------------------------------------------------------------
// Details: The SIGWINCH signal is sent to a process when its controlling terminal
//          changes its size (a window change).
// Type:    Function.
// Args:    vSigno  - (R) Signal number.
// Return:  None.
// Throws:  None.
//--
void
sigwinch_handler(int vSigno)
{
    MIunused(vSigno);

    struct winsize window_size;
    if (::isatty(STDIN_FILENO) && ::ioctl(STDIN_FILENO, TIOCGWINSZ, &window_size) == 0)
    {
        CMIDriverMgr &rDriverMgr = CMIDriverMgr::Instance();
        if (window_size.ws_col > 0)
        {
            rDriverMgr.DriverResizeWindow((uint32_t)window_size.ws_col);
        }
    }

    CMICmnLog::Instance().WriteLog(CMIUtilString::Format(MIRSRC(IDS_PROCESS_SIGNAL_RECEIVED), "SIGWINCH", vSigno));
}

// CODETAG_IOR_SIGNALS
//++ ------------------------------------------------------------------------------------
// Details: The SIGINT signal is sent to a process by its controlling terminal when a
//          user wishes to interrupt the process. This is typically initiated by pressing
//          Control-C, but on some systems, the "delete" character or "break" key can be
//          used.
//          Be aware this function may be called on another thread besides the main thread.
// Type:    Function.
// Args:    vSigno  - (R) Signal number.
// Return:  None.
// Throws:  None.
//--
void
sigint_handler(int vSigno)
{
    static bool g_interrupt_sent = false;
    CMIDriverMgr &rDriverMgr = CMIDriverMgr::Instance();
    lldb::SBDebugger *pDebugger = rDriverMgr.DriverGetTheDebugger();
    if (pDebugger != nullptr)
    {
        if (!g_interrupt_sent)
        {
            g_interrupt_sent = true;
            pDebugger->DispatchInputInterrupt();
            g_interrupt_sent = false;
        }
    }

    CMICmnLog::Instance().WriteLog(CMIUtilString::Format(MIRSRC(IDS_PROCESS_SIGNAL_RECEIVED), "SIGINT", vSigno));

    // CODETAG_DEBUG_SESSION_RUNNING_PROG_RECEIVED_SIGINT_PAUSE_PROGRAM
    // Signal MI to shutdown or halt a running debug session
    CMICmnStreamStdin::Instance().SetCtrlCHit();
}

// ToDo: Reevaluate if this function needs to be implemented like the UNIX equivalent
// CODETAG_IOR_SIGNALS
//++ ------------------------------------------------------------------------------------
// Details: The SIGTSTP signal is sent to a process by its controlling terminal to
//          request it to stop temporarily. It is commonly initiated by the user pressing
//          Control-Z. Unlike SIGSTOP, the process can register a signal handler for or
//          ignore the signal.
//          *** The function does not behave ATM like the UNIX equivalent ***
// Type:    Function.
// Args:    vSigno  - (R) Signal number.
// Return:  None.
// Throws:  None.
//--
void
sigtstp_handler(int vSigno)
{
    CMIDriverMgr &rDriverMgr = CMIDriverMgr::Instance();
    lldb::SBDebugger *pDebugger = rDriverMgr.DriverGetTheDebugger();
    if (pDebugger != nullptr)
    {
        pDebugger->SaveInputTerminalState();
    }

    CMICmnLog::Instance().WriteLog(CMIUtilString::Format(MIRSRC(IDS_PROCESS_SIGNAL_RECEIVED), "SIGTSTP", vSigno));

    // Signal MI to shutdown
    CMICmnStreamStdin::Instance().SetCtrlCHit();
}

// ToDo: Reevaluate if this function needs to be implemented like the UNIX equivalent
// CODETAG_IOR_SIGNALS
//++ ------------------------------------------------------------------------------------
// Details: The SIGCONT signal instructs the operating system to continue (restart) a
//          process previously paused by the SIGSTOP or SIGTSTP signal. One important use
//          of this signal is in job control in the UNIX shell.
//          *** The function does not behave ATM like the UNIX equivalent ***
// Type:    Function.
// Args:    vSigno  - (R) Signal number.
// Return:  None.
// Throws:  None.
//--
void
sigcont_handler(int vSigno)
{
    CMIDriverMgr &rDriverMgr = CMIDriverMgr::Instance();
    lldb::SBDebugger *pDebugger = rDriverMgr.DriverGetTheDebugger();
    if (pDebugger != nullptr)
    {
        pDebugger->RestoreInputTerminalState();
    }

    CMICmnLog::Instance().WriteLog(CMIUtilString::Format(MIRSRC(IDS_PROCESS_SIGNAL_RECEIVED), "SIGCONT", vSigno));

    // Signal MI to shutdown
    CMICmnStreamStdin::Instance().SetCtrlCHit();
}

//++ ------------------------------------------------------------------------------------
// Details: Init the MI driver system. Initialize the whole driver system which includes
//          both the original LLDB driver and the MI driver.
// Type:    Function.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
DriverSystemInit(void)
{
    bool bOk = MIstatus::success;

#if MICONFIG_COMPILE_MIDRIVER_WITH_LLDBDRIVER
    Driver *pDriver = Driver::CreateSelf();
    if (pDriver == nullptr)
        return MIstatus::failure;
#endif // MICONFIG_COMPILE_MIDRIVER_WITH_LLDBDRIVER

    CMIDriver &rMIDriver = CMIDriver::Instance();
    CMIDriverMgr &rDriverMgr = CMIDriverMgr::Instance();
    bOk = rDriverMgr.Initialize();

    // Register MIDriver first as it needs to initialize and be ready
    // for the Driver to get information from MIDriver when it initializes
    // (LLDB Driver is registered with the Driver Manager in MI's Initialize())
    bOk = bOk && rDriverMgr.RegisterDriver(rMIDriver, "MIDriver"); // Will be main driver

    return bOk;
}

//++ ------------------------------------------------------------------------------------
// Details: Shutdown the debugger system. Release / terminate resources external to
//          specifically the MI driver.
// Type:    Function.
// Args:    vbAppExitOk - (R) True = No problems, false = App exiting with problems (investigate!).
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
DriverSystemShutdown(const bool vbAppExitOk)
{
    bool bOk = MIstatus::success;

    // *** Order is important here ***
    CMIDriverMgr::Instance().Shutdown();

#if MICONFIG_COMPILE_MIDRIVER_WITH_LLDBDRIVER
    delete g_driver;
    g_driver = nullptr;
#endif // MICONFIG_COMPILE_MIDRIVER_WITH_LLDBDRIVER

    return bOk;
}

#else
void
sigwinch_handler(int signo)
{
    struct winsize window_size;
    if (isatty(STDIN_FILENO) && ::ioctl(STDIN_FILENO, TIOCGWINSZ, &window_size) == 0)
    {
        if ((window_size.ws_col > 0) && g_driver != NULL)
        {
            g_driver->ResizeWindow(window_size.ws_col);
        }
    }
}

void
sigint_handler(int signo)
{
    static bool g_interrupt_sent = false;
    if (g_driver)
    {
        if (!g_interrupt_sent)
        {
            g_interrupt_sent = true;
            g_driver->GetDebugger().DispatchInputInterrupt();
            g_interrupt_sent = false;
            return;
        }
    }

    exit(signo);
}

void
sigtstp_handler(int signo)
{
    g_driver->GetDebugger().SaveInputTerminalState();
    signal(signo, SIG_DFL);
    kill(getpid(), signo);
    signal(signo, sigtstp_handler);
}

void
sigcont_handler(int signo)
{
    g_driver->GetDebugger().RestoreInputTerminalState();
    signal(signo, SIG_DFL);
    kill(getpid(), signo);
    signal(signo, sigcont_handler);
}
#endif // #if MICONFIG_COMPILE_MIDRIVER_VERSION

//++ ------------------------------------------------------------------------------------
// Details: MI's application start point of execution. The applicaton runs in two modes.
//          An LLDB native driver mode where it acts no different from the LLDB driver.
//          The other mode is the MI when it finds on the command line
//          the --interpreter option. Command line argument --help on its own will give
//          help for the LLDB driver. If entered with --interpreter then application
//          help will provided.
// Type:    Method.
// Args:    argc    - (R) An integer that contains the count of arguments that follow in
//                        argv. The argc parameter is always greater than or equal to 1.
//          argv    - (R) An array of null-terminated strings representing command-line
//                        arguments entered by the user of the program. By convention,
//                        argv[0] is the command with which the program is invoked.
// Return:  int -  0 =   Normal exit, program success.
//                >0    = Program success with status i.e. Control-C signal status
//                <0    = Program failed.
//              -1      = Program failed reason not specified here, see MI log file.
//              -1000   = Program failed did not initailize successfully.
// Throws:  None.
//--
#if MICONFIG_COMPILE_MIDRIVER_VERSION
int
main(int argc, char const *argv[])
{
#if MICONFIG_DEBUG_SHOW_ATTACH_DBG_DLG
#ifdef _WIN32
    CMIUtilDebug::ShowDlgWaitForDbgAttach();
#else
    CMIUtilDebug::WaitForDbgAttachInfinteLoop();
#endif //  _WIN32
#endif // MICONFIG_DEBUG_SHOW_ATTACH_DBG_DLG

    // *** Order is important here ***
    bool bOk = DriverSystemInit();
    if (!bOk)
    {
        DriverSystemShutdown(bOk);
        return -1000;
    }

    // CODETAG_IOR_SIGNALS
    signal(SIGPIPE, SIG_IGN);
    signal(SIGWINCH, sigwinch_handler);
    signal(SIGINT, sigint_handler);
    signal(SIGTSTP, sigtstp_handler);
    signal(SIGCONT, sigcont_handler);

    bool bExiting = false;
    CMIDriverMgr &rDriverMgr = CMIDriverMgr::Instance();
    bOk = bOk && rDriverMgr.ParseArgs(argc, argv, bExiting);
    if (bOk && !bExiting)
        bOk = rDriverMgr.DriverParseArgs(argc, argv, stdout, bExiting);
    if (bOk && !bExiting)
        bOk = rDriverMgr.DriverMainLoop();

    // Logger and other resources shutdown now
    DriverSystemShutdown(bOk);

    const int appResult = bOk ? 0 : -1;

    return appResult;
}
#else  // Operate the lldb Driver only version of the code
int
main(int argc, char const *argv[], char *envp[])
{
    MIunused(envp);
    using namespace lldb;
    SBDebugger::Initialize();

    SBHostOS::ThreadCreated("<lldb.driver.main-thread>");

    signal(SIGPIPE, SIG_IGN);
    signal(SIGWINCH, sigwinch_handler);
    signal(SIGINT, sigint_handler);
    signal(SIGTSTP, sigtstp_handler);
    signal(SIGCONT, sigcont_handler);

    // Create a scope for driver so that the driver object will destroy itself
    // before SBDebugger::Terminate() is called.
    {
        Driver driver;

        bool exiting = false;
        SBError error(driver.ParseArgs(argc, argv, stdout, exiting));
        if (error.Fail())
        {
            const char *error_cstr = error.GetCString();
            if (error_cstr)
                ::fprintf(stderr, "error: %s\n", error_cstr);
        }
        else if (!exiting)
        {
            driver.MainLoop();
        }
    }

    SBDebugger::Terminate();
    return 0;
}
#endif // MICONFIG_COMPILE_MIDRIVER_VERSION
