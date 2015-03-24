//===-- MIDriverMgr.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

// Third party headers:
#include <map>
#include "lldb/API/SBDebugger.h"

// In-house headers:
#include "MICmnBase.h"
#include "MIUtilString.h"
#include "MICmnLog.h"
#include "MIUtilSingletonBase.h"

//++ ============================================================================
// Details: MI Driver Manager. Register lldb::SBBroadcaster derived Driver type
//          objects with *this manager. The manager does not own driver objects
//          registered with it and so will not delete when this manager is
//          shutdown. The Driver flagged as "use this one" will be set as current
//          driver and will be the one that is used. Other drivers are not
//          operated. A Driver can call another Driver should it not handle a
//          command.
//          It also initializes other resources as part it's setup such as the
//          Logger and Resources objects (explicit indicate *this object requires
//          those objects (modules/components) to support it's own functionality).
//          The Driver manager is the first object instantiated as part of the
//          MI code base. It is also the first thing to interpret the command
//          line arguments passed to the executeable. Bases on options it
//          understands the manage will set up the appropriate driver or give
//          help information. Other options are passed on to the driver chosen
//          to do work.
//          Each driver instance (the CMIDriver, LLDB::Driver) has its own
//          LLDB::SBDebugger.
//          Singleton class.
// Gotchas: None.
// Authors: Illya Rudkin 28/02/2014.
// Changes: None.
//--
class CMIDriverMgr : public CMICmnBase, public MI::ISingleton<CMIDriverMgr>
{
    friend MI::ISingleton<CMIDriverMgr>;

    // Class:
  public:
    //++
    // Description: Driver deriver objects need this interface to work with
    //              *this manager.
    //--
    class IDriver
    {
      public:
        virtual bool DoInitialize(void) = 0;
        virtual bool DoShutdown(void) = 0;
        virtual bool DoMainLoop(void) = 0;
        virtual lldb::SBError DoParseArgs(const int argc, const char *argv[], FILE *vpStdOut, bool &vwbExiting) = 0;
        virtual CMIUtilString GetError(void) const = 0;
        virtual const CMIUtilString &GetName(void) const = 0;
        virtual lldb::SBDebugger &GetTheDebugger(void) = 0;
        virtual bool GetDriverIsGDBMICompatibleDriver(void) const = 0;
        virtual bool SetId(const CMIUtilString &vId) = 0;
        virtual const CMIUtilString &GetId(void) const = 0;
        virtual void DeliverSignal(int signal) = 0;

        // Not part of the interface, ignore
        /* dtor */ virtual ~IDriver(void) {}
    };

    // Methods:
  public:
    // MI system
    bool Initialize(void);
    bool Shutdown(void);
    //
    CMIUtilString GetAppVersion(void) const;
    bool RegisterDriver(const IDriver &vrADriver, const CMIUtilString &vrDriverID);
    bool UnregisterDriver(const IDriver &vrADriver);
    bool
    SetUseThisDriverToDoWork(const IDriver &vrADriver); // Specify working main driver
    IDriver *GetUseThisDriverToDoWork(void) const;
    bool ParseArgs(const int argc, const char *argv[], bool &vwbExiting);
    IDriver *GetDriver(const CMIUtilString &vrDriverId) const;
    //
    // MI Proxy fn to current specified working driver
    bool DriverMainLoop(void);
    bool DriverParseArgs(const int argc, const char *argv[], FILE *vpStdOut, bool &vwbExiting);
    CMIUtilString DriverGetError(void) const;
    CMIUtilString DriverGetName(void) const;
    lldb::SBDebugger *DriverGetTheDebugger(void);
    void DeliverSignal(int signal);

    // Typedef:
  private:
    typedef std::map<CMIUtilString, IDriver *> MapDriverIdToDriver_t;
    typedef std::pair<CMIUtilString, IDriver *> MapPairDriverIdToDriver_t;

    // Methods:
  private:
    /* ctor */ CMIDriverMgr(void);
    /* ctor */ CMIDriverMgr(const CMIDriverMgr &);
    void operator=(const CMIDriverMgr &);
    //
    bool HaveDriverAlready(const IDriver &vrMedium) const;
    bool UnregisterDriverAll(void);
    IDriver *GetFirstMIDriver(void) const;
    IDriver *GetFirstNonMIDriver(void) const;
    CMIUtilString GetHelpOnCmdLineArgOptions(void) const;

    // Overridden:
  private:
    // From CMICmnBase
    /* dtor */ virtual ~CMIDriverMgr(void);

    // Attributes:
  private:
    MapDriverIdToDriver_t m_mapDriverIdToDriver;
    IDriver *m_pDriverCurrent; // This driver is used by this manager to do work. It is the main driver.
    bool m_bInMi2Mode;         // True = --interpreter entered on the cmd line, false = operate LLDB driver (non GDB)
};
