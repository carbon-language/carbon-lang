//===-- MICmdCmdBreak.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Overview:    CMICmdCmdBreakInsert            interface.
//              CMICmdCmdBreakDelete            interface.
//              CMICmdCmdBreakDisable           interface.
//              CMICmdCmdBreakEnable            interface.
//              CMICmdCmdBreakAfter             interface.
//              CMICmdCmdBreakCondition         interface.
//
//              To implement new MI commands derive a new command class from the command base
//              class. To enable the new command for interpretation add the new command class
//              to the command factory. The files of relevance are:
//                  MICmdCommands.cpp
//                  MICmdBase.h / .cpp
//                  MICmdCmd.h / .cpp
//              For an introduction to adding a new command see CMICmdCmdSupportInfoMiCmdQuery
//              command class as an example.

#pragma once

// Third party headers:
#include "lldb/API/SBBreakpoint.h"

// In-house headers:
#include "MICmdBase.h"

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "break-insert".
//          This command does not follow the MI documentation exactly.
// Gotchas: None.
// Authors: Illya Rudkin 11/03/2014.
// Changes: None.
//--
class CMICmdCmdBreakInsert : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdBreakInsert(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdBreakInsert(void) override;

    // Enumerations:
  private:
    //++ ===================================================================
    // Details: The type of break point give in the MI command text.
    //--
    enum BreakPoint_e
    {
        eBreakPoint_Invalid = 0,
        eBreakPoint_ByFileLine,
        eBreakPoint_ByFileFn,
        eBreakPoint_ByName,
        eBreakPoint_ByAddress,
        eBreakPoint_count,
        eBreakPoint_NotDefineYet
    };

    // Attributes:
  private:
    bool m_bBrkPtIsTemp;
    bool m_bHaveArgOptionThreadGrp;
    CMIUtilString m_brkName;
    CMIUtilString m_strArgOptionThreadGrp;
    lldb::SBBreakpoint m_brkPt;
    bool m_bBrkPtIsPending;
    MIuint m_nBrkPtIgnoreCount;
    bool m_bBrkPtEnabled;
    bool m_bBrkPtCondition;
    CMIUtilString m_brkPtCondition;
    bool m_bBrkPtThreadId;
    MIuint m_nBrkPtThreadId;
    const CMIUtilString m_constStrArgNamedTempBrkPt;
    const CMIUtilString m_constStrArgNamedHWBrkPt; // Not handled by *this command
    const CMIUtilString m_constStrArgNamedPendinfBrkPt;
    const CMIUtilString m_constStrArgNamedDisableBrkPt;
    const CMIUtilString m_constStrArgNamedTracePt; // Not handled by *this command
    const CMIUtilString m_constStrArgNamedConditionalBrkPt;
    const CMIUtilString m_constStrArgNamedInoreCnt;
    const CMIUtilString m_constStrArgNamedRestrictBrkPtToThreadId;
    const CMIUtilString m_constStrArgNamedLocation;
    const CMIUtilString m_constStrArgNamedThreadGroup; // Not specified in MI spec but Eclipse gives this option sometimes
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "break-delete".
// Gotchas: None.
// Authors: Illya Rudkin 11/03/2014.
// Changes: None.
//--
class CMICmdCmdBreakDelete : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdBreakDelete(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdBreakDelete(void) override;

    // Attributes:
  private:
    const CMIUtilString m_constStrArgNamedBrkPt;
    const CMIUtilString m_constStrArgNamedThreadGrp; // Not specified in MI spec but Eclipse gives this option
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "break-disable".
// Gotchas: None.
// Authors: Illya Rudkin 19/05/2014.
// Changes: None.
//--
class CMICmdCmdBreakDisable : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdBreakDisable(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdBreakDisable(void) override;

    // Attributes:
  private:
    const CMIUtilString m_constStrArgNamedThreadGrp; // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgNamedBrkPt;
    bool m_bBrkPtDisabledOk;
    MIuint m_nBrkPtId;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "break-enable".
// Gotchas: None.
// Authors: Illya Rudkin 19/05/2014.
// Changes: None.
//--
class CMICmdCmdBreakEnable : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdBreakEnable(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdBreakEnable(void) override;

    // Attributes:
  private:
    const CMIUtilString m_constStrArgNamedThreadGrp; // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgNamedBrkPt;
    bool m_bBrkPtEnabledOk;
    MIuint m_nBrkPtId;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "break-after".
// Gotchas: None.
// Authors: Illya Rudkin 29/05/2014.
// Changes: None.
//--
class CMICmdCmdBreakAfter : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdBreakAfter(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdBreakAfter(void) override;

    // Attributes:
  private:
    const CMIUtilString m_constStrArgNamedThreadGrp; // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgNamedNumber;
    const CMIUtilString m_constStrArgNamedCount;
    MIuint m_nBrkPtId;
    MIuint m_nBrkPtCount;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "break-condition".
// Gotchas: None.
// Authors: Illya Rudkin 29/05/2014.
// Changes: None.
//--
class CMICmdCmdBreakCondition : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdBreakCondition(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdBreakCondition(void) override;

    // Methods:
  private:
    CMIUtilString GetRestOfExpressionNotSurroundedInQuotes(void);

    // Attributes:
  private:
    const CMIUtilString m_constStrArgNamedThreadGrp; // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgNamedNumber;
    const CMIUtilString m_constStrArgNamedExpr;
    const CMIUtilString m_constStrArgNamedExprNoQuotes; // Not specified in MI spec, we need to handle expressions not surrounded by quotes
    MIuint m_nBrkPtId;
    CMIUtilString m_strBrkPtExpr;
};
