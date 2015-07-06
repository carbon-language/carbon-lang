//===-- MICmdCmdVar.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Overview:    CMICmdCmdVarCreate              interface.
//              CMICmdCmdVarUpdate              interface.
//              CMICmdCmdVarDelete              interface.
//              CMICmdCmdVarAssign              interface.
//              CMICmdCmdVarSetFormat           interface.
//              CMICmdCmdVarListChildren        interface.
//              CMICmdCmdVarEvaluateExpression  interface.
//              CMICmdCmdVarInfoPathExpression  interface.
//              CMICmdCmdVarShowAttributes      interface.
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

// In-house headers:
#include "MICmdBase.h"
#include "MICmnMIValueTuple.h"
#include "MICmnMIValueList.h"
#include "MICmnLLDBDebugSessionInfo.h"
#include "MICmnLLDBDebugSessionInfoVarObj.h"

// Declarations:
class CMICmnLLDBDebugSessionInfoVarObj;

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "var-create".
// Gotchas: None.
// Authors: Illya Rudkin 24/03/2014.
// Changes: None.
//--
class CMICmdCmdVarCreate : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdVarCreate(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;

    // Overridden:
  public:
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdVarCreate(void) override;

    // Methods:
  private:
    void CompleteSBValue(lldb::SBValue &vrwValue);

    // Attribute:
  private:
    CMIUtilString m_strVarName;
    MIuint m_nChildren;
    MIuint64 m_nThreadId;
    CMIUtilString m_strType;
    bool m_bValid; // True = Variable is valid, false = not valid
    CMIUtilString m_strExpression;
    CMIUtilString m_strValue;
    const CMIUtilString m_constStrArgThread;      // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgThreadGroup; // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgFrame;       // Not specified in MI spec but Eclipse gives this option
    const CMIUtilString m_constStrArgName;
    const CMIUtilString m_constStrArgFrameAddr;
    const CMIUtilString m_constStrArgExpression;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "var-update".
// Gotchas: None.
// Authors: Illya Rudkin 25/03/2014.
// Changes: None.
//--
class CMICmdCmdVarUpdate : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdVarUpdate(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;

    // Overridden:
  public:
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdVarUpdate(void) override;

    // Methods:
  private:
    bool ExamineSBValueForChange(lldb::SBValue &vrwValue, bool &vrwbChanged);
    bool MIFormResponse(const CMIUtilString &vrStrVarName, const char *const vpValue, const CMIUtilString &vrStrScope);

    // Attribute:
  private:
    const CMIUtilString m_constStrArgPrintValues;
    const CMIUtilString m_constStrArgName;
    bool m_bValueChanged; // True = yes value changed, false = no change
    CMICmnMIValueList m_miValueList;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "var-delete".
// Gotchas: None.
// Authors: Illya Rudkin 25/03/2014.
// Changes: None.
//--
class CMICmdCmdVarDelete : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdVarDelete(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdVarDelete(void) override;

    // Attribute:
  private:
    const CMIUtilString m_constStrArgName;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "var-assign".
// Gotchas: None.
// Authors: Illya Rudkin 25/03/2014.
// Changes: None.
//--
class CMICmdCmdVarAssign : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdVarAssign(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdVarAssign(void) override;

    // Attributes:
  private:
    bool m_bOk; // True = success, false = failure
    CMIUtilString m_varObjName;
    const CMIUtilString m_constStrArgName;
    const CMIUtilString m_constStrArgExpression;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "var-set-format".
// Gotchas: None.
// Authors: Illya Rudkin 26/03/2014.
// Changes: None.
//--
class CMICmdCmdVarSetFormat : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdVarSetFormat(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdVarSetFormat(void) override;

    // Attributes:
  private:
    CMIUtilString m_varObjName;
    const CMIUtilString m_constStrArgName;
    const CMIUtilString m_constStrArgFormatSpec;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "var-list-children".
// Gotchas: None.
// Authors: Illya Rudkin 26/03/2014.
// Changes: None.
//--
class CMICmdCmdVarListChildren : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdVarListChildren(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdVarListChildren(void) override;

    // Attributes:
  private:
    const CMIUtilString m_constStrArgPrintValues;
    const CMIUtilString m_constStrArgName;
    const CMIUtilString m_constStrArgFrom;
    const CMIUtilString m_constStrArgTo;
    bool m_bValueValid; // True = yes SBValue object is valid, false = not valid
    MIuint m_nChildren;
    CMICmnMIValueList m_miValueList;
    bool m_bHasMore;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "var-evaluate-expression".
// Gotchas: None.
// Authors: Illya Rudkin 26/03/2014.
// Changes: None.
//--
class CMICmdCmdVarEvaluateExpression : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdVarEvaluateExpression(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdVarEvaluateExpression(void) override;

    // Attributes:
  private:
    bool m_bValueValid; // True = yes SBValue object is valid, false = not valid
    CMIUtilString m_varObjName;
    const CMIUtilString m_constStrArgFormatSpec; // Not handled by *this command
    const CMIUtilString m_constStrArgName;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "var-info-path-expression".
// Gotchas: None.
// Authors: Illya Rudkin 26/03/2014.
// Changes: None.
//--
class CMICmdCmdVarInfoPathExpression : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdVarInfoPathExpression(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdVarInfoPathExpression(void) override;

    // Attributes:
  private:
    bool m_bValueValid; // True = yes SBValue object is valid, false = not valid
    CMIUtilString m_strPathExpression;
    const CMIUtilString m_constStrArgName;
};

//++ ============================================================================
// Details: MI command class. MI commands derived from the command base class.
//          *this class implements MI command "var-show-attributes".
// Gotchas: None.
// Authors: Illya Rudkin 19/05/2014.
// Changes: None.
//--
class CMICmdCmdVarShowAttributes : public CMICmdBase
{
    // Statics:
  public:
    // Required by the CMICmdFactory when registering *this command
    static CMICmdBase *CreateSelf(void);

    // Methods:
  public:
    /* ctor */ CMICmdCmdVarShowAttributes(void);

    // Overridden:
  public:
    // From CMICmdInvoker::ICmd
    bool Execute(void) override;
    bool Acknowledge(void) override;
    bool ParseArgs(void) override;
    // From CMICmnBase
    /* dtor */ ~CMICmdCmdVarShowAttributes(void) override;

    // Attributes:
  private:
    const CMIUtilString m_constStrArgName;
};
