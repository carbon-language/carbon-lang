//===-- MICmdCmdSymbol.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Overview:    CMICmdCmdSymbolListLines     implementation.

// Third Party Headers:
#include "lldb/API/SBCommandInterpreter.h"

// In-house headers:
#include "MICmdArgValFile.h"
#include "MICmdCmdSymbol.h"
#include "MICmnLLDBDebugSessionInfo.h"
#include "MICmnMIResultRecord.h"
#include "MICmnMIValueList.h"
#include "MICmnMIValueTuple.h"

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdSymbolListLines constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdSymbolListLines::CMICmdCmdSymbolListLines(void)
    : m_constStrArgNameFile("file")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "symbol-list-lines";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdSymbolListLines::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdSymbolListLines destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdSymbolListLines::~CMICmdCmdSymbolListLines(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function. The parses the command line options
//          arguments to extract values for each of those arguments.
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdSymbolListLines::ParseArgs(void)
{
    bool bOk = m_setCmdArgs.Add(*(new CMICmdArgValFile(m_constStrArgNameFile, true, true)));
    return (bOk && ParseValidateCmdOptions());
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function. The command does work in this function.
//          The command is likely to communicate with the LLDB SBDebugger in here.
//          Synopsis: -symbol-list-lines file
//          Ref: http://sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI-Symbol-Query.html#GDB_002fMI-Symbol-Query
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdSymbolListLines::Execute(void)
{
    CMICMDBASE_GETOPTION(pArgFile, File, m_constStrArgNameFile);

    const CMIUtilString &strFilePath(pArgFile->GetValue());
    const CMIUtilString strCmd(CMIUtilString::Format("target modules dump line-table \"%s\"", strFilePath.AddSlashes().c_str()));

    CMICmnLLDBDebugSessionInfo &rSessionInfo(CMICmnLLDBDebugSessionInfo::Instance());
    const lldb::ReturnStatus rtn = rSessionInfo.GetDebugger().GetCommandInterpreter().HandleCommand(strCmd.c_str(), m_lldbResult);
    MIunused(rtn);

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function. The command prepares a MI Record Result
//          for the work carried out in the Execute().
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdSymbolListLines::Acknowledge(void)
{
    if (m_lldbResult.GetErrorSize() > 0)
    {
        const char *pLldbErr = m_lldbResult.GetError();
        const CMIUtilString strMsg(CMIUtilString(pLldbErr).StripCRAll());
        const CMICmnMIValueConst miValueConst(strMsg);
        const CMICmnMIValueResult miValueResult("message", miValueConst);
        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult);
        m_miResultRecord = miRecordResult;
    }
    else
    {
        CMIUtilString::VecString_t vecLines;
        const CMIUtilString strLldbMsg(m_lldbResult.GetOutput());
        const MIuint nLines(strLldbMsg.SplitLines(vecLines));

        CMICmnMIValueList miValueList(true);
        for (MIuint i = 1; i < nLines; ++i)
        {
            // String looks like:
            // 0x0000000100000e70: /path/to/file:3[:4]
            const CMIUtilString &rLine(vecLines[i]);

            // 0x0000000100000e70: /path/to/file:3[:4]
            // ^^^^^^^^^^^^^^^^^^ -- pc
            const size_t nAddrEndPos = rLine.find(':');
            const CMIUtilString strAddr(rLine.substr(0, nAddrEndPos).c_str());
            const CMICmnMIValueConst miValueConst(strAddr);
            const CMICmnMIValueResult miValueResult("pc", miValueConst);
            CMICmnMIValueTuple miValueTuple(miValueResult);

            // 0x0000000100000e70: /path/to/file:3[:4]
            //                                   ^ -- line
            const size_t nLineOrColumnStartPos = rLine.rfind(':');
            const CMIUtilString strLineOrColumn(rLine.substr(nLineOrColumnStartPos + 1).c_str());
            const size_t nPathOrLineStartPos = rLine.rfind(':', nLineOrColumnStartPos - 1);
            const size_t nPathOrLineLen = nLineOrColumnStartPos - nPathOrLineStartPos - 1;
            const CMIUtilString strPathOrLine(rLine.substr(nPathOrLineStartPos + 1, nPathOrLineLen).c_str());
            const CMIUtilString strLine(strPathOrLine.IsNumber() ? strPathOrLine : strLineOrColumn);
            const CMICmnMIValueConst miValueConst2(strLine);
            const CMICmnMIValueResult miValueResult2("line", miValueConst2);
            bool bOk = miValueTuple.Add(miValueResult2);

            bOk = bOk && miValueList.Add(miValueTuple);
            if (!bOk)
                return MIstatus::failure;
        }

        // MI print "%s^done,lines=[{pc=\"%d\",line=\"%d\"}...]"
        const CMICmnMIValueResult miValueResult("lines", miValueList);
        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult);
        m_miResultRecord = miRecordResult;
    }

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Required by the CMICmdFactory when registering *this command. The factory
//          calls this function to create an instance of *this command.
// Type:    Static method.
// Args:    None.
// Return:  CMICmdBase * - Pointer to a new command.
// Throws:  None.
//--
CMICmdBase *
CMICmdCmdSymbolListLines::CreateSelf(void)
{
    return new CMICmdCmdSymbolListLines();
}
