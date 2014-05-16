//===-- MICmdCmdBreak.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmdCmdBreak.h
//
// Overview:	CMICmdCmdBreakInsert			interface.
//				CMICmdCmdBreakDelete			interface.
//
//				To implement new MI commands derive a new command class from the command base 
//				class. To enable the new command for interpretation add the new command class
//				to the command factory. The files of relevance are:
//					MICmdCommands.cpp
//					MICmdBase.h / .cpp
//					MICmdCmd.h / .cpp
//				For an introduction to adding a new command see CMICmdCmdSupportInfoMiCmdQuery
//				command class as an example.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

#pragma once

// Third party headers:
#include <lldb/API/SBBreakpoint.h>

// In-house headers:
#include "MICmdBase.h"

//++ ============================================================================
// Details:	MI command class. MI commands derived from the command base class.
//			*this class implements MI command "break-insert".
//			This command does not follow the MI documentation exactly.
// Gotchas:	None.
// Authors:	Illya Rudkin 11/03/2014.
// Changes:	None.
//--
class CMICmdCmdBreakInsert : public CMICmdBase
{
// Statics:
public:
	// Required by the CMICmdFactory when registering *this commmand
	static CMICmdBase *	CreateSelf( void );

// Methods:
public:
	/* ctor */	CMICmdCmdBreakInsert( void );

// Overridden:
public:
	// From CMICmdInvoker::ICmd
	virtual bool	Execute( void );
	virtual bool	Acknowledge( void );
	virtual bool	ParseArgs( void );

// Overridden:
public:
	// From CMICmnBase
	/* dtor */ virtual ~CMICmdCmdBreakInsert( void );

// Enumerations:
private:
	//++ ===================================================================
	// Details:	The type of break point give in the MI command text.
	//--
	enum BreakPoint_e
	{
		eBreakPoint_Invalid	= 0,
		eBreakPoint_ByFileLine,
		eBreakPoint_ByFileFn,
		eBreakPoint_ByName,
		eBreakPoint_ByAddress,
		eBreakPoint_count,
		eBreakPoint_NotDefineYet
	};

// Attributes:
private:
	bool				m_bBrkPtIsTemp;
	bool				m_bHaveArgOptionThreadGrp;
	CMIUtilString		m_brkName;
	CMIUtilString		m_strArgOptionThreadGrp;
	lldb::SBBreakpoint	m_brkPt;
	const CMIUtilString	m_constStrArgNamedTempBrkPt;
	const CMIUtilString	m_constStrArgNamedHWBrkPt;					// Not handled by *this command
	const CMIUtilString	m_constStrArgNamedPendinfBrkPt;				// Not handled by *this command
	const CMIUtilString	m_constStrArgNamedDisableBrkPt;				// Not handled by *this command
	const CMIUtilString	m_constStrArgNamedTracePt;					// Not handled by *this command
	const CMIUtilString	m_constStrArgNamedConditionalBrkPt;			// Not handled by *this command
	const CMIUtilString	m_constStrArgNamedInoreCnt;					// Not handled by *this command
	const CMIUtilString	m_constStrArgNamedRestrictBrkPtToThreadId;	// Not handled by *this command
	const CMIUtilString	m_constStrArgNamedLocation;
	const CMIUtilString	m_constStrArgNamedThreadGroup;				// Not specified in MI spec but Eclipse gives this option sometimes
};

//++ ============================================================================
// Details:	MI command class. MI commands derived from the command base class.
//			*this class implements MI command "break-delete".
// Gotchas:	None.
// Authors:	Illya Rudkin 11/03/2014.
// Changes:	None.
//--
class CMICmdCmdBreakDelete : public CMICmdBase
{
// Statics:
public:
	// Required by the CMICmdFactory when registering *this commmand
	static CMICmdBase *	CreateSelf( void );

// Methods:
public:
	/* ctor */	CMICmdCmdBreakDelete( void );

// Overridden:
public:
	// From CMICmdInvoker::ICmd
	virtual bool	Execute( void );
	virtual bool	Acknowledge( void );
	virtual bool	ParseArgs( void );

// Overridden:
public:
	// From CMICmnBase
	/* dtor */ virtual ~CMICmdCmdBreakDelete( void );

// Attributes:
private:
	const CMIUtilString	m_constStrArgNamedBrkPt;
	const CMIUtilString	m_constStrArgNamedThreadGrp;	// Not specified in MI spec but Eclipse gives this option
};
