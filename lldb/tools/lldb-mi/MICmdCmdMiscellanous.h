//===-- MICmdCmdMiscellanous.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmdCmdMiscellanous.h
//
// Overview:	CMICmdCmdGdbSet					interface.
//				CMICmdCmdGdbExit				interface.
//				CMICmdCmdListThreadGroups		interface.
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

// In-house headers:
#include "MICmdBase.h"
#include "MICmnMIValueTuple.h"
#include "MICmnMIValueList.h"

//++ ============================================================================
// Details:	MI command class. MI commands derived from the command base class.
//			*this class implements MI command "gdb-set".
//			This command does not follow the MI documentation exactly.
// Gotchas:	None.
// Authors:	Illya Rudkin 03/03/2014.
// Changes:	None.
//--
class CMICmdCmdGdbSet : public CMICmdBase
{
// Statics:
public:
	// Required by the CMICmdFactory when registering *this commmand
	static CMICmdBase *	CreateSelf( void );

// Methods:
public:
	/* ctor */	CMICmdCmdGdbSet( void );

// Overridden:
public:
	// From CMICmdInvoker::ICmd
	virtual bool	Execute( void );
	virtual bool	Acknowledge( void );

// Overridden:
public:
	// From CMICmnBase
	/* dtor */ virtual ~CMICmdCmdGdbSet( void );
};

//++ ============================================================================
// Details:	MI command class. MI commands derived from the command base class.
//			*this class implements MI command "gdb-exit".
// Gotchas:	None.
// Authors:	Illya Rudkin 04/03/2014.
// Changes:	None.
//--
class CMICmdCmdGdbExit : public CMICmdBase
{
// Statics:
public:
	// Required by the CMICmdFactory when registering *this commmand
	static CMICmdBase *	CreateSelf( void );

// Methods:
public:
	/* ctor */	CMICmdCmdGdbExit( void );

// Overridden:
public:
	// From CMICmdInvoker::ICmd
	virtual bool	Execute( void );
	virtual bool	Acknowledge( void );

// Overridden:
public:
	// From CMICmnBase
	/* dtor */ virtual ~CMICmdCmdGdbExit( void );
};

//++ ============================================================================
// Details:	MI command class. MI commands derived from the command base class.
//			*this class implements MI command "list-thread-groups".
//			This command does not follow the MI documentation exactly.
//			http://sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI-Miscellaneous-Commands.html#GDB_002fMI-Miscellaneous-Commands
// Gotchas:	None.
// Authors:	Illya Rudkin 06/03/2014.
// Changes:	None.
//--
class CMICmdCmdListThreadGroups : public CMICmdBase
{
// Statics:
public:
	// Required by the CMICmdFactory when registering *this commmand
	static CMICmdBase *	CreateSelf( void );

// Methods:
public:
	/* ctor */	CMICmdCmdListThreadGroups( void );

// Overridden:
public:
	// From CMICmdInvoker::ICmd
	virtual bool	Execute( void );
	virtual bool	Acknowledge( void );
	virtual bool	ParseArgs( void );

// Overridden:
public:
	// From CMICmnBase
	/* dtor */ virtual ~CMICmdCmdListThreadGroups( void );

// Typedefs:
private:
	typedef std::vector< CMICmnMIValueTuple >	VecMIValueTuple_t;

// Attributes:
private:
	bool				m_bIsI1;					// True = Yes command argument equal "i1", false = no match
	bool				m_bHaveArgOption;			// True = Yes "--available" present, false = not found
	bool				m_bHaveArgRecurse;			// True = Yes command argument "--recurse", false = no found
	VecMIValueTuple_t	m_vecMIValueTuple;
	const CMIUtilString	m_constStrArgNamedAvailable;
	const CMIUtilString	m_constStrArgNamedRecurse;
	const CMIUtilString	m_constStrArgNamedGroup;
	const CMIUtilString	m_constStrArgNamedThreadGroup;
};