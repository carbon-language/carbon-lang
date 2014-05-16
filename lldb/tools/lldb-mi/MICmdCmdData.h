//===-- MICmdCmdData.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmdCmdData.h
//
// Overview:	CMICmdCmdDataEvaluateExpression	interface.
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

//++ ============================================================================
// Details:	MI command class. MI commands derived from the command base class.
//			*this class implements MI command "data-evaluate-expression".
// Gotchas:	None.
// Authors:	Illya Rudkin 26/03/2014.
// Changes:	None.
//--
class CMICmdCmdDataEvaluateExpression : public CMICmdBase
{
// Statics:
public:
	// Required by the CMICmdFactory when registering *this commmand
	static CMICmdBase *	CreateSelf( void );

// Methods:
public:
	/* ctor */	CMICmdCmdDataEvaluateExpression( void );

// Overridden:
public:
	// From CMICmdInvoker::ICmd
	virtual bool	Execute( void );
	virtual bool	Acknowledge( void );
	virtual bool	ParseArgs( void );

// Overridden:
public:
	// From CMICmnBase
	/* dtor */ virtual ~CMICmdCmdDataEvaluateExpression( void );

// Attributes:
private:
	bool				m_bExpressionValid;		// True = yes is valid, false = not valid
	bool				m_bEvaluatedExpression;	// True = yes is expression evaluated, false = failed
	CMIUtilString		m_strValue;
	CMICmnMIValueTuple	m_miValueTuple;
	bool				m_bCompositeVarType;	// True = yes composite type, false = internal type
	const CMIUtilString	m_constStrArgThread;	// Not specified in MI spec but Eclipse gives this option. Not handled by command.
	const CMIUtilString	m_constStrArgFrame;		// Not specified in MI spec but Eclipse gives this option. Not handled by command.
	const CMIUtilString	m_constStrArgExpr;		
};

