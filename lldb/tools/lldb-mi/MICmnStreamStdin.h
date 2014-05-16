//===-- MIUtilStreamStdin.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MIUtilStreamStdin.h
//
// Overview:	CMICmnStreamStdin interface.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

#pragma once

// In-house headers:
#include "MIUtilString.h"
#include "MIUtilThreadBaseStd.h"
#include "MICmnBase.h"
#include "MIUtilSingletonBase.h"

//++ ============================================================================
// Details:	MI common code class. Used to handle stream data from Stdin.
//			Singleton class using the Visitor pattern. A driver using the interface
//			provide can receive callbacks when a new line of data is received.
//			Each line is determined by a carriage return. Function WaitOnStdin()
//			monitors the Stdin stream.
//			A singleton class.
// Gotchas:	None.
// Authors:	Illya Rudkin 10/02/2014.
// Changes:	None.
//--
class CMICmnStreamStdin
:	public CMICmnBase
,	public CMIUtilThreadActiveObjBase
,	public MI::ISingleton< CMICmnStreamStdin >
{
	// Give singleton access to private constructors
	friend MI::ISingleton< CMICmnStreamStdin >;

// Class:
public:
	//++
	// Description: Visitor pattern. Driver(s) use this interface to get a callback
	//				on each new line of data received from stdin.
	//--
	class IStreamStdin
	{
	public:
		virtual bool ReadLine( const CMIUtilString & vStdInBuffer, bool & vbYesExit ) = 0;

		/* dtor */ virtual ~IStreamStdin( void ) {};
	};
		
// Methods:
public:
	bool	Initialize( void );
	bool	Shutdown( void );
	//
	const CMIUtilString &	GetPrompt( void ) const;
	bool					SetPrompt( const CMIUtilString & vNewPrompt );
	void					SetEnablePrompt( const bool vbYes );
	bool					GetEnablePrompt( void ) const;
	void					SetCtrlCHit( void );
	bool					SetVisitor( IStreamStdin & vrVisitor );
	
// Overridden:
public:
	// From CMIUtilThreadActiveObjBase
	virtual const CMIUtilString & ThreadGetName( void ) const;
	
// Overridden:
protected:
	// From CMIUtilThreadActiveObjBase
	virtual bool ThreadRun( bool & vrIsAlive );
	virtual bool ThreadFinish( void );						// Let this thread clean up after itself

// Methods:
private:
	/* ctor */	CMICmnStreamStdin( void );
	/* ctor */	CMICmnStreamStdin( const CMICmnStreamStdin & );
	void		operator=( const CMICmnStreamStdin & );
	
	bool			MonitorStdin( bool & vrbYesExit );
	const MIchar *	ReadLine( CMIUtilString & vwErrMsg );
	bool			InputAvailable( bool & vbAvail ) const;					// Bytes are available on stdin

// Overridden:
private:
	// From CMICmnBase
	/* dtor */ virtual ~CMICmnStreamStdin( void );

// Attributes:
private:
	const CMIUtilString	m_constStrThisThreadname;
	const MIuint		m_constBufferSize;
	MIchar *			m_pCmdBuffer;
	IStreamStdin *		m_pVisitor;
	CMIUtilString		m_strPromptCurrent;		// Command line prompt as shown to the user
	volatile bool		m_bKeyCtrlCHit;			// True = User hit Ctrl-C, false = has not yet
	FILE *				m_pStdin;
	bool				m_bShowPrompt;			// True = Yes prompt is shown/output to the user (stdout), false = no prompt
	bool				m_bRedrawPrompt;		// True = Prompt needs to be redrawn
	MIchar *			m_pStdinBuffer;			// Custom buffer to store std input
};

