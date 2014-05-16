//===-- MIDriver.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MIDriver.h
//
// Overview:	CMIDriver interface.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

#pragma once

// Third party headers
#include <queue>

// In-house headers:
#include "MICmnConfig.h"
#include "MICmnBase.h"
#include "MIDriverBase.h"
#include "MIDriverMgr.h"
#include "MICmnStreamStdin.h"
#include "MICmdData.h"
#include "MIUtilSingletonBase.h"

// Declarations:
class CMICmnLLDBDebugger;
class CMICmnStreamStdout;

//++ ============================================================================
// Details:	MI driver implementation class. A singleton class derived from
//			LLDB SBBroadcaster class. Register the instance of *this class with
//			the CMIDriverMgr. The CMIDriverMgr sets the driver(s) of to start
//			work depending on the one selected to work. A driver can if not able
//			to handle an instruction or 'command' can pass that command onto 
//			another driver object registered with the Driver Manager.
// Gotchas:	None.
// Authors:	Illya Rudkin 29/01/2014.
// Changes:	None.
//--
class CMIDriver 
:	public CMICmnBase
,	public CMIDriverMgr::IDriver
,	public CMIDriverBase
,	public CMICmnStreamStdin::IStreamStdin
,	public MI::ISingleton< CMIDriver >
{
	friend class MI::ISingleton< CMIDriver >;

// Methods:
public:
	// MI system
	bool	Initialize( void );
	bool	Shutdown( void );
	
	// MI information about itself
	const CMIUtilString &	GetAppNameShort( void ) const;
	const CMIUtilString &	GetAppNameLong( void ) const;
	const CMIUtilString &	GetVersionDescription( void ) const;
		
	// MI do work
	bool	WriteMessageToLog( const CMIUtilString & vMessage );
	bool	SetEnableFallThru( const bool vbYes );
	bool	GetEnableFallThru( void ) const;

// Overridden:
public:
	// From CMIDriverMgr::IDriver
	virtual bool					DoInitialize( void );
	virtual bool					DoShutdown( void );
	virtual bool					DoMainLoop( void );
	virtual void					DoResizeWindow( const uint32_t vWindowSizeWsCol );
	virtual lldb::SBError			DoParseArgs( const int argc, const char * argv[], FILE * vpStdOut, bool & vwbExiting );
	virtual CMIUtilString 			GetError( void ) const;
	virtual const CMIUtilString & 	GetName( void ) const;
	virtual lldb::SBDebugger &		GetTheDebugger( void );	
	virtual bool					GetDriverIsGDBMICompatibleDriver( void ) const;
	virtual bool					SetId( const CMIUtilString & vId );
	virtual const CMIUtilString &	GetId( void ) const;
	// From CMIDriverBase
	virtual void					SetExitApplicationFlag( void );
	virtual bool					DoFallThruToAnotherDriver( const CMIUtilString & vCmd, CMIUtilString & vwErrMsg );
	virtual bool					SetDriverToFallThruTo( const CMIDriverBase & vrOtherDriver );
	virtual FILE *					GetStdin( void ) const;		
	virtual FILE *					GetStdout( void ) const;	
	virtual FILE *					GetStderr( void ) const;	
	virtual const CMIUtilString & 	GetDriverName( void ) const;
	virtual const CMIUtilString &	GetDriverId( void ) const;
	// From CMICmnStreamStdin
	virtual bool ReadLine( const CMIUtilString & vStdInBuffer, bool & vrbYesExit );

// Typedefs:
private:
	typedef std::queue< CMIUtilString >	QueueStdinLine_t;

// Methods:
private:
	/* ctor */	CMIDriver( void );
	/* ctor */	CMIDriver( const CMIDriver & );
	void		operator=( const CMIDriver & );
	
	lldb::SBError	ParseArgs( const int argc, const char * argv[], FILE * vpStdOut, bool & vwbExiting );
	bool			ReadStdinLineQueue( void );
	bool			DoAppQuit( void );
	bool			InterpretCommand( const CMIUtilString & vTextLine, bool & vwbCmdYesValid );
	bool			InterpretCommandFallThruDriver( const CMIUtilString & vTextLine, bool & vwbCmdYesValid );
	bool			ExecuteCommand( const SMICmdData & vCmdData );
	bool			StartWorkerThreads( void );
	bool			StopWorkerThreads( void );

// Overridden:
private:
	// From CMICmnBase
	/* dtor */ virtual ~CMIDriver( void );

// Attributes:
private:
	static const CMIUtilString	ms_constAppNameShort;
	static const CMIUtilString	ms_constAppNameLong;
	static const CMIUtilString	ms_constMIVersion;
	//
	bool					m_bFallThruToOtherDriverEnabled;	// True = yes fall through, false = do not pass on command
	CMIUtilThreadMutex		m_threadMutex;
	QueueStdinLine_t		m_queueStdinLine;					// Producer = stdin monitor, consumer = *this driver 
	bool					m_bDriverIsExiting;					// True = yes, driver told to quit, false = continue working
	void *					m_handleMainThread;					// *this driver is run by the main thread
	CMICmnStreamStdin &		m_rStdin;	
	CMICmnLLDBDebugger &	m_rLldbDebugger;
	CMICmnStreamStdout &	m_rStdOut;
};
