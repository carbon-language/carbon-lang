//===-- MIUtilStreamStdin.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MIUtilStreamStdin.cpp
//
// Overview:	CMICmnStreamStdin implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// In-house headers:
#include "MICmnStreamStdin.h"
#include "MICmnStreamStdout.h"
#include "MICmnResources.h"
#include "MICmnLog.h"
#include "MICmnThreadMgrStd.h"
#include "MIUtilSingletonHelper.h"
#include "MIDriver.h"
#if defined( _MSC_VER )
#include "MIUtilSystemWindows.h"
#include "MICmnStreamStdinWindows.h"
#else
#include "MICmnStreamStdinLinux.h"
#endif // defined( _MSC_VER )

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnStreamStdin constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmnStreamStdin::CMICmnStreamStdin( void )
:	m_constStrThisThreadname( "MI stdin thread" )
,	m_pVisitor( nullptr )
,	m_strPromptCurrent( "(gdb)" )
,	m_bKeyCtrlCHit( false )
,	m_bShowPrompt( false )
,	m_bRedrawPrompt( true )
,	m_pStdinReadHandler( nullptr )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnStreamStdin destructor.
// Type:	Overridable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmnStreamStdin::~CMICmnStreamStdin( void )
{
	Shutdown();
}

//++ ------------------------------------------------------------------------------------
// Details:	Initialize resources for *this Stdin stream.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdin::Initialize( void )
{
	m_clientUsageRefCnt++;

	if( m_bInitialized )
		return MIstatus::success;

	bool bOk = MIstatus::success;
	CMIUtilString errMsg;

	// Note initialisation order is important here as some resources depend on previous
	MI::ModuleInit< CMICmnLog >			( IDS_MI_INIT_ERR_LOG      , bOk, errMsg );
	MI::ModuleInit< CMICmnResources >	( IDS_MI_INIT_ERR_RESOURCES, bOk, errMsg );
	MI::ModuleInit< CMICmnThreadMgrStd >( IDS_MI_INIT_ERR_THREADMGR, bOk, errMsg );
#ifdef _MSC_VER
	MI::ModuleInit< CMICmnStreamStdinWindows >( IDS_MI_INIT_ERR_OS_STDIN_HANDLER, bOk, errMsg );
	bOk = bOk && SetOSStdinHandler( CMICmnStreamStdinWindows::Instance() );
#else
	MI::ModuleInit< CMICmnStreamStdinLinux >( IDS_MI_INIT_ERR_OS_STDIN_HANDLER, bOk, errMsg );
	bOk = bOk && SetOSStdinHandler( CMICmnStreamStdinLinux::Instance() );
#endif // ( _MSC_VER )

	// The OS specific stdin stream handler must be set before *this class initialises
	if( bOk && m_pStdinReadHandler == nullptr )
	{
		CMIUtilString strInitError( CMIUtilString::Format( MIRSRC( IDS_MI_INIT_ERR_STREAMSTDIN_OSHANDLER ), errMsg.c_str() ) );
		SetErrorDescription( strInitError );
		return MIstatus::failure;
	}

	// Other resources required
	if( bOk )
	{
		m_bKeyCtrlCHit = false; // Reset
	}

	m_bInitialized = bOk;

	if( !bOk )
	{
		CMIUtilString strInitError( CMIUtilString::Format( MIRSRC( IDS_MI_INIT_ERR_STREAMSTDIN ), errMsg.c_str() ) );
		SetErrorDescription( strInitError );
		return MIstatus::failure;
	}

	return  MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Release resources for *this Stdin stream.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdin::Shutdown( void )
{
	if( --m_clientUsageRefCnt > 0 )
		return MIstatus::success;
	
	if( !m_bInitialized )
		return MIstatus::success;

	m_bInitialized = false;

	ClrErrorDescription();

	bool bOk = MIstatus::success;
	CMIUtilString errMsg;

	m_pVisitor = nullptr;
	m_bKeyCtrlCHit = false;

	// Note shutdown order is important here 	
#ifndef _MSC_VER
	MI::ModuleShutdown< CMICmnStreamStdinLinux >( IDS_MI_SHTDWN_ERR_OS_STDIN_HANDLER, bOk, errMsg );
#else
	MI::ModuleShutdown< CMICmnStreamStdinWindows >( IDS_MI_SHTDWN_ERR_OS_STDIN_HANDLER, bOk, errMsg );
#endif // ( _MSC_VER )
	MI::ModuleShutdown< CMICmnThreadMgrStd >( IDS_MI_SHTDWN_ERR_THREADMGR, bOk, errMsg );
	MI::ModuleShutdown< CMICmnResources >   ( IDE_MI_SHTDWN_ERR_RESOURCES, bOk, errMsg );
	MI::ModuleShutdown< CMICmnLog >         ( IDS_MI_SHTDWN_ERR_LOG      , bOk, errMsg );

	if( !bOk )
	{
		SetErrorDescriptionn( MIRSRC( IDE_MI_SHTDWN_ERR_STREAMSTDIN ), errMsg.c_str() );
	}

	return MIstatus::success;
}	

//++ ------------------------------------------------------------------------------------
// Details:	Validate and set the text that forms the prompt on the command line.
// Type:	Method.
// Args:	vNewPrompt	- (R) Text description.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdin::SetPrompt( const CMIUtilString & vNewPrompt )
{
	if( vNewPrompt.empty() )
	{
		const CMIUtilString msg( CMIUtilString::Format( MIRSRC( IDS_STDIN_ERR_INVALID_PROMPT), vNewPrompt.c_str() ) );
		CMICmnStreamStdout::Instance().Write( msg );
		return MIstatus::failure;
	}

	m_strPromptCurrent = vNewPrompt;

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the command line prompt text currently being used.
// Type:	Method.
// Args:	None.
// Return:	const CMIUtilString & - Functional failed.
// Throws:	None.
//--
const CMIUtilString & CMICmnStreamStdin::GetPrompt( void ) const
{
	return m_strPromptCurrent;
}

//++ ------------------------------------------------------------------------------------
// Details:	Wait on input from stream Stdin. On each line of input received it is 
//			validated and providing there are no errors on the stream or the input
//			buffer is not exceeded the data is passed to the visitor.
// Type:	Method.
// Args:	vrVisitor	- (W) A client deriver callback.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdin::SetVisitor( IStreamStdin & vrVisitor )
{
	m_pVisitor = &vrVisitor;
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Set whether to display optional command line prompt. The prompt is output to
//			stdout. Disable it when this may interfere with the client reading stdout as
//			input and it tries to interpret the prompt text to.
// Type:	Method.
// Args:	vbYes	- (R) True = Yes prompt is shown/output to the user (stdout), false = no prompt.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
void CMICmnStreamStdin::SetEnablePrompt( const bool vbYes )
{
	m_bShowPrompt = vbYes;
}

//++ ------------------------------------------------------------------------------------
// Details: Get whether to display optional command line prompt. The prompt is output to
//			stdout. Disable it when this may interfere with the client reading stdout as
//			input and it tries to interpret the prompt text to.
// Type:	Method.
// Args:	None. 
// Return:	bool - True = Yes prompt is shown/output to the user (stdout), false = no prompt.
// Throws:	None.
//--
bool CMICmnStreamStdin::GetEnablePrompt( void ) const
{
	return m_bShowPrompt;
}

//++ ------------------------------------------------------------------------------------
// Details:	Determine if stdin has any characters present in its buffer.
// Type:	Method.
// Args:	vwbAvail	- (W) True = There is chars available, false = nothing there.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdin::InputAvailable( bool & vwbAvail )
{
	return m_pStdinReadHandler->InputAvailable( vwbAvail );
}

//++ ------------------------------------------------------------------------------------
// Details:	The monitoring on new line data calls back to the visitor object registered 
//			with *this stdin monitoring. The monitoring to stops when the visitor returns 
//			true for bYesExit flag. Errors output to log file.
//			This function runs in the thread "MI stdin monitor".
// Type:	Method.
//			vrwbYesAlive	- (W) False = yes exit stdin monitoring, true = continue monitor.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdin::MonitorStdin( bool & vrwbYesAlive )
{
	if( m_bShowPrompt )
	{
		CMICmnStreamStdout & rStdoutMan = CMICmnStreamStdout::Instance();
		rStdoutMan.WriteMIResponse( m_strPromptCurrent.c_str() );
		m_bRedrawPrompt = false;
	}

	// CODETAG_DEBUG_SESSION_RUNNING_PROG_RECEIVED_SIGINT_PAUSE_PROGRAM
	if( m_bKeyCtrlCHit )
	{
		CMIDriver & rMIDriver = CMIDriver::Instance();
		rMIDriver.SetExitApplicationFlag( false );
		if( rMIDriver.GetExitApplicationFlag() )
		{
			vrwbYesAlive = false;
			return MIstatus::success;
		}

		// Reset - the MI Driver received SIGINT during a running debug programm session
		m_bKeyCtrlCHit = false;
	}

#if MICONFIG_POLL_FOR_STD_IN
	bool bAvail = true;
	// Check if there is stdin available
	if( InputAvailable( bAvail ) )
	{
		// Early exit when there is no input
		if( !bAvail )
			return MIstatus::success;
	}
	else
	{
		vrwbYesAlive = false;
		CMIDriver::Instance().SetExitApplicationFlag( true );
		return MIstatus::failure;
	}
#endif // MICONFIG_POLL_FOR_STD_IN

	// Read a line from std input
	CMIUtilString stdinErrMsg;
	const MIchar * pText = ReadLine( stdinErrMsg );

	// Did something go wrong
	const bool bHaveError( !stdinErrMsg.empty() );
	if( (pText == nullptr) || bHaveError )
	{
		if( bHaveError )
		{
			CMICmnStreamStdout::Instance().Write( stdinErrMsg );
		}
		return MIstatus::failure;
	}

	// We have text so send it off to the visitor
	bool bOk = MIstatus::success;
	if( m_pVisitor != nullptr )
	{
		bool bYesExit = false;
		bOk = m_pVisitor->ReadLine( CMIUtilString( pText ), bYesExit );
		m_bRedrawPrompt = true;
		vrwbYesAlive = !bYesExit;
	}
	
	return bOk;
}

//++ ------------------------------------------------------------------------------------
// Details:	Wait on new line of data from stdin stream (completed by '\n' or '\r').
// Type:	Method.
// Args:	vwErrMsg	- (W) Empty string ok or error description.			
// Return:	MIchar * - text buffer pointer or NULL on failure.
// Throws:	None.
//--
const MIchar * CMICmnStreamStdin::ReadLine( CMIUtilString & vwErrMsg )
{
	return m_pStdinReadHandler->ReadLine( vwErrMsg );
}

//++ ------------------------------------------------------------------------------------
// Details:	Inform *this stream that the user hit Control-C key to exit.
//			The function is normally called by the SIGINT signal in sigint_handler() to 
//			simulate kill app from the client.
//			This function is called by a Kernel thread.
// Type:	Method.
// Args:	None.			
// Return:	None.
// Throws:	None.
//--
void CMICmnStreamStdin::SetCtrlCHit( void )
{
	CMIUtilThreadLock lock( m_mutex );
	m_bKeyCtrlCHit = true;
}

//++ ------------------------------------------------------------------------------------
// Details:	The main worker method for this thread.
// Type:	Overridden.
// Args:	vrbIsAlive	= (W) True = *this thread is working, false = thread has exited.			
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdin::ThreadRun( bool & vrbIsAlive )
{
	return MonitorStdin( vrbIsAlive );
}

//++ ------------------------------------------------------------------------------------
// Details:	Let this thread clean up after itself.
// Type:	Overridden.
// Args:	None.			
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdin::ThreadFinish( void )
{
	// Do nothing - override to implement
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve *this thread object's name.
// Type:	Overridden.
// Args:	None.			
// Return:	CMIUtilString & - Text.
// Throws:	None.
//--
const CMIUtilString & CMICmnStreamStdin::ThreadGetName( void ) const
{
	return m_constStrThisThreadname;
}

//++ ------------------------------------------------------------------------------------
// Details:	Mandatory set the OS specific stream stdin handler. *this class utilises the
//			handler to read data from the stdin stream and put into a queue for the 
//			driver to read when able.
// Type:	Method.
// Args:	None.			
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdin::SetOSStdinHandler( IOSStdinHandler & vrHandler )
{
	m_pStdinReadHandler = &vrHandler;

	return MIstatus::success;
}
