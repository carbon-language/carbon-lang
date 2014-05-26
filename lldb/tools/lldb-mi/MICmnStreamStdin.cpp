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

// Third Party Headers:
#if !defined( _MSC_VER )
#include <sys/select.h>
#include <termios.h>
#else
#include <stdio.h>
#include <Windows.h>
#include <io.h>
#include <conio.h>
#endif // !defined( _MSC_VER )
#include <string.h> // For std::strerror()

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
,	m_constBufferSize( 1024 )
,	m_pCmdBuffer( nullptr )
,	m_pVisitor( nullptr )
,	m_strPromptCurrent( "(gdb)" )
,	m_bKeyCtrlCHit( false )
,	m_pStdin( nullptr )
,	m_bShowPrompt( false )
,	m_bRedrawPrompt( true )
,	m_pStdinBuffer( nullptr )
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
 
	// Note initialization order is important here as some resources depend on previous
	MI::ModuleInit< CMICmnLog >         ( IDS_MI_INIT_ERR_LOG      , bOk, errMsg );
	MI::ModuleInit< CMICmnResources >   ( IDS_MI_INIT_ERR_RESOURCES, bOk, errMsg );
	MI::ModuleInit< CMICmnThreadMgrStd >( IDS_MI_INIT_ERR_THREADMGR, bOk, errMsg );

	// Other resources required
	if( bOk )
	{
		m_pCmdBuffer = new MIchar[ m_constBufferSize ];
		m_bKeyCtrlCHit = false; // Reset
		m_pStdin = stdin;

#if MICONFIG_CREATE_OWN_STDIN_BUFFER
		// Give stdinput a user defined buffer
		m_pStdinBuffer = new char[ 1024 ];
		::setbuf( stdin, m_pStdinBuffer );
#endif // MICONFIG_CREATE_OWN_STDIN_BUFFER
	}

	// Clear error indicators for std input
	clearerr( stdin );

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

	if( m_pCmdBuffer != nullptr )
	{
		delete [] m_pCmdBuffer;
		m_pCmdBuffer = nullptr;
	}
	m_pVisitor = nullptr;
	m_bKeyCtrlCHit = false;
	m_pStdin = nullptr;

#if MICONFIG_CREATE_OWN_STDIN_BUFFER
	if ( m_pStdinBuffer )
		delete [] m_pStdinBuffer;
	m_pStdinBuffer = nullptr;
#endif // MICONFIG_CREATE_OWN_STDIN_BUFFER

	// Note shutdown order is important here 	
	MI::ModuleShutdown< CMICmnThreadMgrStd >( IDS_MI_INIT_ERR_THREADMGR, bOk, errMsg );
	MI::ModuleShutdown< CMICmnResources >   ( IDS_MI_INIT_ERR_RESOURCES, bOk, errMsg );
	MI::ModuleShutdown< CMICmnLog >         ( IDS_MI_INIT_ERR_LOG      , bOk, errMsg );

	if( !bOk )
	{
		SetErrorDescriptionn( MIRSRC( IDS_MI_SHUTDOWN_ERR ), errMsg.c_str() );
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
// Details:	Retreive the command line prompt text currently being used.
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
//			buffer is not exceeded the data is passed to the vistor.
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
//			stdout. Disable it when this may interfer with the client reading stdout as
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
//			stdout. Disable it when this may interfer with the client reading stdout as
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
bool CMICmnStreamStdin::InputAvailable( bool & vwbAvail ) const
{
// Windows method to check how many bytes are in stdin
#ifdef _MSC_VER
	// Get a windows handle to std input stream
	HANDLE handle = ::GetStdHandle( STD_INPUT_HANDLE );
	DWORD nBytesWaiting = 0;
	
	// If running in a terminal use _kbhit()
	if( ::_isatty( ::fileno( stdin ) ) )
		nBytesWaiting = ::_kbhit();
	else
	{
		// Ask how many bytes are available
		if( ::PeekNamedPipe( handle, nullptr, 0, nullptr, &nBytesWaiting, nullptr ) == FALSE )
		{
			// This can occur when the client i.e. Eclipse closes the stdin stream 'cause it deems its work is finished
			// for that debug session. May be we should be handling SIGKILL somehow?
			const CMIUtilString osErrMsg( CMIUtilSystemWindows().GetOSLastError().StripCRAll() );
			SetErrorDescription( CMIUtilString::Format( MIRSRC( IDS_STDIN_ERR_CHKING_BYTE_AVAILABLE ), osErrMsg.c_str() ) );
			return MIstatus::failure;
		}
	}

	// Return the number of bytes waiting
	vwbAvail = (nBytesWaiting > 0);

	return MIstatus::success;

// Unix method to check how many bytes are in stdin
#else
/* AD: disable while porting to linux
	static const int STDIN = 0;
    static bool bInitialized = false;

    if( !bInitialized )
	{
        // Use termios to turn off line buffering
        ::termios term;
        ::tcgetattr( STDIN, &term );
        ::term.c_lflag &= ~ICANON;
        ::tcsetattr( STDIN, TCSANOW, &term );
        ::setbuf( stdin, NULL );
        bInitialized = true;
    }

    int nBytesWaiting;
    ::ioctl( STDIN, FIONREAD, &nBytesWaiting );
    vwbAvail = (nBytesWaiting > 0);

	return MIstatus::success;
*/
	return MIstatus::success;
#endif // _MSC_VER
}

//++ ------------------------------------------------------------------------------------
// Details:	The monitoring on new line data calls back to the visitor object registered 
//			with *this stdin monitoring. The monitoring to stops when the visitor returns 
//			true for bYesExit flag. Errors output to log file.
//			This function runs in the thread "MI stdin monitor".
// Type:	Method.
//			vrbYesExit		- (W) True = yes exit stdin monitoring, false = continue monitor.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnStreamStdin::MonitorStdin( bool & vrbYesExit )
{
	if( m_bShowPrompt )
	{
		CMICmnStreamStdout & rStdoutMan = CMICmnStreamStdout::Instance();
		rStdoutMan.WriteMIResponse( m_strPromptCurrent.c_str() );
		m_bRedrawPrompt = false;
	}

	if( m_bKeyCtrlCHit )
	{
		vrbYesExit = true;
		CMIDriver::Instance().SetExitApplicationFlag();
		return MIstatus::failure;
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
		vrbYesExit = true;
		CMIDriver::Instance().SetExitApplicationFlag();
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
			CMICmnStreamStdout::Instance().Write( pText );
		}
		return MIstatus::failure;
	}

	// We have text so send it off to the visitor
	bool bOk = MIstatus::success;
	if( m_pVisitor != nullptr )
	{
		bOk = m_pVisitor->ReadLine( CMIUtilString( pText ), vrbYesExit );
		m_bRedrawPrompt = true;
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
	vwErrMsg.clear();
	
	// Read user input
	const MIchar * pText = ::fgets( &m_pCmdBuffer[ 0 ], m_constBufferSize, stdin );
	if( pText == nullptr )
	{
		if( ::ferror( m_pStdin ) != 0 )
			vwErrMsg = ::strerror( errno );
		return nullptr;
	}
	
	// Strip off new line characters
	for( MIchar * pI = m_pCmdBuffer; *pI != '\0'; pI++ )
	{
		if( (*pI == '\n') || (*pI == '\r') )
		{
			*pI = '\0';
			break;
		}
	}

	return pText;
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
// Type:	Overidden.
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
const CMIUtilString &CMICmnStreamStdin::ThreadGetName( void ) const
{
	return m_constStrThisThreadname;
}
